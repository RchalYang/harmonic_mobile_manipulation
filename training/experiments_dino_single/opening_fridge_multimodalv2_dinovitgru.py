from typing import Any, Sequence, Union

try:
    from typing import final
except ImportError:
    from typing_extensions import final


import gym
import numpy as np
import torch
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, RGBSensor
from allenact.utils.experiment_utils import Builder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
)
from allenact.utils.system import get_logger

# from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from training.tasks.opening_fridge import (
    OpeningfridgeGraspTask,
    OpeningfridgeGraspTwoStageTask,
    OpeningfridgeGraspIterativeTask,
    OpeningfridgeNWGraspTask,
    OpeningfridgeNWGraspTwoStageTask,
    OpeningfridgeNWGraspIterativeTask,
    OpeningfridgeTaskSampler,
)
from training import cfg

from ..sensors.vision import RGBSensorStretchControllerNavigation
from ..sensors.vision import RGBSensorStretchControllerManipulation

from ..sensors.vision import RGBSensorStretchControllerNavigationHist
from ..sensors.vision import RGBSensorStretchControllerManipulationHist

from training.sensors.proprioception import (
    StretchArmHistSensorThorController,
    StretchArmSensorV2ThorController
)
from .opening_fridge_base import OpeningfridgeBaseConfig
from ..models.ncamera import TaskIdSensor
from ..models.ncamera_dino import (
    VitMultiModalNCameraActorCritic,
    VitMultiModalPrevActNCameraActorCritic,
    VitMultiModalPrevActV2NCameraActorCritic,
    VitMultiModalHistNCameraActorCritic,
    VitVisOnlyPrevActV2NCameraActorCritic
)
from utils.utils import log_ac_return, ForkedPdb
from training.preprocessors.dinovit_aug_preprocessor import DinoViTAugPreprocessor, DinoViTAugHistoryPreprocessor

from allenact.base_abstractions.task import TaskSampler
from training.utils.types import RewardConfig, TaskSamplerArgs

def get_opening_fridge_task(task_string):
    return eval(task_string)

def get_RGB_sensor(sensor_type, cfg, uuid):
    # sensor_type = eval(sensor_name)
    if sensor_type is RGBSensorStretchControllerNavigationHist or sensor_type is RGBSensorStretchControllerManipulationHist:
        return sensor_type(
            height=cfg.model.image_size,
            width=cfg.model.image_size,
            use_resnet_normalization=False,
            uuid=uuid,
            output_channels=3,
        )
    elif sensor_type is RGBSensorStretchControllerNavigation or sensor_type is RGBSensorStretchControllerManipulation:
        return sensor_type(
            height=cfg.model.image_size,
            width=cfg.model.image_size,
            use_resnet_normalization=False,
            uuid=uuid,
            output_channels=3,
        )

def get_dino_preprocessor(
    preprocessor_type, cfg, sensor, output_uuid,
):
    # preprocessor_type = eval(preprocessor_name)
    if preprocessor_type is DinoViTAugHistoryPreprocessor:
        return preprocessor_type(
            augment_obs=not (cfg.eval and cfg.evaluation.no_eval_aug),
            rgb_input_uuid=sensor.uuid,
            # clip_model_type=cfg.model.clip.model_type,
            dino_model_type=cfg.model.dino.model_type,
            flatten=False,
            output_uuid=output_uuid,
            his_len=cfg.model.his_len
        )
    elif preprocessor_type is DinoViTAugPreprocessor:
        return preprocessor_type(
            augment_obs=not (cfg.eval and cfg.evaluation.no_eval_aug),
            rgb_input_uuid=sensor.uuid,
            # clip_model_type=cfg.model.clip.model_type,
            dino_model_type=cfg.model.dino.model_type,
            flatten=False,
            output_uuid=output_uuid
        )

def get_model(
    model_type_name, cfg, task_type, kwargs, vit_preprocessor_uuids
):
    model_type = eval(model_type_name)
    model = model_type(
        action_space=gym.spaces.Box(
            low=-1, high=1,
            shape=(task_type.continuous_action_dim(),)
        ),
        observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
        vit_preprocessor_uuids=vit_preprocessor_uuids,
        stretch_proprio_uuid="stretch_arm_sensor",
        hidden_size=512,
        # goal_dims=32,
        action_embed_size=task_type.continuous_action_dim(),
        # add_prev_actions=cfg.model.add_prev_actions_embedding,
        visualize=cfg.eval, # TODO set this more explicitly/finer-grained later. Eval decent proxy for now,
        init_std=cfg.model.init_std,
        cfg=cfg
    )
    return model


def get_proprio_sensor(proprio_type, cfg):
    if proprio_type is StretchArmHistSensorThorController:
        return proprio_type(
            uuid="stretch_arm_sensor", his_len=cfg.model.his_len, domain_randomization=not (cfg.eval and cfg.evaluation.no_eval_aug)
        )
    elif proprio_type is StretchArmSensorV2ThorController:
        return proprio_type(
            uuid="stretch_arm_sensor", domain_randomization=not (cfg.eval and cfg.evaluation.no_eval_aug)
        )


class OpeningfridgeGraspTwoStageFullMultiModalDinoViTPPOExperimentConfig(OpeningfridgeBaseConfig):
    """An Object Navigation experiment configuration with RGB input."""

    # OPENING_fridge_TASK_TYPE = OpeningfridgeGraspTwoStageFullTask
    OPENING_fridge_TASK_TYPE = eval(cfg.task.name)
    NAVIGATION_CAM_SENSOR_TYPE = RGBSensorStretchControllerNavigationHist if cfg.sensor.enable_history else RGBSensorStretchControllerNavigation
    MANIPULATION_CAM_SENSOR_TYPE = RGBSensorStretchControllerManipulationHist if cfg.sensor.enable_history else RGBSensorStretchControllerManipulation
    PROPRIO_SENSOR_TYPE = StretchArmHistSensorThorController if cfg.sensor.enable_history else StretchArmSensorV2ThorController

    DINO_MODEL = cfg.model.dino.model_type

    DINO_PREPROCESSOR_TYPE = DinoViTAugHistoryPreprocessor if cfg.sensor.enable_history else DinoViTAugPreprocessor

    MODEL_TYPE = cfg.model.model_type_name

    SENSORS = [
        get_RGB_sensor(NAVIGATION_CAM_SENSOR_TYPE, cfg, "rgb_lowres_navigation"),
        get_RGB_sensor(MANIPULATION_CAM_SENSOR_TYPE, cfg, "rgb_lowres_manipulation"),
        get_proprio_sensor(PROPRIO_SENSOR_TYPE, cfg),
        TaskIdSensor(),
    ]


    vit_preprocessor_uuids = [
        "rgb_navigation_dino_vit",
        "rgb_manipulation_dino_vit"
    ]

    @classmethod
    def tag(cls):
        return "Openingfridge-MultiModalV2-Augmented-DinoViTGRU-DDPPO"

    # @classmethod
    # @final
    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        rgb_sensor_navigation = next((
            s for s in self.SENSORS if isinstance(s, RGBSensorStretchControllerNavigationHist) or isinstance(s, RGBSensorStretchControllerNavigation)),
            None
        )
        if rgb_sensor_navigation is not None:
            preprocessors.append(
                get_dino_preprocessor(self.DINO_PREPROCESSOR_TYPE, cfg, rgb_sensor_navigation, "rgb_navigation_dino_vit")
            )

        rgb_sensor_manipulation = next(
            (s for s in self.SENSORS if isinstance(s, RGBSensorStretchControllerManipulationHist) or isinstance(s, RGBSensorStretchControllerManipulation)),
            None
        )
        if rgb_sensor_manipulation is not None:
            preprocessors.append(
                get_dino_preprocessor(self.DINO_PREPROCESSOR_TYPE, cfg, rgb_sensor_manipulation, "rgb_manipulation_dino_vit")
            )

        return preprocessors

    @classmethod
    def make_sampler_fn(
        cls, task_sampler_args: TaskSamplerArgs, **kwargs
    ) -> TaskSampler:
        if cfg.run_real:
            return RealOpeningfridgeTaskSampler(
                args=task_sampler_args, opening_fridge_task_type=cls.OPENING_fridge_TASK_TYPE
            )
        else:
            return OpeningfridgeTaskSampler(
                args=task_sampler_args, opening_fridge_task_type=cls.OPENING_fridge_TASK_TYPE
            )
    # @classmethod
    # @final
    def create_model(self, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in self.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in self.SENSORS)

        vit_preprocessor_uuids = [s for s in self.vit_preprocessor_uuids if "_only_viz" not in s]

        model = get_model(
            self.MODEL_TYPE, cfg, self.OPENING_fridge_TASK_TYPE, kwargs, vit_preprocessor_uuids
        )

        get_logger().warning(model)

        if (
            cfg.pretrained_model.only_load_model_state_dict
            and cfg.checkpoint is not None
        ):
            if not torch.cuda.is_available():
                model.load_state_dict(torch.load(cfg.checkpoint,map_location=torch.device('cpu'))["model_state_dict"])
            else:
                model.load_state_dict(torch.load(cfg.checkpoint)["model_state_dict"])

        return model
