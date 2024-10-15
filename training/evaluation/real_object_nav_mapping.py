from training.experiments.rgb_clipresnet50gru_ddppo import (
    ObjectNavRGBClipResNet50PPOExperimentConfig,
)
from training.tasks.mapping import ObjectNavRealMappingTask, ObjectNavRealMappingSampler


from training import cfg
import torch
import gym

try:
    from typing import final
except ImportError:
    from typing_extensions import final


from allenact.embodiedai.sensors.vision_sensors import DepthSensor, RGBSensor
from training.experiments.rgb_clipresnet50gru_ddppo import (
    ObjectNavRGBClipResNet50PPOExperimentConfig, TaskIdSensor
)
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from models.mapping_models import ResnetTensorNavMappingActorCritic
from training.sensors.vision import RGBSensorThorController,AggregateMapSensor, CurrentMapSensor


class ObjectNavCurrentRealOnly(ObjectNavRGBClipResNet50PPOExperimentConfig):

    OBJECT_NAV_TASK_TYPE = ObjectNavRealMappingTask
    @classmethod
    def tag(cls):
        return super().tag() + "-real_evaluation"

    def make_sampler_fn(self, task_sampler_args, **kwargs):
        return ObjectNavRealMappingSampler(args=task_sampler_args,object_nav_task_type=self.OBJECT_NAV_TASK_TYPE)

    SENSORS = [
        RGBSensorThorController(
            height=cfg.model.image_size,
            width=cfg.model.image_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
            mean=cfg.model.rgb_means,
            stdev=cfg.model.rgb_stds,
        ),
        GoalObjectTypeThorSensor(
            object_types=cfg.target_object_types,
        ),
        TaskIdSensor(),
        AggregateMapSensor(
            uuid="aggregate_map_sensor",
        ),
        CurrentMapSensor(
            uuid="current_map_sensor",
        ),
    ] 
    
    @classmethod
    @final
    def create_model(cls, **kwargs) -> torch.nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)

        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        model = ResnetTensorNavMappingActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavRealMappingTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet" if has_rgb else None,
            # depth_resnet_preprocessor_uuid="depth_clip_resnet" if has_depth else None,
            hidden_size=512,
            goal_dims=32,
            add_prev_actions=cfg.model.add_prev_actions_embedding,
            visualize=cfg.eval # TODO set this more explicitly/finer-grained later. Eval decent proxy for now
        )

        if (
            cfg.pretrained_model.only_load_model_state_dict
            and cfg.checkpoint is not None
        ):
            if not torch.cuda.is_available():
                model.load_state_dict(torch.load(cfg.checkpoint,map_location=torch.device('cpu'))["model_state_dict"])
            else:
                model.load_state_dict(torch.load(cfg.checkpoint)["model_state_dict"])

        return model