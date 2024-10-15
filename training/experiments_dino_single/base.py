from abc import ABC
from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch.distributions.utils import lazy_property

try:
    from typing import Literal, final
except ImportError:
    from typing_extensions import Literal, final

import datasets
import numpy as np
import prior
import torch
import torch.optim as optim
from ai2thor.platform import CloudRendering
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.preprocessor import (
    Preprocessor,
    SensorPreprocessorGraph,
)
from allenact.base_abstractions.sensor import (
    ExpertActionSensor,
    Sensor,
    SensorSuite,
    Union,
)
from allenact.base_abstractions.task import TaskSampler
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    TrainingSettings,
    evenly_distribute_count_into_bins,
)
from allenact.utils.system import get_logger
from training import cfg
from training.tasks.object_nav import ObjectNavTaskSampler, ObjectNavTask
from training.utils.types import RewardConfig, TaskSamplerArgs
from training.robot import STRETCH_ENV_ARGS

from training.tasks.door_parsing import filter_dataset

class ObjectNavBaseConfig(ExperimentConfig, ABC):
    """The base config for all ObjectNav experiments."""

    NUM_TRAIN_PROCESSES = cfg.machine.num_train_processes
    NUM_VAL_PROCESSES = cfg.machine.num_val_processes
    NUM_TEST_PROCESSES = cfg.machine.num_test_processes

    @staticmethod
    def get_devices(split: Literal["train", "valid", "test"]) -> Tuple[torch.device]:
        if not torch.cuda.is_available():
            return (torch.device("cpu"),)

        if split == "train":
            gpus = cfg.machine.num_train_gpus
        elif split == "valid":
            gpus = cfg.machine.num_val_gpus
        elif split == "test":
            gpus = cfg.machine.num_test_gpus
        else:
            raise ValueError(f"Unknown split {split}")

        if gpus is None:
            gpus = torch.cuda.device_count()

        return tuple(torch.device(f"cuda:{i}") for i in range(gpus))

    @property
    def TRAIN_DEVICES(self) -> Tuple[torch.device]:
        return self.get_devices("train")

    @property
    def VAL_DEVICES(self) -> Tuple[torch.device]:
        return self.get_devices("valid")

    @property
    def TEST_DEVICES(self) -> Tuple[torch.device]:
        return self.get_devices("test")

    TEST_ON_VALIDATION: bool = cfg.evaluation.test_on_validation

    AGENT_MODE = cfg.agent.agent_mode
    CAMERA_WIDTH = cfg.agent.camera_width
    CAMERA_HEIGHT = cfg.agent.camera_height
    STEP_SIZE = cfg.agent.step_size
    VISIBILITY_DISTANCE = cfg.agent.visibility_distance
    FIELD_OF_VIEW = cfg.agent.field_of_view
    ROTATE_STEP_DEGREES = cfg.agent.rotate_step_degrees

    SENSORS: Sequence[Sensor] = []

    DISTANCE_TYPE = "l2"  # "geo"  # Can be "geo" or "l2"

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[
        int
    ] = cfg.training.advance_scene_rollout_period
    RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN = (
        -1
    )  # Should be > 0 if `ADVANCE_SCENE_ROLLOUT_PERIOD` is `None`
    RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE = 100

    OBJECT_NAV_TASK_TYPE = ObjectNavTask

    @lazy_property
    def HOUSE_DATASET(self):
        return prior.load_dataset("procthor-10k")

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()

    @staticmethod
    def get_platform(
        gpu_index: int, platform: Literal["CloudRendering", "Linux64"]
    ) -> Dict[str, Any]:
        """Return the platform specific args to be passed into AI2-THOR.

        Parameters:
        - gpu_index: The index of the GPU to use. Must be in the range [0,
          torch.cuda.device_count() - 1].
        """
        if gpu_index < 0:
            return {}
        elif gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"gpu_index must be in the range [0, {torch.cuda.device_count()}]."
                f" You gave {gpu_index}."
            )

        if platform == "CloudRendering":
            # NOTE: There is an off-by-1 error with cloud rendering where
            # gpu_index cannot be set to 1. It maps 0=>0, 2=>1, 3=>2, etc.
            # if gpu_index > 0:
            #     gpu_index += 1
            return {"gpu_device": gpu_index, "platform": CloudRendering}
        elif platform == "Linux64":
            return {"x_display": f":0.{gpu_index}"}
        else:
            raise ValueError(f"Unknown platform: {platform}")

    def machine_params(self, mode: Literal["train", "valid", "test"], **kwargs):
        devices: Sequence[torch.device]
        nprocesses: int
        if mode == "train":
            devices = self.TRAIN_DEVICES * cfg.distributed.nodes
            nprocesses = self.NUM_TRAIN_PROCESSES * cfg.distributed.nodes
        elif mode == "valid":
            devices = self.VAL_DEVICES
            nprocesses = self.NUM_VAL_PROCESSES
        elif mode == "test":
            devices = self.TEST_DEVICES
            nprocesses = self.NUM_TEST_PROCESSES
        else:
            raise NotImplementedError

        nprocesses = (
            evenly_distribute_count_into_bins(count=nprocesses, nbins=len(devices))
            if nprocesses > 0
            else [0] * len(devices)
        )

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        params = MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=devices,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

        # NOTE: for distributed setup
        if mode == "train" and "machine_id" in kwargs:
            machine_id = kwargs["machine_id"]
            assert (
                0 <= machine_id < cfg.distributed.nodes
            ), f"machine_id {machine_id} out of range [0, {cfg.distributed.nodes} - 1]"
            local_worker_ids = list(
                range(
                    len(self.TRAIN_DEVICES) * machine_id,
                    len(self.TRAIN_DEVICES) * (machine_id + 1),
                )
            )
            params.set_local_worker_ids(local_worker_ids)

        return params

    @classmethod
    def make_sampler_fn(
        cls, task_sampler_args: TaskSamplerArgs, **kwargs
    ) -> TaskSampler:
        return ObjectNavTaskSampler(
            args=task_sampler_args, object_nav_task_type=cls.OBJECT_NAV_TASK_TYPE
        )

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(
            np.linspace(start=0, stop=n, num=num_parts + 1, endpoint=True)
        ).astype(np.int32)

    def _get_sampler_args_for_scene_split(
        self,
        houses: datasets.Dataset,
        mode: Literal["train", "eval"],
        resample_same_scene_freq: int,
        allow_oversample: bool,
        allow_flipping: bool,
        process_ind: int,
        total_processes: int,
        max_tasks: Optional[int],
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        extra_controller_args: Optional[Dict[str, Any]] = None,
    ) -> TaskSamplerArgs:
        # NOTE: oversample some scenes -> bias
        oversample_warning = (
            f"Warning: oversampling some of the houses ({houses}) to feed all processes ({total_processes})."
            " You can avoid this by setting a number of workers divisible by the number of scenes"
        )
        house_inds = list(range(len(houses)))

        get_logger().warning(total_processes)
        get_logger().warning(house_inds)

        if total_processes > len(houses):
            if not allow_oversample:
                raise RuntimeError(
                    f"Cannot have `total_processes > len(houses)`"
                    f" ({total_processes} > {len(houses)}) when `allow_oversample` is `False`."
                )

            if total_processes % len(houses) != 0:
                get_logger().warning(oversample_warning)
            house_inds = house_inds * ceil(total_processes / len(houses))
            house_inds = house_inds[
                : total_processes * (len(house_inds) // total_processes)
            ]
        elif len(houses) % total_processes != 0:
            if process_ind == 0:  # Only print warning once
                get_logger().warning(
                    f"Number of houses {len(houses)} is not cleanly divisible by the number"
                    f" of processes ({total_processes}). Because of this, not all processes will"
                    f" be fed the same number of houses."
                )

        inds = self._partition_inds(len(house_inds), total_processes)
        house_inds = house_inds[inds[process_ind] : inds[process_ind + 1]]
        get_logger().warning(house_inds)
        # controller_args = {
        #     "branch": "main",
        #     "width": self.CAMERA_WIDTH,
        #     "height": self.CAMERA_HEIGHT,
        #     "rotateStepDegrees": self.ROTATE_STEP_DEGREES,
        #     "visibilityDistance": self.VISIBILITY_DISTANCE,
        #     "gridSize": self.STEP_SIZE,
        #     "agentMode": self.AGENT_MODE,
        #     "fieldOfView": self.FIELD_OF_VIEW,
        #     "snapToGrid": False,
        #     "renderDepthImage": any(isinstance(s, DepthSensor) for s in self.SENSORS),
        #     **self.get_platform(
        #         gpu_index=devices[process_ind % len(devices)],
        #         platform=cfg.ai2thor.platform,
        #     ),
        # }
        controller_args = STRETCH_ENV_ARGS.copy()
        controller_args.update({
            "rotateStepDegrees": self.ROTATE_STEP_DEGREES
        })
        get_logger().warning(
            devices[process_ind % len(devices)]
        )
        get_logger().warning(self.get_platform(
                gpu_index=devices[process_ind % len(devices)],
                platform=cfg.ai2thor.platform,
            )
        )
        controller_args.update(
            self.get_platform(
                gpu_index=devices[process_ind % len(devices)],
                platform=cfg.ai2thor.platform,
            )
        )
        if extra_controller_args:
            controller_args.update(extra_controller_args)
            if "commit_id" in extra_controller_args:
                del controller_args["branch"]

        return TaskSamplerArgs(
            process_ind=process_ind,
            mode=mode,
            house_inds=house_inds,
            houses=houses,
            sensors=self.SENSORS,
            controller_args=controller_args,
            target_object_types=cfg.target_object_types,
            max_steps=cfg.mdp.max_steps,
            seed=seeds[process_ind] if seeds is not None else None,
            deterministic_cudnn=deterministic_cudnn,
            reward_config=RewardConfig(**cfg.mdp.reward[mode]),
            max_tasks=max_tasks if max_tasks is not None else len(house_inds),
            allow_flipping=allow_flipping,
            distance_type=self.DISTANCE_TYPE,
            resample_same_scene_freq=resample_same_scene_freq,
        )

    def train_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        train_houses = self.HOUSE_DATASET["train"]
        train_houses = filter_dataset(train_houses, close_door=True)
        if cfg.procthor.num_train_houses:
            train_houses = train_houses.select(range(cfg.procthor.num_train_houses))

        get_logger().warning(len(train_houses))
        out = self._get_sampler_args_for_scene_split(
            houses=train_houses,
            mode="train",
            allow_oversample=True,
            max_tasks=float("inf"),
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN,
            # extra_controller_args=dict(scene="Procedural"),
            **kwargs,
        )
        return {"task_sampler_args": out}

    def valid_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        val_houses = self.HOUSE_DATASET["val"]
        val_houses = filter_dataset(val_houses, close_door=True)
        out = self._get_sampler_args_for_scene_split(
            houses=val_houses.select(range(100)),
            mode="eval",
            allow_oversample=False,
            max_tasks=200,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,
            # extra_controller_args=dict(scene="Procedural"),
            **kwargs,
        )
        return {"task_sampler_args": out}

    def test_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        if self.TEST_ON_VALIDATION:
            return self.valid_task_sampler_args(**kwargs)

        test_houses = self.HOUSE_DATASET["test"]
        test_houses = filter_dataset(test_houses, close_door=True)
        out = self._get_sampler_args_for_scene_split(
            houses=test_houses.select(range(100)),
            mode="eval",
            allow_oversample=False,
            max_tasks=10,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,
            # extra_controller_args=dict(scene="Procedural"),
            **kwargs,
        )
        return {"task_sampler_args": out}

    def training_pipeline(self, **kwargs):
        log_interval_small = (
            cfg.distributed.nodes * cfg.machine.num_train_processes * 32 * 10
            if torch.cuda.is_available
            else 1
        )
        log_interval_medium = (
            cfg.distributed.nodes * cfg.machine.num_train_processes * 64 * 5
            if torch.cuda.is_available
            else 1
        )
        log_interval_large = (
            cfg.distributed.nodes * cfg.machine.num_train_processes * 128 * 5
            if torch.cuda.is_available
            else 1
        )

        batch_steps_0 = int(10e6)
        batch_steps_1 = int(10e6)
        batch_steps_2 = int(1e9) - batch_steps_1 - batch_steps_0

        return TrainingPipeline(
            save_interval=cfg.training.save_interval,
            metric_accumulate_interval=cfg.training.log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=cfg.training.lr)),
            num_mini_batch=cfg.training.num_mini_batch,
            update_repeats=cfg.training.update_repeats,
            max_grad_norm=cfg.training.max_grad_norm,
            num_steps=cfg.training.num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=cfg.training.gamma,
            use_gae=cfg.training.use_gae,
            gae_lambda=cfg.training.gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=batch_steps_0,
                    training_settings=TrainingSettings(
                        num_steps=32, metric_accumulate_interval=log_interval_small
                    ),
                ),
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=batch_steps_1,
                    training_settings=TrainingSettings(
                        num_steps=64, metric_accumulate_interval=log_interval_medium
                    ),
                ),
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=batch_steps_2,
                    training_settings=TrainingSettings(
                        num_steps=128, metric_accumulate_interval=log_interval_large
                    ),
                ),
            ],
        )
