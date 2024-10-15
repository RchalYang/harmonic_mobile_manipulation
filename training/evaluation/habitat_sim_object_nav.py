from typing import Any

import gym
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from training import cfg
from training.experiments.rgb_clipresnet50gru_ddppo import (
    ObjectNavRGBClipResNet50PPOExperimentConfig, TaskIdSensor,
)
from training.sensors.vision import RGBSensorThorControllerFOVFix
from training.tasks.object_nav import SimOfCurrentRealObjectNavTaskSampler
from training.utils.utils import ForkedPdb

class ObjectNavHabitat(ObjectNavRGBClipResNet50PPOExperimentConfig):

    # To use this:
    # comment out lines 63-65 of allenact_plugins/ithor_plugin/ithor_sensors.py
    # change object types in sim of current real
    # uncomment look up in task sampler
    # verify correct object order in 

    CAMERA_WIDTH = 300
    CAMERA_HEIGHT = 400

    SENSORS = [
        RGBSensorThorControllerFOVFix(
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
    ]

    @classmethod
    def tag(cls):
        return super().tag() + "-habitat"

    def make_sampler_fn(self, task_sampler_args, **kwargs):
        return SimOfCurrentRealObjectNavTaskSampler(args=task_sampler_args)