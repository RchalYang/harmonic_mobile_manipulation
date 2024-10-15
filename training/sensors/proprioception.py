from typing import Dict, Any
import numpy as np
import copy
import gym

from allenact.utils.system import get_logger

from ai2thor.controller import Controller

from allenact.base_abstractions.task import Task
from allenact.embodiedai.sensors.vision_sensors import RGBSensor
from allenact.base_abstractions.sensor import Sensor

from training.utils import corruptions
from training.utils import map_utils
from training.utils.utils import ForkedPdb

class StretchArmSensorThorController(Sensor[Controller, Task[Controller]]):
    def __init__(self, uuid:str):
        observation_space = gym.spaces.Box(
            low=np.float32(-1),
            high=np.float32(1),
            shape=(4,),
        )
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(
        self, env: Controller, task: Task[Controller], *args: Any, **kwargs: Any
    ) -> Any:
        """Returns observations from the environment (or task).

        # Parameters

        env : The environment the sensor is used upon.
        task : (Optionally) a Task from which the sensor should get data.

        # Returns

        Current observation for Sensor.
        """
        arm_info = env.get_arm_proprioception()

        y = arm_info["position"]["y"]
        z = arm_info["position"]["z"]

        w_rotation = arm_info["rotation"]["y"]

        w_sin = np.sin(w_rotation / 360 * 2 * np.pi)
        w_cos = np.cos(w_rotation / 360 * 2 * np.pi)
        return np.array([
            y, z, w_sin, w_cos
        ], dtype=np.float32)

class StretchArmSensorV2ThorController(Sensor[Controller, Task[Controller]]):
    def __init__(self, uuid:str, domain_randomization=True):
        observation_space = gym.spaces.Box(
            low=np.float32(-1),
            high=np.float32(1),
            shape=(5,),
        )
        self.domain_randomization = domain_randomization
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(
        self, env: Controller, task: Task[Controller], *args: Any, **kwargs: Any
    ) -> Any:
        """Returns observations from the environment (or task).

        # Parameters

        env : The environment the sensor is used upon.
        task : (Optionally) a Task from which the sensor should get data.

        # Returns

        Current observation for Sensor.
        """
        # arm_position = env.get_relative_stretch_current_arm_state
        arm_info = env.get_arm_proprioception()

        y= arm_info[1]

        z = arm_info[2]

        w_rotation = arm_info[3]

        pos_encoding = 2 * (((w_rotation + 270) % 360) / 360) - 1

        w_sin = np.sin(w_rotation / 360 * 2 * np.pi)
        w_cos = np.cos(w_rotation / 360 * 2 * np.pi)
        obs = np.array([
            y, z, w_sin, w_cos, pos_encoding
            # y, z, pos_encoding
        ], dtype=np.float32)
        if self.domain_randomization:
            obs += np.random.normal(size=5) * 0.05
        return obs
    

class StretchArmHistSensorThorController(Sensor[Controller, Task[Controller]]):
    def __init__(
        self, 
        uuid:str, 
        domain_randomization=True,
        his_len: int=5
    ):
        observation_space = gym.spaces.Box(
            low=np.float32(-1),
            high=np.float32(1),
            shape=(5 * his_len,),
        )
        self.his_len = his_len
        self.domain_randomization = domain_randomization
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(
        self, env: Controller, task: Task[Controller], *args: Any, **kwargs: Any
    ) -> Any:
        """Returns observations from the environment (or task).

        # Parameters

        env : The environment the sensor is used upon.
        task : (Optionally) a Task from which the sensor should get data.

        # Returns

        Current observation for Sensor.
        """
        obs = env.proprio_his.copy()
        if self.domain_randomization:
            obs += np.random.normal(size=5 * self.his_len) * 0.05
        return obs