from typing import Dict, Any
import numpy as np
import copy
import gym
from torchvision import transforms

from ai2thor.controller import Controller

from allenact.base_abstractions.task import Task
from allenact.embodiedai.sensors.vision_sensors import VisionSensor, RGBSensor
from allenact.embodiedai.sensors.vision_sensors import Sensor

from training.utils import corruptions
from training.utils import map_utils
from training.utils.utils import ForkedPdb
import torchvision
from torchvision.transforms import Compose, Normalize
import torch
import PIL

from allenact.utils.system import get_logger

from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.tensor_utils import ScaleBothSides

from typing import Optional, Tuple, Any, cast, Union, Sequence


class RGBSensorThorController(VisionSensor[Controller, Task[Controller]]):
    """Sensor for RGB images in THOR.

    Returns from a running instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: Controller, task: Task[Controller]) -> np.ndarray:
        # options are "Defocus Blur", "Motion Blur", "Lighting", "Speckle Noise", "Spatter"
        return corruptions.apply_corruption_sequence(np.array(env.last_event.frame.copy()),[], [])

class RGBStretchControllerSensor(VisionSensor[Controller, Task[Controller]]):
    def __init__(
        self,
        use_resnet_normalization: bool = False,
        mean: Union[Sequence[float], np.ndarray, None] = None,
        stdev: Union[Sequence[float], np.ndarray, None] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "vision",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: Optional[int] = None,
        unnormalized_infimum: float = -np.inf,
        unnormalized_supremum: float = np.inf,
        scale_first: bool = True,
        **kwargs: Any
    ):
        """Initializer.

        # Parameters

        mean : The images will be normalized with the given mean
        stdev : The images will be normalized with the given standard deviations.
        height : If it's a non-negative integer and `width` is also non-negative integer, the image returned from the
                environment will be rescaled to have `height` rows and `width` columns using bilinear sampling.
        width : If it's a non-negative integer and `height` is also non-negative integer, the image returned from the
                environment will be rescaled to have `height` rows and `width` columns using bilinear sampling.
        uuid : The universally unique identifier for the sensor.
        output_shape : Optional observation space shape (alternative to `output_channels`).
        output_channels : Optional observation space number of channels (alternative to `output_shape`).
        unnormalized_infimum : Lower limit(s) for the observation space range.
        unnormalized_supremum : Upper limit(s) for the observation space range.
        scale_first : Whether to scale image before normalization (if needed).
        kwargs : Extra kwargs. Currently unused.
        """

        if not use_resnet_normalization:
            mean, stdev = None, None

        if isinstance(mean, tuple):
            mean = np.array(mean, dtype=np.float32).reshape((1, 1, len(mean)))
        if isinstance(stdev, tuple):
            stdev = np.array(stdev, dtype=np.float32).reshape((1, 1, len(stdev)))

        self._norm_means = np.array(mean) if mean is not None else None
        self._norm_sds = np.array(stdev) if stdev is not None else None
        assert (self._norm_means is None) == (self._norm_sds is None), (
            "In VisionSensor's config, "
            "either both mean/stdev must be None or neither."
        )
        get_logger().warning(mean)
        get_logger().warning(stdev)

        get_logger().warning(type(self._norm_means))
        get_logger().warning(type(self._norm_sds))
        get_logger().warning(self._norm_means is not None)
        self._should_normalize = self._norm_means is not None

        self._height = height
        self._width = width
        # output_channels
        assert (self._width is None) == (self._height is None), (
            "In VisionSensor's config, "
            "either both height/width must be None or neither."
        )

        self._scale_first = scale_first

        self.scaler: Optional[ScaleBothSides] = None
        if self._width is not None:
            self.scaler = ScaleBothSides(
                width=cast(int, self._width), height=cast(int, self._height)
            )

        self.to_pil = transforms.ToPILImage()  # assumes mode="RGB" for 3 channels

        self._observation_space = self._make_observation_space(
            output_shape=output_shape,
            output_channels=output_channels,
            unnormalized_infimum=unnormalized_infimum,
            unnormalized_supremum=unnormalized_supremum,
        )

        assert int(PIL.__version__.split(".")[0]) != 7, (
            "We found that Pillow version >=7.* has broken scaling,"
            " please downgrade to version 6.2.1 or upgrade to >=8.0.0"
        )

        observation_space = self._get_observation_space()
        get_logger().warning(locals())
        get_logger().warning(prepare_locals_for_super(locals()))
        super(VisionSensor, self).__init__(**prepare_locals_for_super(locals()))

    def process_img(self, img: np.ndarray):
        assert (
            np.issubdtype(img.dtype, np.float32)
            and (len(img.shape) == 2 or img.shape[-1] == 1)
        ) or (img.shape[-1] == 3 and np.issubdtype(img.dtype, np.uint8)), (
            "Input frame must either have 3 channels and be of"
            " type np.uint8 or have one channel and be of type np.float32"
        )

        assert img.dtype in [np.uint8, np.float32]

        assert not self._should_normalize

        return img

class RGBSensorStretchControllerNavigation(RGBStretchControllerSensor):
    """Sensor for RGB images in THOR.

    Returns from a running instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: Controller, task: Task[Controller]) -> np.ndarray:
        # options are "Defocus Blur", "Motion Blur", "Lighting", "Speckle Noise", "Spatter"
        frame = np.array(env.navigation_camera.copy())
        return frame



class RGBSensorStretchControllerManipulation(RGBStretchControllerSensor):
    """Sensor for RGB images in THOR.

    Returns from a running instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: Controller, task: Task[Controller]) -> np.ndarray:
        # options are "Defocus Blur", "Motion Blur", "Lighting", "Speckle Noise", "Spatter"
        frame = np.array(env.manipulation_camera.copy())
        return frame


class RGBSensorStretchControllerNavigationHist(RGBStretchControllerSensor):
    """Sensor for RGB images in THOR.

    Returns from a running instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: Controller, task: Task[Controller]) -> np.ndarray:
        # options are "Defocus Blur", "Motion Blur", "Lighting", "Speckle Noise", "Spatter"
        frame = np.array(env.nav_cam_his.copy())
        return frame



class RGBSensorStretchControllerManipulationHist(RGBStretchControllerSensor):
    """Sensor for RGB images in THOR.

    Returns from a running instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: Controller, task: Task[Controller]) -> np.ndarray:
        # options are "Defocus Blur", "Motion Blur", "Lighting", "Speckle Noise", "Spatter"
        frame = np.array(env.manip_cam_his.copy())
        return frame

class RGBSensorThorControllerFOVFix(RGBSensor[Controller, Task[Controller]]):
    """Sensor for RGB images in THOR.

    Returns from a running instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """
        
    """
    Use with
    agent:
        camera_width: 300
        camera_height: 400
    and zoom 1.5 to make the vertical FOV work out.

    DO NOT use this for real episodes.
    """

    def frame_from_env(self, env: Controller, task: Task[Controller]) -> np.ndarray:
        # options are "Defocus Blur", "Motion Blur", "Lighting", "Speckle Noise", "Spatter"
        return corruptions.apply_corruption_sequence(np.array(env.last_event.frame.copy()),["Zoom"], [1.5])

class AggregateMapSensor(Sensor[Controller, Task[Controller]]):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(uuid, observation_space, **kwargs)

    def get_observation(
        self,
        env: Controller,
        task
    ) -> Any:
        return task.aggregate_map


class CurrentMapSensor(Sensor[Controller, Task[Controller]]):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(uuid, observation_space, **kwargs)

    def get_observation(
        self,
        env: Controller,
        task
    ) -> Any:
        return task.instant_map


        