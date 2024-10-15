from typing import List, Optional, Any, cast, Dict, Tuple

import clip
import gym
import numpy as np
import torch
import torch.nn as nn
from clip.model import CLIP

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super

from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor

import torchvision
from torchvision.transforms import Compose, Normalize

from allenact.utils.system import get_logger

class ToFloat:
    # def __init__(self, device):
    #     self.device = device
    def __call__(self, tensor):
        return tensor.float() / 255.0


class ClipResNetAugPreprocessor(ClipResNetPreprocessor):
    """Preprocess RGB or depth image using a ResNet model with CLIP model
    weights."""

    CLIP_RGB_MEANS = (0.48145466, 0.4578275, 0.40821073)
    CLIP_RGB_STDS = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        augment_obs: bool,
        rgb_input_uuid: str,
        clip_model_type: str,
        pool: bool,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        input_img_height_width: Tuple[int, int] = (224, 224),
        **kwargs: Any,
    ):
        super().__init__(**prepare_locals_for_super(locals()))
        self.augment_obs = augment_obs
        aug_list_of_transformations= [
            # torchvision.transforms.ToTensor(),
            # ToDevice(self.device),
            torchvision.transforms.Resize(
                size=input_img_height_width,
            ),
            torchvision.transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05
            ),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
            torchvision.transforms.RandomResizedCrop(
                size=input_img_height_width,
                scale=(0.9, 1),
            ),
            torchvision.transforms.RandomPosterize(bits=7, p=0.2),
            torchvision.transforms.RandomPosterize(bits=6, p=0.2),
            torchvision.transforms.RandomPosterize(bits=5, p=0.2),
            torchvision.transforms.RandomPosterize(bits=4, p=0.2),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            ToFloat(),
            torchvision.transforms.Normalize(
                self.CLIP_RGB_MEANS,
                self.CLIP_RGB_STDS
            )
        ]
        self.aug_transformation = Compose(aug_list_of_transformations)
        noaug_list_of_transformations= [
            torchvision.transforms.Resize(
                size=input_img_height_width,
            ),
            ToFloat(),
            torchvision.transforms.Normalize(
                self.CLIP_RGB_MEANS,
                self.CLIP_RGB_STDS
            )
        ]
        self.noaug_transformation = Compose(noaug_list_of_transformations)
        self.input_img_height_width = input_img_height_width

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        if self.augment_obs:
            x = self.aug_transformation(x)
        else:
            x = self.noaug_transformation(x)
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x).float()
        return x



class ClipResNetAugHistoryPreprocessor(ClipResNetAugPreprocessor):
    """Preprocess RGB or depth image using a ResNet model with CLIP model
    weights."""

    def __init__(
        self,
        augment_obs: bool,
        rgb_input_uuid: str,
        clip_model_type: str,
        pool: bool,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        input_img_height_width: Tuple[int, int] = (224, 224),
        **kwargs: Any,
    ):
        super().__init__(**prepare_locals_for_super(locals()))
        self.clip_channel = 2048

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        # res_list = []
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 1, 4, 2, 3)  # b[his]hwc -> b[his]chw
        batch_size, his_len, channel, height, width = x.shape

        x = x.reshape(
            batch_size * his_len, channel,
            height, width
        )
        if self.augment_obs:
            x = self.aug_transformation(x)
        else:
            x = self.noaug_transformation(x)
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x).float()
        x = x.reshape(batch_size, self.clip_channel * his_len,
            *self.observation_space.shape[1:]
        )
        return x
