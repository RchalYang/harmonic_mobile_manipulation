"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import platform
from datetime import datetime
from typing import Tuple, Optional, List, Dict, cast
import platform
from datetime import datetime

import gym
import torch
from allenact.algorithms.onpolicy_sync.policy import ObservationType

from gym.spaces.dict import Dict as SpaceDict
from torch import nn
from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)

from utils.utils import log_ac_return

class ResnetTensorNavNCameraActorCritic(VisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        add_prev_actions=False,
        add_prev_action_null_token=False,
        action_embed_size=6,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        resnet_preprocessor_uuids: Optional[List[str]] = None,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        visualize=False,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
        )
        self.visualize = visualize

        self.goal_visual_encoder = ResnetNCameraTensorGoalEncoder(  # type:ignore
            self.observation_space,
            goal_sensor_uuid,
            resnet_preprocessor_uuids,
            goal_dims,
            resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims,
        )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

        self.starting_time = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super(ResnetTensorNavNCameraActorCritic, self).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory


class ResnetNCameraTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        resnet_preprocessor_uuids: List[str],
        goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuids = resnet_preprocessor_uuids
        self.number_cameras = len(resnet_preprocessor_uuids)
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]
        if isinstance(self.goal_space, gym.spaces.Discrete):
            self.embed_goal = nn.Embedding(
                num_embeddings=self.goal_space.n, embedding_dim=self.goal_embed_dims,
            )
        elif isinstance(self.goal_space, gym.spaces.Box):
            self.embed_goal = nn.Linear(self.goal_space.shape[-1], self.goal_embed_dims)
        else:
            raise NotImplementedError

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.resnet_uuids])
        )
        if not self.blind:
            self.resnet_compressors = nn.ModuleList()
            self.target_obs_combiners = nn.ModuleList()
            self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuids[0]].shape

            for cam in self.resnet_uuids:
                resnet_output_shape = observation_spaces.spaces[cam].shape
                assert resnet_output_shape == self.resnet_tensor_shape # annoying if untrue

                self.resnet_compressors.append(nn.Sequential(
                    nn.Conv2d(resnet_output_shape[0], self.resnet_hid_out_dims[0], 1),
                    nn.ReLU(),
                    nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                    nn.ReLU(),
                ))

                self.target_obs_combiners.append(nn.Sequential(
                    nn.Conv2d(
                        self.resnet_hid_out_dims[1] + self.goal_embed_dims,
                        self.combine_hid_out_dims[0],
                        1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
                ))

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return (
                self.number_cameras
                * self.combine_hid_out_dims[-1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_resnet(self,observations,idx):
        return self.resnet_compressors[idx](observations[self.resnet_uuids[idx]])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        first_input = observations[self.resnet_uuids[0]] # privileged input
        use_agent = False
        nagent = 1
        if len(first_input.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = first_input.shape[:3]
        else:
            nstep, nsampler = first_input.shape[:2]

        for cam in self.resnet_uuids:
            input_resnet = observations[cam]
            assert input_resnet.shape == first_input.shape
            observations[cam] = input_resnet.view(-1, *input_resnet.shape[-3:])
        
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent


    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        
        viz_x = []
        for cam_idx in range(len(self.resnet_uuids)):
            viz_embs = [
                self.compress_resnet(observations,cam_idx),
                self.distribute_target(observations),
            ]
            viz_x.append(self.target_obs_combiners[cam_idx](torch.cat(viz_embs, dim=1,)))
        x = torch.cat(viz_x,dim=1)
        x = x.reshape(x.shape[0], -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


from typing import Sequence, Union, Type, Any

import attr
import gym
import numpy as np
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor


class TaskIdSensor(Sensor):
    def __init__(
        self,
        uuid: str = "task_id_sensor",
        **kwargs: Any,
    ):
        super().__init__(uuid=uuid, observation_space=gym.spaces.Discrete(1))

    def _get_observation_space(self):
        if self.target_to_detector_map is None:
            return gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            return gym.spaces.Discrete(len(self.detector_types))

    def get_observation(
        self,
        env,
        task,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "id" in task.task_info:
            id=task.task_info["id"]
        else:
            id="no_task_id_provided"
        out = [ord(k) for k in id]
        for _ in range(len(out), 1000):
            out.append(ord(" "))
        return out



@attr.s(kw_only=True)
class ClipResNetPreprocessNCameraGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    clip_model_type: str = attr.ib()
    screen_size: int = attr.ib()
    pool: bool = attr.ib(default=False)

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        self.resnet_preprocessor_uuids = []
        for camera in [s for s in self.sensors if (isinstance(s, RGBSensor) or isinstance(s,DepthSensor))]:
            if "_only_viz" not in camera.uuid:
                if isinstance(camera, RGBSensor):
                    assert (
                        np.linalg.norm(
                            np.array(camera._norm_means)
                            - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
                        )
                        < 1e-5
                    )
                    assert (
                        np.linalg.norm(
                            np.array(camera._norm_sds)
                            - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
                        )
                        < 1e-5
                    )
                preprocessors.append(
                    ClipResNetPreprocessor(
                        rgb_input_uuid=camera.uuid,
                        clip_model_type=self.clip_model_type,
                        pool=self.pool,
                        output_uuid=camera.uuid+"_clip_resnet",
                        input_img_height_width=(camera.height,camera.width)
                    )
                )
                self.resnet_preprocessor_uuids.append(camera.uuid+"_clip_resnet")

            else:
                self.resnet_preprocessor_uuids.append(camera.uuid)

        return preprocessors

    def create_model(self, num_actions: int, visualize: bool, **kwargs) -> nn.Module:
        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )
        self.resnet_preprocessor_uuids = [s for s in self.resnet_preprocessor_uuids if "_only_viz" not in s]

        # display or assert sensor order here? possible source for sneaky failure if they're not the same 
        # as in pretraining when loading weights.

        return ResnetTensorNavNCameraActorCritic(
            action_space=gym.spaces.Discrete(num_actions),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            resnet_preprocessor_uuids=self.resnet_preprocessor_uuids,
            hidden_size=512,
            goal_dims=32,
            add_prev_actions=True,
            visualize=visualize
        )