"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import platform
from datetime import datetime
from typing import Tuple, Optional, List, Dict, cast
import platform
from datetime import datetime
import torch.nn.functional as F

import gym
import torch
from allenact.algorithms.onpolicy_sync.policy import ObservationType, DistributionType

from gym.spaces.dict import Dict as SpaceDict
from torch import nn
from allenact.embodiedai.models.visual_nav_models import (
    # VisualNavActorCritic,
    FusionType,
)
from .continuous_visual_nav import ContinuousVisualNavActorCritic, LinearGaussianActorSLHead, LinearCriticSLHead

from training.utils.utils import log_ac_return

from allenact.utils.system import get_logger

from omegaconf import OmegaConf
from allenact.base_abstractions.misc import ActorCriticOutput, Memory

from allenact.utils.misc_utils import prepare_locals_for_super

from allenact_plugins.gym_plugin.gym_distributions import GaussianDistr

class VitTensorNavNCameraActorCritic(ContinuousVisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        # goal_sensor_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        # add_prev_actions=False,
        # add_prev_action_null_token=False,
        action_embed_size=6,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        vit_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        Vit_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        # combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
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

        self.visual_encoder = VitNCameraTensorEncoder(  # type:ignore
            self.observation_space,
            # goal_sensor_uuid,
            vit_preprocessor_uuids,
            # goal_dims,
            Vit_compressor_hidden_out_dims,
            # combiner_hidden_out_dims,
        )

        self.create_state_encoders(
            obs_embed_size=self.visual_encoder.output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            # add_prev_actions=add_prev_actions,
            # add_prev_action_null_token=add_prev_action_null_token,
            # prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

        self.starting_time = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super(VitTensorNavNCameraActorCritic, self).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory


class VitMultiModalNCameraActorCritic(ContinuousVisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        # goal_sensor_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        # add_prev_actions=False,
        # add_prev_action_null_token=False,
        action_embed_size=6,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        vit_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        stretch_proprio_uuid: str = None,
        cfg: OmegaConf= None,
        visualize=False,
        add_tanh=False,
        init_std: float =3.0,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
            add_tanh=add_tanh,
            init_std=init_std
        )
        self.action_embed_size = action_embed_size
        self.cfg = cfg
        self.visualize = visualize
        self.create_encoder(
            vit_preprocessor_uuids,
            stretch_proprio_uuid,
        )

        self.create_state_encoders(
            obs_embed_size=self.visual_encoder.output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

        self.starting_time = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))

    def create_encoder(
        self, vit_preprocessor_uuids, stretch_proprio_uuid
    ):
        self.visual_encoder = self.visual_encoder_type(  # type:ignore
            self.observation_space,
            vit_preprocessor_uuids,
            stretch_proprio_uuid,
            self.cfg,
        )

    @property
    def visual_encoder_type(self):
        return VitMultiModalNCameraTensorEncoder

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super().forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory


class VitMultiModalPrevActNCameraActorCritic(VitMultiModalNCameraActorCritic):
    def create_encoder(
        self, vit_preprocessor_uuids, stretch_proprio_uuid
    ):
        self.visual_encoder = self.visual_encoder_type(  # type:ignore
            self.observation_space,
            vit_preprocessor_uuids,
            stretch_proprio_uuid,
            self.cfg,
            action_embed_size=self.action_embed_size,
        )

    @property
    def visual_encoder_type(self):
        return VitMultiModalPrevActNCameraTensorEncoder

    def forward_encoder(self, observations: ObservationType, prev_actions: torch.Tensor) -> torch.FloatTensor:
        # get_logger().warning("Forward PrecAct Encoder")
        return self.visual_encoder(observations, prev_actions)

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        action_scale_single = self.cfg.agent.action_scale[0]

        prev_actions = torch.clamp(
            prev_actions * action_scale_single,
            -1, 1
        )

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds = self.forward_encoder(observations, prev_actions)

        # 1.2 use embedding model to get prev_action embeddings
        # if self.prev_action_embedder.input_size == self.action_space.n + 1:
        #     # In this case we have a unique embedding for the start of an episode
        #     prev_actions_embeds = self.prev_action_embedder(
        #         torch.where(
        #             condition=0 != masks.view(*prev_actions.shape),
        #             input=prev_actions + 1,
        #             other=torch.zeros_like(prev_actions),
        #         )
        #     )
        # else:
        #     prev_actions_embeds = self.prev_action_embedder(prev_actions)
        # joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)
        joint_embeds = obs_embeds

        # 2. use RNNs to get single/multiple beliefs
        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            beliefs_dict[key], rnn_hidden_states = model(
                joint_embeds, memory.tensor(key), masks
            )
            memory.set_tensor(key, rnn_hidden_states)  # update memory here

        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(
            beliefs_dict, obs_embeds
        )  # fused beliefs

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    "beliefs": (
                        beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs
                    ),
                    "obs_embeds": obs_embeds,
                    "aux_model": (
                        self.aux_models[aux_uuid]
                        if aux_uuid in self.aux_models
                        else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=self.critic(beliefs),
            extras=extras,
        )
        return actor_critic_output, memory


class VitMultiModalPrevActV2NCameraActorCritic(VitMultiModalPrevActNCameraActorCritic):
    @property
    def visual_encoder_type(self):
        return VitMultiModalPrevActV2NCameraTensorEncoder

    def create_actorcritic_head(self):
        self.actor = LinearGaussianActorSLHead(
            self._hidden_size, self.action_space.shape[0],
            add_tanh=self.add_tanh,
            init_std=self.init_std
        )
        self.critic = LinearCriticSLHead(self._hidden_size)


class VitVisOnlyPrevActV2NCameraActorCritic(VitMultiModalPrevActV2NCameraActorCritic):
    @property
    def visual_encoder_type(self):
        return VitVisOnlyPrevActV2NCameraTensorEncoder

    def create_actorcritic_head(self):
        self.actor = LinearGaussianActorSLHead(
            self._hidden_size, self.action_space.shape[0],
            add_tanh=self.add_tanh,
            init_std=self.init_std
        )
        self.critic = LinearCriticSLHead(self._hidden_size)


class VitEmbVisOnlyPrevActV2NCameraActorCritic(VitMultiModalPrevActV2NCameraActorCritic):
    @property
    def visual_encoder_type(self):
        return VitEmbVisOnlyPrevActV2NCameraTensorEncoder

    def create_actorcritic_head(self):
        self.actor = LinearGaussianActorSLHead(
            self._hidden_size, self.action_space.shape[0],
            add_tanh=self.add_tanh,
            init_std=self.init_std
        )
        self.critic = LinearCriticSLHead(self._hidden_size)


class VitMultiModalDenseNCameraActorCritic(VitMultiModalNCameraActorCritic):
    @property
    def visual_encoder_type(self):
        return VitMultiModalDenseNCameraTensorEncoder


class VitMultiModalHistNCameraActorCritic(VitMultiModalNCameraActorCritic):
    @property
    def visual_encoder_type(self):
        return VitMultiModalHistNCameraTensorEncoder


from typing import Sequence, Union, Type, Any

import attr
import gym
import numpy as np
import torch.nn as nn

from allenact.base_abstractions.sensor import Sensor

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


class VitNCameraTensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        vit_preprocessor_uuids: List[str],
        # goal_embed_dims: int = 32,
        Vit_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.vit_uuids = vit_preprocessor_uuids
        self.number_cameras = len(vit_preprocessor_uuids)
        self.vit_hid_out_dims = Vit_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.vit_uuids])
        )
        assert not self.blind
        # if not self.blind:
        self.Vit_compressors = nn.ModuleList()
        # self.target_obs_combiners = nn.ModuleList()
        self.vit_tensor_shape = observation_spaces.spaces[self.vit_uuids[0]].shape

        for cam in self.vit_uuids:
            vit_output_shape = observation_spaces.spaces[cam].shape
            assert vit_output_shape == self.vit_tensor_shape # annoying if untrue

            self.Vit_compressors.append(nn.Sequential(
                nn.Conv2d(vit_output_shape[0], self.vit_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.vit_hid_out_dims[0:2], 1),
                nn.ReLU(),
            ))


    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        # if self.blind:
        #     return self.goal_embed_dims
        # else:
        return (
            self.number_cameras
            * self.vit_hid_out_dims[1]
            * self.vit_tensor_shape[1]
            * self.vit_tensor_shape[2]
        )

    def compress_Vit(self,observations,idx):
        return self.Vit_compressors[idx](observations[self.vit_uuids[idx]])

    def adapt_input(self, observations):
        first_input = observations[self.vit_uuids[0]] # privileged input
        use_agent = False
        nagent = 1
        if len(first_input.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = first_input.shape[:3]
        else:
            nstep, nsampler = first_input.shape[:2]

        for cam in self.vit_uuids:
            input_Vit = observations[cam]
            assert input_Vit.shape == first_input.shape
            observations[cam] = input_Vit.view(-1, *input_Vit.shape[-3:])
        
        # observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

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

        viz_x = []
        for cam_idx in range(len(self.vit_uuids)):
            viz_embs = self.compress_Vit(observations,cam_idx)
            viz_x.append(viz_embs)
        x = torch.cat(viz_x,dim=1)
        x = x.reshape(x.shape[0], -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class VitMultiModalNCameraTensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        vit_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        cfg: OmegaConf = None
    ) -> None:
        super().__init__()
        self.vit_uuids = vit_preprocessor_uuids
        self.number_cameras = len(vit_preprocessor_uuids)
        self.cfg = cfg
        self.vit_hid_out_dims = self.cfg.model.vit_compressor_hidden_out_dims
        self.combine_hid_out_dims = self.cfg.model.combiner_hidden_out_dims
        self.proprio_hid_out_dims = self.cfg.model.proprio_hidden_out_dims

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.vit_uuids])
        )
        assert not self.blind
        # if not self.blind:
        self.vit_compressors = nn.ModuleList()
        # self.target_obs_combiners = nn.ModuleList()
        self.vit_tensor_shape = observation_spaces.spaces[self.vit_uuids[0]].shape

        for cam in self.vit_uuids:
            vit_output_shape = observation_spaces.spaces[cam].shape
            assert vit_output_shape == self.vit_tensor_shape # annoying if untrue

            self.vit_compressors.append(nn.Sequential(
                nn.Conv2d(vit_output_shape[0], self.vit_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.vit_hid_out_dims[0:2], 1),
                nn.ReLU(),
            ))

        self.proprioception_uuid = proprioception_uuid
        get_logger().warning(observation_spaces.spaces[self.proprioception_uuid].shape)
        self.proprioception_head = nn.Sequential(
            nn.Linear(np.prod(observation_spaces.spaces[self.proprioception_uuid].shape), self.proprio_hid_out_dims[0]),
            nn.ReLU(),
            nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
            nn.ReLU(),
        )
        self.add_vision_compressor = cfg.model.add_vision_compressor
        if self.add_vision_compressor:
            self.vision_compressor = nn.Sequential(
                nn.Linear(
                    self.number_cameras * self.vit_hid_out_dims[1] * self.vit_tensor_shape[1] * self.vit_tensor_shape[2],
                    self.add_vision_compressor
                ),
                nn.ReLU()
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.add_vision_compressor:
            return (
                self.add_vision_compressor + self.proprio_hid_out_dims[1]
            )
        else:
            return (
                self.number_cameras
                * self.vit_hid_out_dims[1]
                * self.vit_tensor_shape[1]
                * self.vit_tensor_shape[2] + self.proprio_hid_out_dims[1]
            )

    def compress_vit(self,observations,idx):
        return self.vit_compressors[idx](observations[self.vit_uuids[idx]])

    def adapt_input(self, observations):
        first_input = observations[self.vit_uuids[0]] # privileged input
        use_agent = False
        nagent = 1
        if len(first_input.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = first_input.shape[:3]
        else:
            nstep, nsampler = first_input.shape[:2]

        for cam in self.vit_uuids:
            input_Vit = observations[cam]
            assert input_Vit.shape == first_input.shape
            observations[cam] = input_Vit.view(-1, *input_Vit.shape[-3:])
        
        # observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

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
        
        viz_x = []
        for cam_idx in range(len(self.vit_uuids)):
            viz_embs = self.compress_vit(observations,cam_idx)
            viz_x.append(viz_embs)

        viz_x = torch.cat(viz_x,dim=1)
        viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
        if self.add_vision_compressor:
            viz_x = self.vision_compressor(viz_x)
        proprio_x = self.proprioception_head(observations[self.proprioception_uuid])
        proprio_x = proprio_x.reshape(viz_x.shape[0], -1)
        x = torch.cat([viz_x, proprio_x], dim=-1)

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)

class VitMultiModalPrevActNCameraTensorEncoder(VitMultiModalNCameraTensorEncoder):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        vit_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        cfg: OmegaConf = None,
        action_embed_size: int = 5
    ) -> None:
        super().__init__(
            observation_spaces=observation_spaces,
            # goal_sensor_uuid: str,
            vit_preprocessor_uuids=vit_preprocessor_uuids,
            proprioception_uuid=proprioception_uuid,
            cfg=cfg
        )

        self.action_embed_size = action_embed_size
        self.no_prev_act = self.cfg.model.no_prev_act
        if self.no_prev_act:
            self.proprioception_head = nn.Sequential(
                nn.Linear(np.prod(observation_spaces.spaces[self.proprioception_uuid].shape), self.proprio_hid_out_dims[0]),
                nn.ReLU(),
                nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
                nn.ReLU(),
            )
        else:
            self.proprioception_head = nn.Sequential(
                nn.Linear(np.prod(observation_spaces.spaces[self.proprioception_uuid].shape) + self.action_embed_size, self.proprio_hid_out_dims[0]),
                nn.ReLU(),
                nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
                nn.ReLU(),
            )

    def forward(self, observations, prev_actions):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        viz_x = []
        for cam_idx in range(len(self.vit_uuids)):
            viz_embs = self.compress_vit(observations,cam_idx)
            viz_x.append(viz_embs)

        viz_x = torch.cat(viz_x,dim=1)
        viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
        if self.add_vision_compressor:
            viz_x = self.vision_compressor(viz_x)
        if self.no_prev_act:
            proprio_x = self.proprioception_head(observations[self.proprioception_uuid])
        else:
            proprio_x = self.proprioception_head(
                torch.cat([
                    observations[self.proprioception_uuid],
                    prev_actions
                ], dim=-1)
            )

        proprio_x = proprio_x.reshape(viz_x.shape[0], -1)
        x = torch.cat([viz_x, proprio_x], dim=-1)

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)

class VitMultiModalPrevActV2NCameraTensorEncoder(VitMultiModalPrevActNCameraTensorEncoder):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        vit_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        cfg: OmegaConf = None,
        action_embed_size: int = 5
    ) -> None:
        super().__init__(
            observation_spaces=observation_spaces,
            # goal_sensor_uuid: str,
            vit_preprocessor_uuids=vit_preprocessor_uuids,
            proprioception_uuid=proprioception_uuid,
            cfg=cfg,
            action_embed_size=action_embed_size
        )
        del self.proprioception_head

        self.proprio_input_dims = np.prod(observation_spaces.spaces[self.proprioception_uuid].shape)

    @property
    def output_dims(self):
        if self.add_vision_compressor:
            return (
                self.add_vision_compressor + self.proprio_input_dims
            )
        else:
            return (
                self.number_cameras
                * self.vit_hid_out_dims[1]
                * self.vit_tensor_shape[1]
                * self.vit_tensor_shape[2] + self.proprio_input_dims
            )

    def forward(self, observations, prev_actions):
        # get_logger().warning("Forward codebook Encoder")
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        viz_x = []
        for cam_idx in range(len(self.vit_uuids)):
            viz_embs = self.compress_vit(observations,cam_idx)
            viz_x.append(viz_embs)

        viz_x = torch.cat(viz_x,dim=1)
        viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
        if self.add_vision_compressor:
            viz_x = self.vision_compressor(viz_x)

        if self.no_prev_act:
            proprio_x = observations[self.proprioception_uuid]
        else:
            proprio_x = torch.cat([
                observations[self.proprioception_uuid],
                prev_actions
            ], dim=-1)
        proprio_x = proprio_x.reshape(viz_x.shape[0], -1)
        joint_embeds = torch.cat([viz_x, proprio_x], dim=-1)
        return self.adapt_output(joint_embeds, use_agent, nstep, nsampler, nagent)


# class VitMultiModalPrevActV2NCameraTensorEncoder(VitMultiModalCodebookV3NCameraTensorEncoder):
#     pass

class VitVisOnlyPrevActV2NCameraTensorEncoder(VitMultiModalPrevActV2NCameraTensorEncoder):
    @property
    def output_dims(self):
        if self.add_vision_compressor:
            return (
                self.add_vision_compressor
            )
        else:
            return (
                self.number_cameras
                * self.vit_hid_out_dims[1]
                * self.vit_tensor_shape[1]
                * self.vit_tensor_shape[2]
            )

    def forward(self, observations, prev_actions):
        # get_logger().warning("Forward codebook Encoder")
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        viz_x = []
        for cam_idx in range(len(self.vit_uuids)):
            viz_embs = self.compress_vit(observations,cam_idx)
            viz_x.append(viz_embs)

        viz_x = torch.cat(viz_x,dim=1)
        viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
        if self.add_vision_compressor:
            viz_x = self.vision_compressor(viz_x)

        joint_embeds = viz_x
        return self.adapt_output(joint_embeds, use_agent, nstep, nsampler, nagent)

class VitEmbVisOnlyPrevActV2NCameraTensorEncoder(VitMultiModalPrevActV2NCameraTensorEncoder):
    @property
    def output_dims(self):
        if self.add_vision_compressor:
            return (
                self.add_vision_compressor
            )
        else:
            return (
                self.number_cameras
                * self.vit_tensor_shape[0]
                * self.vit_tensor_shape[1]
                * self.vit_tensor_shape[2]
            )

    def forward(self, observations, prev_actions):
        # get_logger().warning("Forward codebook Encoder")
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        viz_x = []
        for cam_idx in range(len(self.vit_uuids)):
            viz_embs = observations[self.vit_uuids[cam_idx]].reshape(
                -1, self.vit_tensor_shape[0]
                * self.vit_tensor_shape[1]
                * self.vit_tensor_shape[2]
            )
            viz_x.append(viz_embs)

        viz_x = torch.cat(viz_x,dim=1)
        viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
        if self.add_vision_compressor:
            viz_x = self.vision_compressor(viz_x)

        joint_embeds = viz_x
        return self.adapt_output(joint_embeds, use_agent, nstep, nsampler, nagent)


class VitMultiModalDenseNCameraTensorEncoder(VitMultiModalNCameraTensorEncoder):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        vit_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        cfg: OmegaConf = None
    ) -> None:
        super().__init__(
            observation_spaces=observation_spaces,
            vit_preprocessor_uuids=vit_preprocessor_uuids,
            proprioception_uuid=proprioception_uuid,
            cfg = cfg
        )
        # self.combine_hid_out_dims = self.cfg.model.combiner_hidden_out_dims

        self.proprio_dim = np.prod(
            observation_spaces.spaces[self.proprioception_uuid].shape
        )
        self.target_obs_combiners = nn.ModuleList()
        for cam in self.vit_uuids:
            self.target_obs_combiners.append(nn.Sequential(
                nn.Conv2d(
                    self.vit_hid_out_dims[1] + self.proprio_dim,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
                nn.ReLU(),
            ))

        if self.add_vision_compressor:
            self.vision_compressor = nn.Sequential(
                nn.Linear(
                    self.number_cameras * self.combine_hid_out_dims[1] * self.vit_tensor_shape[1] * self.vit_tensor_shape[2],
                    self.add_vision_compressor
                ),
                nn.ReLU()
            )


    @property
    def output_dims(self):
        if self.add_vision_compressor:
            return self.add_vision_compressor
        else:
            return (
                self.number_cameras
                * self.combine_hid_out_dims[1]
                * self.vit_tensor_shape[1]
                * self.vit_tensor_shape[2]
            )

    def compress_vit(self,observations,idx, expanded_proprio_obs):
        # batch_size = observations[self.vit_uuids[idx]].shape(0)

        return self.target_obs_combiners[idx](
            torch.cat([
                self.vit_compressors[idx](observations[self.vit_uuids[idx]]),
                expanded_proprio_obs
            ], dim=1
        ))

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        expanded_proprio_obs = observations[self.proprioception_uuid].reshape(
            -1, self.proprio_dim, 1, 1
        ).expand(-1, -1, *self.vit_tensor_shape[1:])
        viz_x = []
        for cam_idx in range(len(self.vit_uuids)):
            viz_embs = self.compress_vit(observations,cam_idx, expanded_proprio_obs)
            viz_x.append(viz_embs)

        viz_x = torch.cat(viz_x,dim=1)
        viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
        if self.add_vision_compressor:
            viz_x = self.vision_compressor(viz_x)

        x = viz_x

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)

class VitMultiModalHistNCameraTensorEncoder(VitMultiModalNCameraTensorEncoder):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        vit_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        # goal_embed_dims: int = 32,
        cfg: OmegaConf = None
    ) -> None:
        super().__init__(
            observation_spaces=observation_spaces,
            vit_preprocessor_uuids=vit_preprocessor_uuids,
            proprioception_uuid=proprioception_uuid,
            cfg = cfg
        )
        # self.Vit_compressors = nn.ModuleList()
        # self.target_obs_combiners = nn.ModuleList()
        # self.vit_tensor_shape = observation_spaces.spaces[self.vit_uuids[0]].shape
        self.his_len = cfg.model.his_len
        self.single_cam_compressors = nn.ModuleList()
        for cam in self.vit_uuids:
            self.single_cam_compressors.append(nn.Sequential(
                nn.Linear(
                    self.his_len * self.vit_hid_out_dims[1] * self.vit_tensor_shape[1] * self.vit_tensor_shape[2],
                    self.proprio_hid_out_dims[1]
                ),
                nn.ReLU()
            ))

        self.proprio_dim = np.prod(
            observation_spaces.spaces[self.proprioception_uuid].shape
        )
        self.target_obs_combiners = nn.ModuleList()
        for cam in self.vit_uuids:
            vit_output_shape = observation_spaces.spaces[cam].shape
            assert vit_output_shape == self.vit_tensor_shape # annoying if untrue

            self.target_obs_combiners.append(nn.Sequential(
                nn.Conv2d(
                    self.vit_hid_out_dims[1] + self.proprio_dim,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
                nn.ReLU(),
            ))

        if self.add_vision_compressor:
            self.vision_compressor = nn.Sequential(
                nn.Linear(
                    self.number_cameras * self.combine_hid_out_dims[1] * self.vit_tensor_shape[1] * self.vit_tensor_shape[2],
                    self.add_vision_compressor
                ),
                nn.ReLU()
            )

        self.proprioception_uuid = proprioception_uuid
        get_logger().warning(observation_spaces.spaces[self.proprioception_uuid].shape)
        self.proprioception_head = nn.Sequential(
            nn.Linear(np.prod(observation_spaces.spaces[self.proprioception_uuid].shape), self.proprio_hid_out_dims[0]),
            nn.ReLU(),
            nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
            nn.ReLU(),
        )

    @property
    def output_dims(self):
        return (
            (self.number_cameras + 1) * self.proprio_hid_out_dims[1]
        )

    def compress_vit(self,observations,idx):
        batch_size = observations[self.vit_uuids[idx]].shape[0]
        x = observations[self.vit_uuids[idx]].reshape(
            batch_size * self.his_len,
            *self.vit_tensor_shape
        )
        x = self.vit_compressors[idx](
            x
        )
        x = x.reshape(batch_size, -1)  # flatten
        x = self.single_cam_compressors[idx](x)
        return x

    def adapt_input(self, observations):
        first_input = observations[self.vit_uuids[0]] # privileged input
        use_agent = False
        nagent = 1
        if len(first_input.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = first_input.shape[:3]
        else:
            nstep, nsampler = first_input.shape[:2]

        for cam in self.vit_uuids:
            input_Vit = observations[cam]
            assert input_Vit.shape == first_input.shape
            observations[cam] = input_Vit.view(-1, *input_Vit.shape[-3:])
        
        # observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        viz_x = []
        for cam_idx in range(len(self.vit_uuids)):
            viz_embs = self.compress_vit(observations,cam_idx)
            viz_x.append(viz_embs)

        viz_x = torch.cat(viz_x,dim=1)
        viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
        if self.add_vision_compressor:
            viz_x = self.vision_compressor(viz_x)
        proprio_x = self.proprioception_head(observations[self.proprioception_uuid])
        proprio_x = proprio_x.reshape(viz_x.shape[0], -1)
        x = torch.cat([viz_x, proprio_x], dim=-1)

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)

