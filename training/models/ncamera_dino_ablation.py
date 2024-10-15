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
import torch.nn as nn
import numpy as np
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


from .ncamera_dino import VitMultiModalPrevActNCameraActorCritic, VitMultiModalPrevActNCameraTensorEncoder


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
    nn.init.orthogonal_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
  return module


def compute_conv2d_next_shape(
    input_shape: tuple,
    out_channels: int,
    stride: int, kernel_size: int, padding: int = 0
):
  """
  take input shape per-layer conv-info as input
  """
  # out_channels, kernel_size, stride, padding = conv_info
  _, h, w = input_shape

  if isinstance(stride, int):
    stride = (stride, stride)
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  if isinstance(padding, int):
    padding = (padding, padding)

  h = int((h + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1)
  w = int((w + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1)
  return (out_channels, h, w)


class BaseCNN(nn.Module):
  """
  Base CNN Encoder
  """

  def __init__(
      self,
  ):
    super().__init__()
    self.groups = 1
    self.input_shape = (3, 224, 224)
    self.input_channels, self.input_h, self.input_w = self.input_shape
    self.non_linear_func = nn.ReLU
    self.flatten = True
    self.build_convs()

  def build_convs(self):
    self.convs = []
    current_shape = self.input_shape
    in_channels = current_shape[0]
    for conv_info in self.conv_info_list:
      out_channels, kernel_size, stride, padding = conv_info
      conv = nn.Conv2d(
          in_channels=in_channels * self.groups,
          out_channels=out_channels * self.groups,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          groups=self.groups,
          padding_mode="reflect"
      )
      self.convs.append(conv)
      self.convs.append(self.non_linear_func())
      in_channels = out_channels
      current_shape = compute_conv2d_next_shape(
          current_shape, out_channels,
          stride, kernel_size, padding
      )

    if self.flatten:
      self.convs.append(
          nn.Flatten()
      )

    self.output_shape = current_shape
    self.output_shape = (
        self.output_shape[0] * self.groups,
    ) + self.output_shape[1:]
    self.layers = nn.Sequential(*self.convs)
    self.apply(orthogonal_init)

    self.output_dim = np.prod(self.output_shape)
    self.apply(orthogonal_init)

  def forward(self, x):
    x = x.view(
        torch.Size([
            np.prod(x.size()[:-3])]
        ) + x.size()[-3:]
    )
    x = self.layers(x)
    return x


class NatureEncoder(BaseCNN):
  """
  Nature Encoder
  """

  def __init__(
      self,
  ):
    # out_channels, kernel_size, stride, pading
    self.conv_info_list = [
        [32, 8, 4, 0],
        [64, 4, 2, 0],
        [64, 4, 2, 0],
        [64, 4, 2, 0],
        [64, 3, 1, 0],
    ]
    super().__init__()


class NoPretrainMultiModalNCameraTensorEncoder(nn.Module):
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

    self.vis_output_dim = None
    for cam in self.vit_uuids:
      vit_output_shape = observation_spaces.spaces[cam].shape
      assert vit_output_shape == self.vit_tensor_shape  # annoying if untrue
      nat_cnn = NatureEncoder()
      self.vis_output_dim = nat_cnn.output_dim
      self.vit_compressors.append(nat_cnn)

    self.proprioception_uuid = proprioception_uuid
    get_logger().warning(
        observation_spaces.spaces[self.proprioception_uuid].shape)

    self.proprio_input_dims = np.prod(
        observation_spaces.spaces[self.proprioception_uuid].shape
    )
    self.add_vision_compressor = cfg.model.add_vision_compressor
    if self.add_vision_compressor:
      self.vision_compressor = nn.Sequential(
          nn.Linear(
              self.number_cameras * self.vis_output_dim,
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
          self.add_vision_compressor + self.proprio_input_dims
      )
    else:
      return (
          self.number_cameras
          * self.vis_output_dim + self.proprio_input_dims
      )

  def compress_vit(self, observations, idx):
    return self.vit_compressors[idx](observations[self.vit_uuids[idx]])

  def adapt_input(self, observations):
    first_input = observations[self.vit_uuids[0]]  # privileged input
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
      viz_embs = self.compress_vit(observations, cam_idx)
      viz_x.append(viz_embs)

    viz_x = torch.cat(viz_x, dim=1)
    viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
    if self.add_vision_compressor:
      viz_x = self.vision_compressor(viz_x)

    proprio_x = observations[self.proprioception_uuid]
    x = torch.cat([viz_x, proprio_x], dim=-1)

    return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class NoPretrainMultiModalPrevActNCameraTensorEncoder(NoPretrainMultiModalNCameraTensorEncoder):
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

  def forward(self, observations, prev_actions):
    observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
        observations
    )

    viz_x = []
    for cam_idx in range(len(self.vit_uuids)):
      viz_embs = self.compress_vit(observations, cam_idx)
      viz_x.append(viz_embs)

    viz_x = torch.cat(viz_x, dim=1)
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
    x = torch.cat([viz_x, proprio_x], dim=-1)

    return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class NoPretrainMultiModalNCameraActorCritic(ContinuousVisualNavActorCritic):
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
      cfg: OmegaConf = None,
      visualize=False,
      add_tanh=False,
      init_std: float = 3.0,
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

    self.starting_time = datetime.now().strftime(
        "{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))

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
    return NoPretrainMultiModalNCameraTensorEncoder

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
      log_ac_return(actor_critic_output,
                    kwargs['observations']["task_id_sensor"])
    return actor_critic_output, memory


class NoPretrainModalPrevActNCameraActorCritic(NoPretrainMultiModalNCameraActorCritic):
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
    return NoPretrainMultiModalPrevActNCameraTensorEncoder

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
