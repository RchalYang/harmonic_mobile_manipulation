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
    # VisualNavActorCritic,
    FusionType,
)
from .continuous_visual_nav import ContinuousVisualNavActorCritic

from training.utils.utils import log_ac_return

from allenact.utils.system import get_logger

class ResnetTensorNavNCameraActorCritic(ContinuousVisualNavActorCritic):
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
        resnet_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
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

        self.visual_encoder = ResnetNCameraTensorEncoder(  # type:ignore
            self.observation_space,
            # goal_sensor_uuid,
            resnet_preprocessor_uuids,
            # goal_dims,
            resnet_compressor_hidden_out_dims,
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
        actor_critic_output, memory = super(ResnetTensorNavNCameraActorCritic, self).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory


class ResnetTensorNavMultiModalNCameraActorCritic(ContinuousVisualNavActorCritic):
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
        resnet_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        stretch_proprio_uuid: str = None,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (16, 16),
        visualize=False,
        add_tanh=False,
        init_std: float =3.0,
        add_vision_compressor=None
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
        self.visualize = visualize
        self.visual_encoder = ResnetMultiModalNCameraTensorEncoder(  # type:ignore
            self.observation_space,
            # goal_sensor_uuid,
            resnet_preprocessor_uuids,
            stretch_proprio_uuid,
            resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims,
            proprio_hidden_out_dims,
            add_vision_compressor=add_vision_compressor
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

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super(ResnetTensorNavMultiModalNCameraActorCritic, self).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory


class ResnetMultiModalHistNCameraActorCritic(ContinuousVisualNavActorCritic):
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
        resnet_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        stretch_proprio_uuid: str = None,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (16, 16),
        visualize=False,
        add_tanh=False,
        init_std: float =3.0,
        add_vision_compressor=None,
        his_len: int = 5
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
        self.visualize = visualize
        self.visual_encoder = ResnetMultiModalHistNCameraTensorEncoder(  # type:ignore
            self.observation_space,
            # goal_sensor_uuid,
            resnet_preprocessor_uuids,
            stretch_proprio_uuid,
            resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims,
            proprio_hidden_out_dims,
            add_vision_compressor=add_vision_compressor,
            his_len=his_len
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

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super(ResnetMultiModalHistNCameraActorCritic, self).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory


class ResnetTensorNavMultiModalTransformerNCameraActorCritic(ContinuousVisualNavActorCritic):
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
        resnet_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        stretch_proprio_uuid: str = None,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        # combiner_hidden_out_dims: Tuple[int, int] = (64, 32),
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
        self.visualize = visualize
        self.visual_encoder = ResnetMultiModalNCameraTransformerTensorEncoder(  # type:ignore
            self.observation_space,
            # goal_sensor_uuid,
            resnet_preprocessor_uuids,
            stretch_proprio_uuid,
            resnet_compressor_hidden_out_dims,
            # combiner_hidden_out_dims,
            proprio_hidden_out_dims,
            transformer_params,
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

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super(ResnetTensorNavMultiModalTransformerNCameraActorCritic, self).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory


class ResnetMultiModalTransformerHistNCameraActorCritic(ContinuousVisualNavActorCritic):
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
        resnet_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        stretch_proprio_uuid: str = None,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (64, 32),
        visualize=False,
        add_tanh=False,
        init_std: float = 3.0,
        his_len: int = 5,
        add_his_emb: bool = True,
        add_modal_emb: bool = True
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
        self.visualize = visualize
        self.visual_encoder = ResnetMultiModalNCameraTransformerHistoryTensorEncoder(  # type:ignore
            self.observation_space,
            # goal_sensor_uuid,
            resnet_preprocessor_uuids,
            stretch_proprio_uuid,
            resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims,
            proprio_hidden_out_dims,
            transformer_params,
            his_len=his_len,
            add_his_emb=add_his_emb,
            add_modal_emb=add_modal_emb
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

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super(
            ResnetMultiModalTransformerHistNCameraActorCritic, self
        ).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory


class ResnetMultiModalTransformerHistV2NCameraActorCritic(ContinuousVisualNavActorCritic):
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
        resnet_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        stretch_proprio_uuid: str = None,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (64, 32),
        visualize=False,
        add_tanh=False,
        init_std: float = 3.0,
        his_len: int = 5,
        add_his_emb: bool = True,
        add_modal_emb: bool = True
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
        self.visualize = visualize
        self.visual_encoder = ResnetMultiModalNCameraTransformerHistoryV2TensorEncoder(  # type:ignore
            self.observation_space,
            # goal_sensor_uuid,
            resnet_preprocessor_uuids,
            stretch_proprio_uuid,
            resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims,
            proprio_hidden_out_dims,
            transformer_params,
            his_len=his_len,
            add_his_emb=add_his_emb,
            add_modal_emb=add_modal_emb
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

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super(
            ResnetMultiModalTransformerHistV2NCameraActorCritic, self
        ).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory



class ResnetMultiModalTransformerHistV3NCameraActorCritic(ContinuousVisualNavActorCritic):
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
        resnet_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        stretch_proprio_uuid: str = None,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (64, 32),
        visualize=False,
        add_tanh=False,
        init_std: float = 3.0,
        his_len: int = 5,
        add_his_emb: bool = True,
        add_modal_emb: bool = True
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
        self.visualize = visualize
        self.visual_encoder = ResnetMultiModalNCameraTransformerHistoryV3TensorEncoder(  # type:ignore
            self.observation_space,
            # goal_sensor_uuid,
            resnet_preprocessor_uuids,
            stretch_proprio_uuid,
            resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims,
            proprio_hidden_out_dims,
            transformer_params,
            his_len=his_len,
            add_his_emb=add_his_emb,
            add_modal_emb=add_modal_emb
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

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super(
            ResnetMultiModalTransformerHistV3NCameraActorCritic, self
        ).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory


class ResnetMultiModalTransformerHistMergeNCameraActorCritic(ContinuousVisualNavActorCritic):
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
        resnet_preprocessor_uuids: Optional[List[str]] = None,
        # goal_dims: int = 32,
        stretch_proprio_uuid: str = None,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (64, 32),
        visualize=False,
        add_tanh=False,
        init_std: float = 3.0,
        his_len: int = 5,
        add_his_emb: bool = True,
        # add_modal_emb: bool = True
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
        self.visualize = visualize
        self.visual_encoder = ResnetMultiModalNCameraTransformerHistoryMergeTensorEncoder(  # type:ignore
            self.observation_space,
            # goal_sensor_uuid,
            resnet_preprocessor_uuids,
            stretch_proprio_uuid,
            resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims,
            proprio_hidden_out_dims,
            transformer_params,
            his_len=his_len,
            add_his_emb=add_his_emb,
            # add_modal_emb=add_modal_emb
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

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def forward(self, **kwargs):
        actor_critic_output, memory = super(
            ResnetMultiModalTransformerHistMergeNCameraActorCritic, self
        ).forward(**kwargs)
        if self.visualize:
            log_ac_return(actor_critic_output,kwargs['observations']["task_id_sensor"])
        return actor_critic_output, memory

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



class ResnetNCameraTensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        resnet_preprocessor_uuids: List[str],
        # goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.resnet_uuids = resnet_preprocessor_uuids
        self.number_cameras = len(resnet_preprocessor_uuids)
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.resnet_uuids])
        )
        assert not self.blind
        # if not self.blind:
        self.resnet_compressors = nn.ModuleList()
        # self.target_obs_combiners = nn.ModuleList()
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

            # self.target_obs_combiners.append(nn.Sequential(
            #     nn.Conv2d(
            #         self.resnet_hid_out_dims[1],
            #         self.combine_hid_out_dims[0],
            #         1,
            #     ),
            #     nn.ReLU(),
            #     nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            # ))

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
            * self.resnet_hid_out_dims[1]
            * self.resnet_tensor_shape[1]
            * self.resnet_tensor_shape[2]
        )

    def compress_resnet(self,observations,idx):
        return self.resnet_compressors[idx](observations[self.resnet_uuids[idx]])

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
        
        # if self.blind:
        #     return self.embed_goal(observations[self.goal_uuid])
        
        viz_x = []
        for cam_idx in range(len(self.resnet_uuids)):
            viz_embs = self.compress_resnet(observations,cam_idx)
            viz_x.append(viz_embs)
        x = torch.cat(viz_x,dim=1)
        x = x.reshape(x.shape[0], -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)




class ResnetMultiModalNCameraTensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        resnet_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        # goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (32, 32),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        add_vision_compressor: int = None
    ) -> None:
        super().__init__()
        self.resnet_uuids = resnet_preprocessor_uuids
        self.number_cameras = len(resnet_preprocessor_uuids)
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.proprio_hid_out_dims = proprio_hidden_out_dims

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.resnet_uuids])
        )
        assert not self.blind
        # if not self.blind:
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
                    self.resnet_hid_out_dims[1],
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
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
        self.add_vision_compressor = add_vision_compressor
        if self.add_vision_compressor:
            self.vision_compressor = nn.Sequential(
                nn.Linear(
                    self.number_cameras * self.combine_hid_out_dims[1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2],
                    self.add_vision_compressor
                ),
                nn.ReLU()
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        # if self.blind:
        #     return self.goal_embed_dims
        # else:
        if self.add_vision_compressor:
            return (
                self.add_vision_compressor + self.proprio_hid_out_dims[1]
            )
        else:
            return (
                self.number_cameras
                * self.combine_hid_out_dims[1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2] + self.proprio_hid_out_dims[1]
            )

    def compress_resnet(self,observations,idx):
        return self.target_obs_combiners[idx](
            self.resnet_compressors[idx](observations[self.resnet_uuids[idx]])
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
        
        # if self.blind:
        #     return self.embed_goal(observations[self.goal_uuid])
        
        viz_x = []
        for cam_idx in range(len(self.resnet_uuids)):
            viz_embs = self.compress_resnet(observations,cam_idx)
            viz_x.append(viz_embs)

        viz_x = torch.cat(viz_x,dim=1)
        viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
        if self.add_vision_compressor:
            viz_x = self.vision_compressor(viz_x)
        # get_logger().warning(viz_x.shape)
        # get_logger().warning(observations[self.proprioception_uuid].shape)
        # get_logger().warning(observations[self.proprioception_uuid].dtype)
        proprio_x = self.proprioception_head(observations[self.proprioception_uuid])
        proprio_x = proprio_x.reshape(viz_x.shape[0], -1)
        x = torch.cat([viz_x, proprio_x], dim=-1)

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class ResnetMultiModalHistNCameraTensorEncoder(ResnetMultiModalNCameraTensorEncoder):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        resnet_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        # goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (32, 32),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        add_vision_compressor: int = None,
        his_len: int = 5
    ) -> None:
        super().__init__(
            observation_spaces=observation_spaces,
            resnet_preprocessor_uuids=resnet_preprocessor_uuids,
            proprioception_uuid=proprioception_uuid,
            resnet_compressor_hidden_out_dims=resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims=combiner_hidden_out_dims,
            proprio_hidden_out_dims=proprio_hidden_out_dims,
            add_vision_compressor=add_vision_compressor,
        )
        # self.resnet_compressors = nn.ModuleList()
        # self.target_obs_combiners = nn.ModuleList()
        # self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuids[0]].shape
        self.his_len = his_len
        self.single_cam_compressors = nn.ModuleList()
        for cam in self.resnet_uuids:
        #     resnet_output_shape = observation_spaces.spaces[cam].shape
        #     assert resnet_output_shape == self.resnet_tensor_shape # annoying if untrue

        #     self.resnet_compressors.append(nn.Sequential(
        #         nn.Conv2d(resnet_output_shape[0], self.resnet_hid_out_dims[0], 1),
        #         nn.ReLU(),
        #         nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
        #         nn.ReLU(),
        #     ))

        #     self.target_obs_combiners.append(nn.Sequential(
        #         nn.Conv2d(
        #             self.resnet_hid_out_dims[1],
        #             self.combine_hid_out_dims[0],
        #             1,
        #         ),
        #         nn.ReLU(),
        #         nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
        #         nn.ReLU(),
        #     ))        if self.add_vision_compressor:
            self.single_cam_compressors.append(nn.Sequential(
                nn.Linear(
                    self.his_len * self.combine_hid_out_dims[1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2],
                    self.proprio_hid_out_dims[1]
                ),
                nn.ReLU()
            ))

        self.proprioception_uuid = proprioception_uuid
        get_logger().warning(observation_spaces.spaces[self.proprioception_uuid].shape)
        self.proprioception_head = nn.Sequential(
            nn.Linear(np.prod(observation_spaces.spaces[self.proprioception_uuid].shape), self.proprio_hid_out_dims[0]),
            nn.ReLU(),
            nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
            nn.ReLU(),
        )
        # self.add_vision_compressor = add_vision_compressor
        # if self.add_vision_compressor:
        #     self.vision_compressor = nn.Sequential(
        #         nn.Linear(
        #             self.number_cameras * self.combine_hid_out_dims[1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2],
        #             self.add_vision_compressor
        #         ),
        #         nn.ReLU()
        #     )

    @property
    def output_dims(self):
        # if self.blind:
        #     return self.goal_embed_dims
        # else:
        # if self.add_vision_compressor:
        #     return (
        #         self.add_vision_compressor + self.proprio_hid_out_dims[1]
        #     )
        # else:
        return (
            (self.number_cameras + 1) * self.proprio_hid_out_dims[1]
        )

    # def compress_resnet(self,observations,idx):
    #     return self.target_obs_combiners[idx](
    #         self.resnet_compressors[idx](observations[self.resnet_uuids[idx]])
    #     )

    def compress_resnet(self,observations,idx):
        batch_size = observations[self.resnet_uuids[idx]].shape[0]
        x = observations[self.resnet_uuids[idx]].reshape(
            batch_size * self.his_len,
            *self.resnet_tensor_shape
        )
        x = self.target_obs_combiners[idx](self.resnet_compressors[idx](
            x
        ))
        x = x.reshape(batch_size, -1)  # flatten
        x = self.single_cam_compressors[idx](x)
        return x

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
        
        # observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        # if self.blind:
        #     return self.embed_goal(observations[self.goal_uuid])
        
        viz_x = []
        for cam_idx in range(len(self.resnet_uuids)):
            viz_embs = self.compress_resnet(observations,cam_idx)
            viz_x.append(viz_embs)

        viz_x = torch.cat(viz_x,dim=1)
        viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten
        if self.add_vision_compressor:
            viz_x = self.vision_compressor(viz_x)
        # get_logger().warning(viz_x.shape)
        # get_logger().warning(observations[self.proprioception_uuid].shape)
        # get_logger().warning(observations[self.proprioception_uuid].dtype)
        proprio_x = self.proprioception_head(observations[self.proprioception_uuid])
        proprio_x = proprio_x.reshape(viz_x.shape[0], -1)
        x = torch.cat([viz_x, proprio_x], dim=-1)

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class ResnetMultiModalNCameraTransformerTensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        resnet_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        # goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        # combiner_hidden_out_dims: Tuple[int, int] = (64, 64),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        add_pos_emb: bool = False,
        add_modal_emb: bool = False,
    ) -> None:
        super().__init__()
        self.resnet_uuids = resnet_preprocessor_uuids
        self.number_cameras = len(resnet_preprocessor_uuids)
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        # self.combine_hid_out_dims = combiner_hidden_out_dims
        self.proprio_hid_out_dims = proprio_hidden_out_dims

        self.add_pos_emb = add_pos_emb
        self.add_modal_emb = add_modal_emb
        

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.resnet_uuids])
        )
        assert not self.blind
        # if not self.blind:
        self.resnet_compressors = nn.ModuleList()
        # self.target_obs_combiners = nn.ModuleList()
        self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuids[0]].shape

        self.trans_layers = nn.ModuleList()

        for cam in self.resnet_uuids:
            resnet_output_shape = observation_spaces.spaces[cam].shape
            assert resnet_output_shape == self.resnet_tensor_shape # annoying if untrue

            self.resnet_compressors.append(nn.Sequential(
                nn.Conv2d(resnet_output_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                # nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                # nn.ReLU(),
            ))

            # self.target_obs_combiners.append(nn.Sequential(
            #     nn.Conv2d(
            #         self.resnet_hid_out_dims[1],
            #         self.combine_hid_out_dims[0],
            #         1,
            #     ),
            #     nn.ReLU(),
            #     nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            #     nn.ReLU(),
            # ))
        self.visual_tokens = self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2]
        self.visual_latent_dim = self.resnet_hid_out_dims[0]

        get_logger().warning(self.resnet_tensor_shape)
        get_logger().warning(transformer_params)
        for n_head, dim_feedforward in transformer_params:
            visual_att_layer = nn.TransformerEncoderLayer(
                self.visual_latent_dim, n_head, dim_feedforward,
                dropout=0
            )
            self.trans_layers.append(visual_att_layer)
        

        self.proprioception_uuid = proprioception_uuid
        get_logger().warning(observation_spaces.spaces[self.proprioception_uuid].shape)
        self.proprioception_head = nn.Sequential(
            nn.Linear(np.prod(observation_spaces.spaces[self.proprioception_uuid].shape), self.proprio_hid_out_dims[0]),
            nn.ReLU(),
            nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
            nn.ReLU(),
        )
        # self.init_weights(self.proprioception_head, [np.sqrt(2)] * 2)


    @ staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
        enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        # if self.blind:
        #     return self.goal_embed_dims
        # else:
        return (
            (self.number_cameras + 1)
            * self.visual_latent_dim
            # * self.resnet_tensor_shape[2] + self.proprio_hid_out_dims[1]
        )

    def compress_resnet(self,observations,idx):
        return self.resnet_compressors[idx](observations[self.resnet_uuids[idx]])

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
        
        # observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent


    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def vis_convert_to_tokens(self, observations, idx):
        vis_tokens = self.compress_resnet(observations, idx)
        # get_logger().warning(f"Vis Tokens Shape: {vis_tokens.shape}")
        vis_tokens = vis_tokens.reshape(
            -1, self.visual_latent_dim, self.visual_tokens
        )
        vis_tokens = vis_tokens.permute(2, 0, 1)
        return vis_tokens

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        # if self.blind:
        #     return self.embed_goal(observations[self.goal_uuid])
        
        viz_x = []
        for cam_idx in range(len(self.resnet_uuids)):
            viz_embs = self.vis_convert_to_tokens(observations, cam_idx)
            viz_x.append(viz_embs)
        batch_size = viz_x[0].shape[1]
        # viz_x = torch.cat(viz_x,dim=1)
        # viz_x = viz_x.reshape(viz_x.shape[0], -1)  # flatten

        # get_logger().warning(viz_x.shape)
        # get_logger().warning(observations[self.proprioception_uuid].shape)
        # get_logger().warning(observations[self.proprioception_uuid].dtype)
        proprio_x = self.proprioception_head(observations[self.proprioception_uuid])
        proprio_x = proprio_x.reshape(batch_size, -1).unsqueeze(0)
        viz_x.append(proprio_x)
        # get_logger().warning(viz_x[0].shape)
        # get_logger().warning(viz_x[1].shape)
        # get_logger().warning(viz_x[2].shape)
        x = torch.cat(viz_x, dim=0)

        for att_layer in self.trans_layers:
            x = att_layer(x)
        final_tokens = []
        for cam_idx in range(len(self.resnet_uuids)):
            cam_token = x[cam_idx * self.visual_tokens: (cam_idx + 1) * self.visual_tokens].mean(dim=0)
            # get_logger().warning(f"{cam_idx * self.visual_tokens}, {(cam_idx + 1) * self.visual_tokens}")
            # get_logger().warning(cam_token)
            final_tokens.append(cam_token)
        final_tokens.append(x[-1])
        # get_logger().warning(x[-1])
        x = torch.cat(final_tokens, dim=-1)
        x = x.reshape(batch_size, -1)
        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)



class ResnetMultiModalNCameraTransformerHistoryTensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        resnet_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        # goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (32, 32),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        add_pos_emb: bool = False,
        add_modal_emb: bool = False,
        add_his_emb: bool = False,
        his_len: int=5,
    ) -> None:
        super().__init__()
        self.resnet_uuids = resnet_preprocessor_uuids
        self.number_cameras = len(resnet_preprocessor_uuids)
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.proprio_hid_out_dims = proprio_hidden_out_dims

        self.add_pos_emb = add_pos_emb
        self.add_modal_emb = add_modal_emb
        self.add_his_emb = add_his_emb

        self.his_len = his_len

        if self.add_his_emb:
            self.his_emb = torch.nn.Embedding(
                num_embeddings=self.his_len,
                embedding_dim=self.proprio_hid_out_dims[1]
            )

        if self.add_modal_emb:
            # Nav / Manip / Proprio
            self.modal_emb = torch.nn.Embedding(
                num_embeddings=3,
                embedding_dim=self.proprio_hid_out_dims[1]
            )
            # self.manip_emb = torch.nn.Parameter(
            #     torch.rand()
            # )

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.resnet_uuids])
        )
        assert not self.blind
        # if not self.blind:
        self.resnet_compressors = nn.ModuleList()
        self.target_obs_combiners = nn.ModuleList()
        self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuids[0]].shape

        self.trans_layers = nn.ModuleList()
        self.vision_compressors = nn.ModuleList()
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
                    self.resnet_hid_out_dims[1],
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
                nn.ReLU(),
            ))
            # self.add_vision_compressor:
            get_logger().warning(self.combine_hid_out_dims[1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2])
            get_logger().warning(self.proprio_hid_out_dims[1])
            self.vision_compressors.append(nn.Sequential(
                nn.Linear(
                    self.combine_hid_out_dims[1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2],
                    self.proprio_hid_out_dims[1]
                ),
                nn.ReLU()
            ))
        self.visual_tokens = self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2] * self.his_len
        self.visual_latent_dim = self.proprio_hid_out_dims[1]

        self.proprioception_uuid = proprioception_uuid
        get_logger().warning(observation_spaces.spaces[self.proprioception_uuid].shape)
        get_logger().warning(self.his_len)
        get_logger().warning(np.prod(observation_spaces.spaces[self.proprioception_uuid].shape) // self.his_len)
        # get_logger().warning()
        self.proprioception_head = nn.Sequential(
            nn.Linear(
                np.prod(observation_spaces.spaces[self.proprioception_uuid].shape) // self.his_len,
                self.proprio_hid_out_dims[0]
            ),
            nn.ReLU(),
            nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
            nn.ReLU(),
        )

        get_logger().warning(self.resnet_tensor_shape)
        get_logger().warning(transformer_params)
        for n_head, dim_feedforward in transformer_params:
            visual_att_layer = nn.TransformerEncoderLayer(
                self.visual_latent_dim, n_head, dim_feedforward,
                dropout=0
            )
            self.trans_layers.append(visual_att_layer)

        # self.init_weights(self.proprioception_head, [np.sqrt(2)] * 2)


    @ staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
        enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        return (
            (self.number_cameras + 1)
            * self.visual_latent_dim
            # * self.resnet_tensor_shape[2] + self.proprio_hid_out_dims[1]
        )

    def compress_resnet(self,observations,idx):
        batch_size = observations[self.resnet_uuids[idx]].shape[0]
        x = observations[self.resnet_uuids[idx]].reshape(
            batch_size * self.his_len,
            *self.resnet_tensor_shape
        )
        x = self.target_obs_combiners[idx](self.resnet_compressors[idx](
            x
        ))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.vision_compressors[idx](x)
        return x
            

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
        
        # observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent


    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def vis_convert_to_tokens(self, observations, idx):
        vis_tokens = self.compress_resnet(observations, idx)
        # get_logger().warning(f"Vis Tokens Shape: {vis_tokens.shape}")
        batch_size_with_his_len = vis_tokens.shape[0]
        vis_tokens = vis_tokens.reshape(
            batch_size_with_his_len // self.his_len,
            self.his_len,
            self.visual_latent_dim
        )
        vis_tokens = vis_tokens.permute(1, 0, 2)
        if self.add_his_emb:
            current_his_emb = self.his_emb(
                torch.arange(self.his_len).to(vis_tokens.device),
            ).unsqueeze(1)
            vis_tokens = vis_tokens + current_his_emb
        return vis_tokens

    def proprio_convert_to_tokens(self, batch_size, observations):
        # batch_size = observations[self.proprioception_uuid].shape[0]
        # Proprioception has 5 dim
        # get_logger().warning(batch_size)
        # get_logger().warning(observations[self.proprioception_uuid].shape)
        x = observations[self.proprioception_uuid].reshape(
            batch_size, self.his_len, 5
        )
        proprio_x = x.permute(1, 0, 2)
        proprio_tokens = self.proprioception_head(proprio_x)
        # proprio_x = proprio_x.reshape(batch_size, -1).unsqueeze(0)
        # proprio_tokens = proprio_x
        if self.add_his_emb:
            current_his_emb = self.his_emb(
                torch.arange(self.his_len).to(proprio_tokens.device),
            ).unsqueeze(1)
            proprio_tokens = proprio_tokens + current_his_emb
        return proprio_tokens


    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        # if self.blind:
        #     return self.embed_goal(observations[self.goal_uuid])
        
        viz_x = []
        for cam_idx in range(len(self.resnet_uuids)):
            viz_embs = self.vis_convert_to_tokens(observations, cam_idx)
            if self.add_modal_emb:
                current_modal_emb = self.modal_emb(
                    torch.Tensor([cam_idx]).long().to(viz_embs.device)
                ).unsqueeze(0)
                viz_embs = viz_embs + current_modal_emb
            viz_x.append(viz_embs)
        batch_size = viz_x[0].shape[1]

        # get_logger().warning(viz_x.shape)
        # get_logger().warning(observations[self.proprioception_uuid].shape)
        # get_logger().warning(observations[self.proprioception_uuid].dtype)
        # proprio_x = self.proprioception_head(observations[self.proprioception_uuid])
        # proprio_x = proprio_x.reshape(batch_size, -1).unsqueeze(0)
        # proprio_x = proprio_x.reshape(batch_size, -1)
        proprio_tokens = self.proprio_convert_to_tokens(batch_size, observations)
        if self.add_modal_emb:
            current_modal_emb = self.modal_emb(
                torch.Tensor([2]).long().to(proprio_tokens.device)
            ).unsqueeze(0)
            proprio_tokens = proprio_tokens + current_modal_emb
        viz_x.append(proprio_tokens)
        # get_logger().warning(viz_x[0].shape)
        # get_logger().warning(viz_x[1].shape)
        # get_logger().warning(viz_x[2].shape)
        x = torch.cat(viz_x, dim=0)

        for att_layer in self.trans_layers:
            x = att_layer(x)
        final_tokens = []
        for cam_idx in range(len(self.resnet_uuids)):
            cam_token = x[
                cam_idx * self.his_len: (cam_idx + 1) * self.his_len
            ].mean(dim=0)
            # get_logger().warning(f"{cam_idx * self.visual_tokens}, {(cam_idx + 1) * self.visual_tokens}")
            # get_logger().warning(cam_token)
            final_tokens.append(cam_token)
        final_tokens.append(
            x[-self.his_len:].mean(dim=0)
        )
        # get_logger().warning(x[-1])
        x = torch.cat(final_tokens, dim=-1)
        x = x.reshape(batch_size, -1)
        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class ResnetMultiModalNCameraTransformerHistoryV2TensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        resnet_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        # goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (32, 32),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        add_pos_emb: bool = False,
        add_modal_emb: bool = False,
        add_his_emb: bool = False,
        his_len: int=5,
    ) -> None:
        super().__init__()
        self.resnet_uuids = resnet_preprocessor_uuids
        self.number_cameras = len(resnet_preprocessor_uuids)
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.proprio_hid_out_dims = proprio_hidden_out_dims

        self.add_pos_emb = add_pos_emb
        self.add_modal_emb = add_modal_emb
        self.add_his_emb = add_his_emb

        self.his_len = his_len

        if self.add_his_emb:
            self.his_emb = torch.nn.Embedding(
                num_embeddings=self.his_len,
                embedding_dim=self.proprio_hid_out_dims[1]
            )

        if self.add_modal_emb:
            # Nav / Manip / Proprio
            self.modal_emb = torch.nn.Embedding(
                num_embeddings=3,
                embedding_dim=self.proprio_hid_out_dims[1]
            )

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.resnet_uuids])
        )
        assert not self.blind
        # if not self.blind:
        self.resnet_compressors = nn.ModuleList()
        self.target_obs_combiners = nn.ModuleList()
        self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuids[0]].shape

        self.trans_layers = nn.ModuleList()
        self.vision_compressors = nn.ModuleList()
        for cam in self.resnet_uuids:
            resnet_output_shape = observation_spaces.spaces[cam].shape
            assert resnet_output_shape == self.resnet_tensor_shape # annoying if untrue

            self.resnet_compressors.append(nn.Sequential(
                nn.Conv2d(resnet_output_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            ))
            # self.add_vision_compressor:
            self.vision_compressors.append(nn.Sequential(
                nn.Linear(
                    self.resnet_hid_out_dims[1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2],
                    self.proprio_hid_out_dims[1]
                ),
                nn.ReLU()
            ))
        self.visual_tokens = self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2] * self.his_len
        self.visual_latent_dim = self.proprio_hid_out_dims[1]

        self.proprioception_uuid = proprioception_uuid
        self.proprioception_head = nn.Sequential(
            nn.Linear(
                np.prod(observation_spaces.spaces[self.proprioception_uuid].shape) // self.his_len,
                self.proprio_hid_out_dims[0]
            ),
            nn.ReLU(),
            nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
            nn.ReLU(),
        )

        get_logger().warning(self.resnet_tensor_shape)
        get_logger().warning(transformer_params)
        for n_head, dim_feedforward in transformer_params:
            visual_att_layer = nn.TransformerEncoderLayer(
                self.visual_latent_dim, n_head, dim_feedforward,
                dropout=0
            )
            self.trans_layers.append(visual_att_layer)


    @ staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
        enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        return (
            (self.number_cameras + 1)
            * self.visual_latent_dim
            # * self.resnet_tensor_shape[2] + self.proprio_hid_out_dims[1]
        )

    def compress_resnet(self,observations,idx):
        batch_size = observations[self.resnet_uuids[idx]].shape[0]
        x = observations[self.resnet_uuids[idx]].reshape(
            batch_size * self.his_len,
            *self.resnet_tensor_shape
        )
        x = self.resnet_compressors[idx](
            x
        )
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.vision_compressors[idx](x)
        return x
            

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
        
        # observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent


    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def vis_convert_to_tokens(self, observations, idx):
        vis_tokens = self.compress_resnet(observations, idx)
        # get_logger().warning(f"Vis Tokens Shape: {vis_tokens.shape}")
        batch_size_with_his_len = vis_tokens.shape[0]
        vis_tokens = vis_tokens.reshape(
            batch_size_with_his_len // self.his_len,
            self.his_len,
            self.visual_latent_dim
        )
        vis_tokens = vis_tokens.permute(1, 0, 2)
        if self.add_his_emb:
            current_his_emb = self.his_emb(
                torch.arange(self.his_len).to(vis_tokens.device),
            ).unsqueeze(1)
            vis_tokens = vis_tokens + current_his_emb
        return vis_tokens

    def proprio_convert_to_tokens(self, batch_size, observations):
        # batch_size = observations[self.proprioception_uuid].shape[0]
        # Proprioception has 5 dim
        # get_logger().warning(batch_size)
        # get_logger().warning(observations[self.proprioception_uuid].shape)
        x = observations[self.proprioception_uuid].reshape(
            batch_size, self.his_len, 5
        )
        proprio_x = x.permute(1, 0, 2)
        proprio_tokens = self.proprioception_head(proprio_x)
        # proprio_x = proprio_x.reshape(batch_size, -1).unsqueeze(0)
        # proprio_tokens = proprio_x
        if self.add_his_emb:
            current_his_emb = self.his_emb(
                torch.arange(self.his_len).to(proprio_tokens.device),
            ).unsqueeze(1)
            proprio_tokens = proprio_tokens + current_his_emb
        return proprio_tokens

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        # if self.blind:
        #     return self.embed_goal(observations[self.goal_uuid])
        
        viz_x = []
        for cam_idx in range(len(self.resnet_uuids)):
            viz_embs = self.vis_convert_to_tokens(observations, cam_idx)
            if self.add_modal_emb:
                current_modal_emb = self.modal_emb(
                    torch.Tensor([cam_idx]).long().to(viz_embs.device)
                ).unsqueeze(0)
                viz_embs = viz_embs + current_modal_emb
            viz_x.append(viz_embs)
        batch_size = viz_x[0].shape[1]

        # proprio_x = self.proprioception_head(observations[self.proprioception_uuid])
        # proprio_x = proprio_x.reshape(batch_size, -1).unsqueeze(0)
        # proprio_x = proprio_x.reshape(batch_size, -1)
        proprio_tokens = self.proprio_convert_to_tokens(batch_size, observations)
        if self.add_modal_emb:
            current_modal_emb = self.modal_emb(
                torch.Tensor([2]).long().to(proprio_tokens.device)
            ).unsqueeze(0)
            proprio_tokens = proprio_tokens + current_modal_emb
        viz_x.append(proprio_tokens)
        # get_logger().warning(viz_x[0].shape)
        # get_logger().warning(viz_x[1].shape)
        # get_logger().warning(viz_x[2].shape)
        x = torch.cat(viz_x, dim=0)

        for att_layer in self.trans_layers:
            x = att_layer(x)
        final_tokens = []
        for cam_idx in range(len(self.resnet_uuids)):
            cam_token = x[(cam_idx + 1) * self.his_len - 1]
            final_tokens.append(cam_token)
        final_tokens.append(x[-1])
        # get_logger().warning(x[-1])
        x = torch.cat(final_tokens, dim=-1)
        x = x.reshape(batch_size, -1)
        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class ResnetMultiModalNCameraTransformerHistoryV3TensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        resnet_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        # goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (32, 32),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        add_pos_emb: bool = False,
        add_modal_emb: bool = False,
        add_his_emb: bool = False,
        his_len: int=5,
    ) -> None:
        super().__init__()
        self.resnet_uuids = resnet_preprocessor_uuids
        self.number_cameras = len(resnet_preprocessor_uuids)
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.proprio_hid_out_dims = proprio_hidden_out_dims

        self.add_pos_emb = add_pos_emb
        self.add_modal_emb = add_modal_emb
        self.add_his_emb = add_his_emb

        self.his_len = his_len

        if self.add_his_emb:
            self.his_emb = torch.nn.Embedding(
                num_embeddings=self.his_len,
                embedding_dim=self.proprio_hid_out_dims[1]
            )

        if self.add_modal_emb:
            # Nav / Manip / Proprio
            self.modal_emb = torch.nn.Embedding(
                num_embeddings=3,
                embedding_dim=self.proprio_hid_out_dims[1]
            )

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.resnet_uuids])
        )
        assert not self.blind
        # if not self.blind:
        self.resnet_compressors = nn.ModuleList()
        self.target_obs_combiners = nn.ModuleList()
        self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuids[0]].shape

        self.trans_layers = nn.ModuleList()
        self.vision_compressors = nn.ModuleList()
        for cam in self.resnet_uuids:
            resnet_output_shape = observation_spaces.spaces[cam].shape
            assert resnet_output_shape == self.resnet_tensor_shape # annoying if untrue

            self.resnet_compressors.append(nn.Sequential(
                nn.Conv2d(resnet_output_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            ))
            # self.add_vision_compressor:
            self.vision_compressors.append(nn.Sequential(
                nn.Linear(
                    self.resnet_hid_out_dims[1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2],
                    self.proprio_hid_out_dims[1]
                ),
                nn.ReLU()
            ))
        self.visual_tokens = self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2] * self.his_len
        self.visual_latent_dim = self.proprio_hid_out_dims[1]

        self.proprioception_uuid = proprioception_uuid
        self.proprioception_head = nn.Sequential(
            nn.Linear(
                np.prod(observation_spaces.spaces[self.proprioception_uuid].shape) // self.his_len,
                self.proprio_hid_out_dims[0]
            ),
            nn.ReLU(),
            nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
            nn.ReLU(),
        )

        get_logger().warning(self.resnet_tensor_shape)
        get_logger().warning(transformer_params)
        for n_head, dim_feedforward in transformer_params:
            visual_att_layer = nn.TransformerEncoderLayer(
                self.visual_latent_dim, n_head, dim_feedforward,
                dropout=0
            )
            self.trans_layers.append(visual_att_layer)


    @ staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
        enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        return (
            (self.number_cameras + 1)
            * self.visual_latent_dim
            # * self.resnet_tensor_shape[2] + self.proprio_hid_out_dims[1]
        )

    def compress_resnet(self,observations,idx):
        batch_size = observations[self.resnet_uuids[idx]].shape[0]
        x = observations[self.resnet_uuids[idx]].reshape(
            batch_size * self.his_len,
            *self.resnet_tensor_shape
        )
        x = self.resnet_compressors[idx](
            x
        )
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.vision_compressors[idx](x)
        return x
            

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
        
        # observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent


    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def vis_convert_to_tokens(self, observations, idx):
        vis_tokens = self.compress_resnet(observations, idx)
        # get_logger().warning(f"Vis Tokens Shape: {vis_tokens.shape}")
        batch_size_with_his_len = vis_tokens.shape[0]
        vis_tokens = vis_tokens.reshape(
            batch_size_with_his_len // self.his_len,
            self.his_len,
            self.visual_latent_dim
        )
        vis_tokens = vis_tokens.permute(1, 0, 2)
        if self.add_his_emb:
            current_his_emb = self.his_emb(
                torch.arange(self.his_len).to(vis_tokens.device),
            ).unsqueeze(1)
            vis_tokens = vis_tokens + current_his_emb
        return vis_tokens

    def proprio_convert_to_tokens(self, batch_size, observations):
        # batch_size = observations[self.proprioception_uuid].shape[0]
        # Proprioception has 5 dim
        # get_logger().warning(batch_size)
        # get_logger().warning(observations[self.proprioception_uuid].shape)
        x = observations[self.proprioception_uuid].reshape(
            batch_size, self.his_len, 5
        )
        proprio_x = x.permute(1, 0, 2)
        proprio_tokens = self.proprioception_head(proprio_x)
        # proprio_x = proprio_x.reshape(batch_size, -1).unsqueeze(0)
        # proprio_tokens = proprio_x
        if self.add_his_emb:
            current_his_emb = self.his_emb(
                torch.arange(self.his_len).to(proprio_tokens.device),
            ).unsqueeze(1)
            proprio_tokens = proprio_tokens + current_his_emb
        return proprio_tokens

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        # if self.blind:
        #     return self.embed_goal(observations[self.goal_uuid])
        
        viz_x = []
        for cam_idx in range(len(self.resnet_uuids)):
            viz_embs = self.vis_convert_to_tokens(observations, cam_idx)
            if self.add_modal_emb:
                current_modal_emb = self.modal_emb(
                    torch.Tensor([cam_idx]).long().to(viz_embs.device)
                ).unsqueeze(0)
                viz_embs = viz_embs + current_modal_emb
            viz_x.append(viz_embs)
        batch_size = viz_x[0].shape[1]

        # proprio_x = self.proprioception_head(observations[self.proprioception_uuid])
        # proprio_x = proprio_x.reshape(batch_size, -1).unsqueeze(0)
        # proprio_x = proprio_x.reshape(batch_size, -1)
        proprio_tokens = self.proprio_convert_to_tokens(batch_size, observations)
        if self.add_modal_emb:
            current_modal_emb = self.modal_emb(
                torch.Tensor([2]).long().to(proprio_tokens.device)
            ).unsqueeze(0)
            proprio_tokens = proprio_tokens + current_modal_emb
        viz_x.append(proprio_tokens)
        # get_logger().warning(viz_x[0].shape)
        # get_logger().warning(viz_x[1].shape)
        # get_logger().warning(viz_x[2].shape)
        x = torch.cat(viz_x, dim=0)

        for att_layer in self.trans_layers:
            x = att_layer(x)
        final_tokens = []
        for cam_idx in range(len(self.resnet_uuids)):
            cam_token = x[
                cam_idx * self.his_len: (cam_idx + 1) * self.his_len
            ].mean(dim=0)
            final_tokens.append(cam_token)
        final_tokens.append(
            x[-self.his_len:].mean(dim=0)
        )
        # get_logger().warning(x[-1])
        x = torch.cat(final_tokens, dim=-1)
        x = x.reshape(batch_size, -1)
        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class ResnetMultiModalNCameraTransformerHistoryMergeTensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        # goal_sensor_uuid: str,
        resnet_preprocessor_uuids: List[str],
        proprioception_uuid: str,
        # goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int] = (64, 64),
        combiner_hidden_out_dims: Tuple[int, int] = (32, 32),
        proprio_hidden_out_dims: Tuple[int, int] = (64, 64),
        transformer_params: Tuple[Tuple[int, int]] = ((8, 256), (8, 256)),
        # add_pos_emb: bool = False,
        # add_modal_emb: bool = False,
        add_his_emb: bool = False,
        his_len: int=5,
    ) -> None:
        super().__init__()
        self.resnet_uuids = resnet_preprocessor_uuids
        self.number_cameras = len(resnet_preprocessor_uuids)
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.proprio_hid_out_dims = proprio_hidden_out_dims

        # self.add_pos_emb = add_pos_emb
        # self.add_modal_emb = add_modal_emb
        self.add_his_emb = add_his_emb

        self.his_len = his_len

        # if self.add_modal_emb:
        #     # Nav / Manip / Proprio
        #     self.modal_emb = torch.nn.Embedding(
        #         num_embeddings=3,
        #         embedding_dim=self.proprio_hid_out_dims[1]
        #     )
            # self.manip_emb = torch.nn.Parameter(
            #     torch.rand()
            # )

        self.blind = (
            any([x not in observation_spaces.spaces for x in self.resnet_uuids])
        )
        assert not self.blind
        # if not self.blind:
        self.resnet_compressors = nn.ModuleList()
        self.target_obs_combiners = nn.ModuleList()
        self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuids[0]].shape

        self.trans_layers = nn.ModuleList()
        self.vision_compressors = nn.ModuleList()
        for cam in self.resnet_uuids:
            resnet_output_shape = observation_spaces.spaces[cam].shape
            assert resnet_output_shape == self.resnet_tensor_shape # annoying if untrue

            self.resnet_compressors.append(nn.Sequential(
                nn.Conv2d(resnet_output_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            ))
            # self.target_obs_combiners.append(nn.Sequential(
            #     nn.Conv2d(
            #         self.resnet_hid_out_dims[1],
            #         self.combine_hid_out_dims[0],
            #         1,
            #     ),
            #     nn.ReLU(),
            #     nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            #     nn.ReLU(),
            # ))
            # self.add_vision_compressor:
            # get_logger().warning(self.combine_hid_out_dims[1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2])
            # get_logger().warning(self.proprio_hid_out_dims[1])
            self.vision_compressors.append(nn.Sequential(
                nn.Linear(
                    self.resnet_hid_out_dims[1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2],
                    self.proprio_hid_out_dims[1]
                ),
                nn.ReLU()
            ))
        # self.visual_tokens = self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2] * self.his_len
        self.latent_dim = self.proprio_hid_out_dims[1] * 3
        self.visual_latent_dim = self.proprio_hid_out_dims[1]

        self.proprioception_uuid = proprioception_uuid
        self.proprioception_head = nn.Sequential(
            nn.Linear(
                np.prod(observation_spaces.spaces[self.proprioception_uuid].shape) // self.his_len,
                self.proprio_hid_out_dims[0]
            ),
            nn.ReLU(),
            nn.Linear(self.proprio_hid_out_dims[0], self.proprio_hid_out_dims[1]),
            nn.ReLU(),
        )

        if self.add_his_emb:
            self.his_emb = torch.nn.Embedding(
                num_embeddings=self.his_len,
                embedding_dim=self.latent_dim
            )

        get_logger().warning(self.resnet_tensor_shape)
        get_logger().warning(transformer_params)
        for n_head, dim_feedforward in transformer_params:
            visual_att_layer = nn.TransformerEncoderLayer(
                self.latent_dim, n_head, dim_feedforward,
                dropout=0
            )
            self.trans_layers.append(visual_att_layer)

        # self.init_weights(self.proprioception_head, [np.sqrt(2)] * 2)


    @ staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
        enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        return self.latent_dim

    def compress_resnet(self,observations,idx):
        batch_size = observations[self.resnet_uuids[idx]].shape[0]
        x = observations[self.resnet_uuids[idx]].reshape(
            batch_size * self.his_len,
            *self.resnet_tensor_shape
        )
        # x = self.target_obs_combiners[idx](self.resnet_compressors[idx](
        #     x
        # ))
        x = self.resnet_compressors[idx](
            x
        )
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.vision_compressors[idx](x)
        return x
            

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
        
        # observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent


    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def vis_convert_to_tokens(self, observations, idx):
        vis_tokens = self.compress_resnet(observations, idx)
        # get_logger().warning(f"Vis Tokens Shape: {vis_tokens.shape}")
        batch_size_with_his_len = vis_tokens.shape[0]
        vis_tokens = vis_tokens.reshape(
            batch_size_with_his_len // self.his_len,
            self.his_len,
            self.visual_latent_dim
        )
        vis_tokens = vis_tokens.permute(1, 0, 2)
        # if self.add_his_emb:
        #     current_his_emb = self.his_emb(
        #         torch.arange(self.his_len).to(vis_tokens.device),
        #     ).unsqueeze(1)
        #     vis_tokens = vis_tokens + current_his_emb
        return vis_tokens

    def proprio_convert_to_tokens(self, batch_size, observations):
        # batch_size = observations[self.proprioception_uuid].shape[0]
        # Proprioception has 5 dim
        # get_logger().warning(batch_size)
        # get_logger().warning(observations[self.proprioception_uuid].shape)
        x = observations[self.proprioception_uuid].reshape(
            batch_size, self.his_len, 5
        )
        proprio_x = x.permute(1, 0, 2)
        proprio_tokens = self.proprioception_head(proprio_x)
        # proprio_x = proprio_x.reshape(batch_size, -1).unsqueeze(0)
        # proprio_tokens = proprio_x
        # if self.add_his_emb:
        #     current_his_emb = self.his_emb(
        #         torch.arange(self.his_len).to(proprio_tokens.device),
        #     ).unsqueeze(1)
        #     proprio_tokens = proprio_tokens + current_his_emb
        return proprio_tokens


    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )
        
        viz_x = []
        for cam_idx in range(len(self.resnet_uuids)):
            viz_embs = self.vis_convert_to_tokens(observations, cam_idx)
            # if self.add_modal_emb:
            #     current_modal_emb = self.modal_emb(
            #         torch.Tensor([cam_idx]).long().to(viz_embs.device)
            #     ).unsqueeze(0)
            #     viz_embs = viz_embs + current_modal_emb
            viz_x.append(viz_embs)
        batch_size = viz_x[0].shape[1]

        proprio_tokens = self.proprio_convert_to_tokens(batch_size, observations)
        # if self.add_modal_emb:
        #     current_modal_emb = self.modal_emb(
        #         torch.Tensor([2]).long().to(proprio_tokens.device)
        #     ).unsqueeze(0)
        #     proprio_tokens = proprio_tokens + current_modal_emb
        viz_x.append(proprio_tokens)
        # get_logger().warning(viz_x[0].shape)
        # get_logger().warning(viz_x[1].shape)
        # get_logger().warning(viz_x[2].shape)
        x = torch.cat(viz_x, dim=-1)
        if self.add_his_emb:
            current_his_emb = self.his_emb(
                torch.arange(self.his_len).to(proprio_tokens.device),
            ).unsqueeze(1)
            x = x + current_his_emb

        for att_layer in self.trans_layers:
            x = att_layer(x)
        # final_tokens = []
        # for cam_idx in range(len(self.resnet_uuids)):
        #     cam_token = x[
        #         cam_idx * self.his_len: (cam_idx + 1) * self.his_len
        #     ].mean(dim=0)
        #     # get_logger().warning(f"{cam_idx * self.visual_tokens}, {(cam_idx + 1) * self.visual_tokens}")
        #     # get_logger().warning(cam_token)
        #     final_tokens.append(cam_token)
        # final_tokens.append(
        #     x[-self.his_len:].mean(dim=0)
        # )
        # get_logger().warning(x[-1])
        # x = torch.cat(final_tokens, dim=-1)
        # x = x.reshape(batch_size, -1)

        # Only Take the Last Token  -> Which Corresponding to most recent one
        # x = x[-1, ...]
        x = x[-1]
        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)
