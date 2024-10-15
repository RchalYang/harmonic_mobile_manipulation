from collections import OrderedDict
from typing import Any, Tuple, Optional, Dict, cast
import os

import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from gym.spaces import Dict as SpaceDict
import torch
from sklearn.metrics import precision_score, recall_score

from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic
)
from allenact.algorithms.onpolicy_sync.policy import DistributionType, ObservationType
from allenact.base_abstractions.misc import ActorCriticOutput,Memory
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss


from utils.utils import log_ac_return,log_map_output, ForkedPdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg') # non-interactive backend for thread-safety 


### borrowed from
### https://raw.githubusercontent.com/xanderchf/MonoDepth-FPN-PyTorch/master/model_fpn.py
def upshuffle(in_planes, out_planes, upscale_factor, kernel_size=2, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
        nn.LeakyReLU()
    )

def upshufflenorelu(in_planes, out_planes, upscale_factor, kernel_size=2, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
    )

def downblock(in_channels, out_channels): # should this be replaced with a pretrained resnet? was in upshuffle example
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=1),
        nn.ReLU(inplace=True),
    ) 

def combine_block_w_do(in_planes, out_planes, dropout=0.):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    )


def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode='bilinear') + y


class ResnetTensorNavMappingActorCritic(ResnetTensorNavActorCritic):
    def __init__(self, visualize=False, **kwargs: Any,):
        super().__init__(**kwargs)
        self.visualize = visualize
        self.map_module = MappingPredictorUNet(
                self.observation_space,
                kwargs['rgb_resnet_preprocessor_uuid'],
            )
    
    def load_state_dict(self, state_dict, **kwargs):
        kwargs['strict'] = False # allows for the mapping module to init randomly
        return super().load_state_dict(state_dict, **kwargs)  


    def forward(self, observations: ObservationType, memory: Memory, prev_actions: torch.Tensor, masks: torch.FloatTensor) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        clip_embedding = observations['rgb_clip_resnet'] # [1, 2048, 7, 7]

        # with torch.no_grad():

        ######## identical copy from visual_nav_models.py - so allenact doesn't need to be changed #################
        
        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds = self.forward_encoder(observations)

        # 1.2 use embedding model to get prev_action embeddings
        if self.prev_action_embedder.input_size == self.action_space.n + 1:
            # In this case we have a unique embedding for the start of an episode
            prev_actions_embeds = self.prev_action_embedder(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions + 1,
                    other=torch.zeros_like(prev_actions),
                )
            )
        else:
            prev_actions_embeds = self.prev_action_embedder(prev_actions)
        joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)

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

        extras['hidden_layer'] = beliefs
        extras['prev_action_embeds'] = prev_actions_embeds

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=self.critic(beliefs),
            extras=extras,
        )

        ########## identical copy ends ##############################################

        
        # actor_critic_output.extras['map_estimate'] = self.map_module(observations, 
        #                                                             beliefs.detach(), 
        #                                                             prev_actions_embeds.detach(), 
        #                                                             clip_embedding)
        
        if self.visualize:
            log_ac_return(actor_critic_output,observations["task_id_sensor"])
            # log_map_output(actor_critic_output,observations["task_id_sensor"])

        return actor_critic_output, memory

class MapWrapActorCritic(ResnetTensorNavActorCritic):
    def __init__(self, visualize=False, **kwargs: Any,):
        super().__init__(**kwargs)
        self.visualize=visualize
        self.inner_rnn_with_pred_map = ResnetTensorNavMappingActorCritic(visualize=False,**kwargs)
        self.aggregate_map_encoder = MapInputEncoder(
                self.observation_space,
                kwargs['rgb_resnet_preprocessor_uuid'],
            )
        
        self.create_state_encoders(
            obs_embed_size=1500-6,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=True,
            add_prev_action_null_token=False,
            prev_action_embed_size=6,
        )

        self.goal_uuid = kwargs['goal_sensor_uuid']
        self.goal_embed_dims = 32
        self.goal_space = kwargs['observation_space'].spaces[self.goal_uuid]
        self.embed_goal = nn.Embedding(
            num_embeddings=self.goal_space.n, embedding_dim=self.goal_embed_dims,
        )

        # hidden layer plus map encoding plus inner prev action plus clip compress plus target
        self.pointwise_conv = combine_block_w_do(512+128+6+32+32, 1500, dropout=0)
    
    def load_state_dict(self, state_dict, **kwargs):
        kwargs['strict'] = False # TODO this will probably need some work to function correctly
        # modify state dict to prepend inner_rnn_with_pred_map to everything
        if any('inner_rnn_with_pred_map' in s for s in state_dict.keys()):
            super().load_state_dict(state_dict, **kwargs)
        else:
            new_state_dict = OrderedDict()
            for key in state_dict.keys():
                new_key = "inner_rnn_with_pred_map." + key
                new_state_dict[new_key] = state_dict[key]
            try:
                return super().load_state_dict(new_state_dict, **kwargs)
            except:
                ForkedPdb().set_trace()
                pass
        

    def _recurrent_memory_specification(self):
        rms = super()._recurrent_memory_specification() 
        rms['inner_rnn'] = ( # the pretrained/frozen RNN
            (
                    ("layer", 1),
                    ("sampler", None),
                    ("hidden", 512),
                ),
                torch.float32,)
        return rms
    
    def outer_rnn_forward(self, observations,memory:Memory, masks, input_vector):

        beliefs, rnn_hidden_states = self.state_encoders['single_belief'](input_vector, memory.tensor('single_belief'), masks)
        memory.set_tensor('single_belief', rnn_hidden_states)  # update memory here

        extras= {'hidden_layer':beliefs}

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=self.critic(beliefs),
            extras=extras,
        )
        if self.visualize:
            log_ac_return(actor_critic_output,observations["task_id_sensor"])

        return actor_critic_output, memory

    
    def forward(self, observations: ObservationType, 
                memory: Memory, 
                prev_actions: torch.Tensor, 
                masks: torch.FloatTensor
            ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        clip_embedding = observations['rgb_clip_resnet'] # preserve this
        inner_hl = memory.tensor('inner_rnn')
        with torch.no_grad():
            inner_rnn_ac,inner_rnn_memory = self.inner_rnn_with_pred_map(observations,Memory({'single_belief':(inner_hl,1)}),prev_actions,masks)
        memory.set_tensor('inner_rnn', inner_rnn_memory.tensor('single_belief'))
        
        # estimated_map = inner_rnn_ac.extras['map_estimate']
        action_embedding = inner_rnn_ac.extras['prev_action_embeds'].detach()
        inner_hidden_layer = inner_rnn_ac.extras['hidden_layer'].detach()

        map_and_clip = self.aggregate_map_encoder(observations,clip_embedding)
        target_emb = self.embed_goal(observations[self.goal_uuid])
        
        # ForkedPdb().set_trace()
        embs = [map_and_clip, target_emb.view(-1, self.goal_embed_dims, 1, 1),
                inner_hidden_layer.view(-1, 512,1,1), action_embedding.view(-1, 6,1,1),]

        x = self.pointwise_conv(torch.cat(embs, dim=1,))
        nstep, nsampler = clip_embedding.shape[:2]
        
        ac_out,memory = self.outer_rnn_forward(observations,memory,masks,x.view(nstep, nsampler, -1))
        return ac_out, memory
        
        # # FOR DEBUGGING PRETRAINED MODEL LOAD ONLY
        # return inner_rnn_ac, memory


class MapInputEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        resnet_preprocessor_uuid: str,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.resnet_uuid = resnet_preprocessor_uuid
        self.map_uuid = "aggregate_map_sensor"
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims

        # want to have a separate clip compressor for localizing
        self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuid].shape
        # ForkedPdb().set_trace()

        self.mapconv_down1 = downblock(3,32)
        self.downsizeenforce = nn.AdaptiveAvgPool2d((40,40))
        self.mapconv_down2 = downblock(32,64)
        self.mapconv_down3 = downblock(64,128)

        self.mapconv_maxpool = nn.MaxPool2d(2)

        self.resnet_compressor = nn.Sequential(
            nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(*self.resnet_hid_out_dims[0:2], kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        ) # output 1, 32, 1, 1


    def adapt_input(self, observations, clip_embedding):
        resnet = clip_embedding
        base_map = observations[self.map_uuid].float()

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        else:
            nstep, nsampler = resnet.shape[:2]
        
        if base_map.shape[1] != 4: # channels not yet permuted & truncated
            observations[self.map_uuid] = torch.permute(base_map.view(-1, *base_map.shape[-3:]),(0,3,1,2))
        else:
           observations[self.map_uuid] = base_map.view(-1, *base_map.shape[-3:])

        return observations, use_agent, nstep, nsampler, nagent
    
    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent): # keep shape instead of flattening
        if use_agent:
            return x.view(nstep, nsampler, nagent,  -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations, clip_embedding, estimated_map=None):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations, clip_embedding
        )

        if estimated_map is not None:
            x = self.mapconv_down1(estimated_map)
        else:
            gt_map = (observations[self.map_uuid][:,1:4,...]>0.0).float()
            
            x = self.mapconv_down1(gt_map)
        conv1 = self.downsizeenforce(x)
        x = self.mapconv_maxpool(conv1)

        conv2 = self.mapconv_down2(x)
        x = self.mapconv_maxpool(conv2)
        
        conv3 = self.mapconv_down3(x)
        x = self.mapconv_maxpool(conv3)   

        clip_compressed = self.compress_resnet(observations)
        x=torch.cat([x,clip_compressed], dim=1,)

        return x #self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class MappingPredictorUNet(MapInputEncoder):
    def __init__(self, 
                 observation_spaces: SpaceDict, 
                 resnet_preprocessor_uuid: str, 
                 resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32)
                 ) -> None:
        super().__init__(observation_spaces, resnet_preprocessor_uuid, resnet_compressor_hidden_out_dims)

        self.mapconv_down1 = downblock(2,32)

        self.pointwise_conv = combine_block_w_do(512+128+6+32, 64, dropout=0)

        self.mapconv_up1 = upshuffle(64, 128, 2, kernel_size=3, stride=1, padding=1)
        self.mapconv_up2 = upshuffle(128, 64, 2, kernel_size=3, stride=1, padding=1)
        self.mapconv_up3 = upshuffle(64, 32, 2, kernel_size=3, stride=1, padding=1)

        self.mapconv_up5 = upshufflenorelu(32, 1, 2, padding=0)

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent): # keep shape instead of flattening
        if use_agent:
            return x.view(nstep, nsampler, nagent,  *x.shape[-2:])
        return x.view(nstep, nsampler * nagent, *x.shape[-2:])

    def forward(self, observations, memory, prev_actions_embeds, clip_embedding):

        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations, clip_embedding
        )

        walls_and_floor = observations[self.map_uuid][:,0:2,...]
        
        x = self.mapconv_down1(walls_and_floor)
        conv1 = self.downsizeenforce(x)
        x = self.mapconv_maxpool(conv1)

        conv2 = self.mapconv_down2(x)
        x = self.mapconv_maxpool(conv2)
        
        conv3 = self.mapconv_down3(x)
        x = self.mapconv_maxpool(conv3)   
        
        # put em all together. Should all be 1x1
        embs = [
            x, self.compress_resnet(observations),
            memory.view(-1, 512,1,1), prev_actions_embeds.view(-1, 6,1,1)
        ]
        try:
            x = self.pointwise_conv(torch.cat(embs, dim=1,))
        except:
            ForkedPdb().set_trace()
        
        # ForkedPdb().set_trace()
        x = self.mapconv_up1(x)
        x = _upsample_add(x, conv3)
        x = self.mapconv_up2(x)
        x = _upsample_add(x, conv2)
        x = self.mapconv_up3(x)
        x = _upsample_add(x, conv1)

        x = self.mapconv_up5(x)

        # ForkedPdb().set_trace()

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class PredictMapLoss(AbstractActorCriticLoss):
    def __init__(self) -> None:
        super().__init__()
        self.episode_index = 0


    def loss(  # type: ignore
            self,
            step_count: int,
            batch: ObservationType,
            actor_critic_output: ActorCriticOutput[CategoricalDistr],
            *args,
            **kwargs
    ):
        observations = cast(Dict[str, torch.Tensor], batch["observations"])
        extra_model_outputs = actor_critic_output.extras

        fix_map_size = nn.AdaptiveAvgPool2d((40,40))
        threshold_aggregate_map = nn.Threshold(0.999,0)
        sig = nn.Sigmoid()

        pred_map = extra_model_outputs['map_estimate']
        pred_map = pred_map.view(-1, *pred_map.shape[-2:]) # process predicted map to be [nsamplers*nteps, 40, 40]

        raw_gt_map = observations['aggregate_map_sensor'][:,2,:,:] # just the position, not walls (nstep, channels, rows, cols)

        
        
        # current_position_only = threshold_aggregate_map(raw_gt_map) # only the latest step
        # resized_gt_map = fix_map_size(current_position_only)

        # # set the max pixel to 1 else zero
        # # for aggregate later just set everything nonzero to 1 maybe. or leave it lesser. will be small because of the avgpool
        # # TODO this whole block isn't great, refactor
        # flat_indexes = resized_gt_map.flatten(start_dim=-2).argmax(1)
        # single_locations = [divmod(idx.item(), resized_gt_map.shape[-1]) for idx in flat_indexes]        
        # for i in range(len(single_locations)):
        #     row,col = single_locations[i]
        #     resized_gt_map[i,:,:] = threshold_aggregate_map(resized_gt_map[i,:,:]) # zeros things. used for convenience
        #     if (row == 0 and col == 0): # No prediction if the house skips. TODO why does it do that twice between eps?
        #         pred_map[i,:,:] = -100 # Don't calculate loss if there's no GT by zeroing pred. (sigmoids to zero)

        #     else:
        #         resized_gt_map[i,row,col] = 1

                
        # for aggregate:
        resized_gt_map = (fix_map_size(raw_gt_map) > 0.0).float()
        for i in range(resized_gt_map.shape[0]):
            if sum(resized_gt_map[i,:,:].flatten()) == 0.0:
                pred_map[i,:,:] = -100 # Don't calculate loss if there's no GT by zeroing pred. (sigmoids to zero)

        try:
            assert resized_gt_map.shape == pred_map.shape
        except:
            ForkedPdb().set_trace()

        loss_function = torch.nn.BCEWithLogitsLoss()
        map_loss = loss_function(pred_map,resized_gt_map)

        # additional metrics
        pred_map_class = (sig(pred_map)>0.1).float().flatten().cpu()
        agg_gt_class = resized_gt_map.flatten().cpu() #(fix_map_size(raw_gt_map) > 0.0).float()
        precision = precision_score(agg_gt_class,pred_map_class)
        recall = recall_score(agg_gt_class,pred_map_class)

        return (
            map_loss,
            {"map_pred_loss": map_loss.item(),
            "precision": precision,
            "recall": recall,
            "sum_predictions": sum((sig(pred_map)>0.1).float().flatten())}
        )
