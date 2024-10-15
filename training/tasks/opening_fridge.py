import os
import sys
import pdb
import random
import copy
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from allenact.utils.misc_utils import prepare_locals_for_super
from shapely.geometry import Polygon, Point

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import gym
import numpy as np
import prior
import cv2
from datetime import datetime

from ai2thor.controller import Controller
import ai2thor.robot_controller
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.cache_utils import DynamicDistanceCache
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from shapely.geometry import Polygon
from training import cfg
from training.utils import (
    distance_to_object_id,
    get_approx_geo_dist,
    get_room_connections,
    nearest_room_to_point,
    position_dist,
)
from training.utils.types import AgentPose, RewardConfig, TaskSamplerArgs, Vector3
from training.utils.utils import (
    randomize_chairs,
    randomize_lighting, 
    randomize_wall_and_floor_materials, 
    StochasticController, 
    ForkedPdb,
    ROBOTHOR_USABLE_POSITIONS,
    BACK_APARTMENT_USABLE_POSITIONS,
    KIANAS_USABLE_POSITIONS,
    findkeys,
)
from datetime import datetime
from training.robot.stretch_controller import StretchController
from training.robot.stretch_initialization_utils import ALL_STRETCH_ACTIONS
from training.robot.type_utils import THORActions

from matplotlib import pyplot as plt

from training.tasks.fridge_filter import get_fridge_property


from training.utils import corruptions
from training.utils.map_utils import (
    get_room_id_from_location,
    get_rooms_polymap_and_type
)
from allenact.utils.misc_utils import prepare_locals_for_super

def spl_metric(
    success: bool, optimal_distance: float, travelled_distance: float
) -> Optional[float]:
    # TODO: eventually should be -> float
    if optimal_distance < 0:
        # TODO: update when optimal_distance must be >= 0.
        # raise ValueError(
        #     f"optimal_distance must be >= 0. You gave: {optimal_distance}."
        # )
        # return None
        return 0.0
    elif not success:
        return 0.0
    elif optimal_distance == 0:
        return 1.0 if travelled_distance == 0 else 0.0
    else:
        return optimal_distance / max(travelled_distance, optimal_distance)


class OpeningfridgeTask(Task[Controller]):
    def __init__(
        self,
        controller: Controller,
        action_scale: List[float],
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_config: RewardConfig,
        distance_cache: DynamicDistanceCache,
        distance_type: Literal["geo", "l2", "approxGeo"] = "geo",
        visualize: Optional[bool] = None,
        house: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            env=controller,
            sensors=sensors,
            task_info=task_info,
            max_steps=max_steps,
            **kwargs,
        )
        self.action_scale = np.array(action_scale)
        self.stretch_controller = controller
        self.house = house
        self.reward_config = reward_config
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.mirror = task_info["mirrored"]

        self._rewards: List[float] = []
        self._manipulation_rewards: List[float] = []
        self._penalties: List[float] = []
        self._manipulation_shaping_moving_rewards: List[float] = []
        self._manipulation_shaping_keep_rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List = (
            []
        )  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0

        self.task_info["followed_path"] = [
            self.stretch_controller.get_current_agent_position()
        ]
        self.task_info["ee_followed_path"] = [
            self.stretch_controller.get_arm_sphere_center()
        ]
        self.task_info["taken_actions"] = []
        self.task_info["action_successes"] = []
        self.task_info["rewards"] = []
        self.task_info["dist_to_target"] = []
        self.task_info["dist_to_knob"] = []
        self.task_info["target_locations"] = []
        self.task_info["openness"] = []

        self.distance_cache = distance_cache

        self.distance_type = distance_type
        if distance_type == "geo":
            self.dist_to_target_func = self.min_geo_distance_to_target
        elif distance_type == "l2":
            self.dist_to_target_func = self.min_l2_distance_to_target
        elif distance_type == "approxGeo":
            self.dist_to_target_func = self.min_approx_geo_distance_to_target
            assert house is not None

            self.room_connection_graph, self.room_id_to_open_key = get_room_connections(
                house
            )

            self.room_polygons = [
                Polygon([(poly["x"], poly["z"]) for poly in room["floorPolygon"]])
                for room in house["rooms"]
            ]
        elif distance_type == "realWorld": 
            self.dist_to_target_func = self.dummy_distance_to_target
        else:
            raise NotImplementedError

        self.last_distance = self.dist_to_target_func()
        self.optimal_distance = self.last_distance
        self.closest_distance = self.last_distance
        self.task_info["dist_to_target"].append(self.last_distance)

        self.visualize = (
            visualize
            if visualize is not None
            else (self.task_info["mode"] == "do_not_viz_eval" or random.random() < 1 / 100)
        )
        if self.visualize:
            if self.distance_type != "realWorld":
                pose = {} 
                pose["fieldOfView"] = 50
                pose["position"] = {}
                pose["position"]["y"] = 3
                pose["position"]["x"] = self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"]
                pose["position"]["z"] = self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
                pose["orthographic"] = False
                pose["farClippingPlane"] = 50
                pose["rotation"] = {'x': 90.0, 'y': 0.0, 'z': 0.0}
                # add the camera to the scene
                event = controller.step(
                    action="AddThirdPartyCamera",
                    **pose,
                    skyboxColor="white",
                    raise_for_failure=True,
                )

            if self.distance_type == "realWorld":
                self.observations = [
                    np.concatenate([
                        self.stretch_controller.navigation_camera,
                        self.stretch_controller.manipulation_camera,
                    ], axis=1)
                ]
            else:

                # self.observations = [self.stretch_controller.last_event.frame]
                # TODO: hacky. Set to auto-id the camera sensor or get obs and de-normalize in a function
                self.observations = [
                    np.concatenate([
                        self.stretch_controller.navigation_camera,
                        self.stretch_controller.manipulation_camera,
                        corruptions.apply_corruption_sequence(np.array(
                                self.stretch_controller.controller.last_event.third_party_camera_frames[1][..., :3]
                            ),[], []
                        )
                    ], axis=1)
                ]
            # self.task_info["taken_actions"].append(
            #     np.zeros(self.continuous_action_dim())
            # )
        self._metrics = None

        self.fridge_id = task_info["fridge_id"]
        self.fridge_corner = task_info["fridge_corner"]
        self.fridge_width = task_info["fridge_width"]
        self.fridge_size = task_info["fridge_size"]
        self.opened_direction = task_info["opened_direction"]
        self.closed_direction = task_info["closed_direction"]
        self.reached_before = False
        self.grasped_before = False
        self.grasped = False
        self.reached_knob_before = False
        self.opened_a_bit_before = False
        self.steps_to_target = self.max_steps
        self.steps_to_knob = self.max_steps
        self.steps_to_grasp = self.max_steps
        self.init_openness = task_info["init_openness"]
        self.max_openness = self.init_openness
        self.knob_offset = (0.95, 1)

        self.task_info["init_dist_to_target"] = self.last_distance

        if self.task_info["init_dist_to_target"] < 2.5:
            dist_cat = "<2.5"
        elif self.task_info["init_dist_to_target"] < 3.5:
            dist_cat = "2.5-3.5"
        elif self.task_info["init_dist_to_target"] < 4.5:
            dist_cat = "3.5-4.5"
        else:
            dist_cat = ">4.5"
        self.task_info["init_dist_category"] = dist_cat

        # self.get_opening_type()
        # self.task_info["opening_type"] = self.opening_type

        self.last_distance_to_knob = self.dist_to_knob()
        self.optimal_distance_to_knob = self.last_distance_to_knob
        self.closest_distance_to_knob = self.last_distance_to_knob
        self.task_info["dist_to_knob"].append(self.last_distance_to_knob)
        self.stuck = False

        self.fridge_reach_distance = self.task_info["fridge_reach_distance"]
        self.knob_reach_distance = self.task_info["knob_reach_distance"]
        self.openfridge_use_section_scale = self.task_info["openfridge_use_section_scale"]
        self.push_need_grasp = self.task_info["push_need_grasp"]
        self.push_need_close_knob = self.task_info["push_need_close_knob"]
        self.pull_need_grasp = self.task_info["pull_need_grasp"]

    @property
    def stretch_action_function(self,):
        return self.stretch_controller.continuous_agent_step

    def min_l2_distance_to_target(self) -> float:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = float("inf")
        obj_id_to_obj_pos = {
            o["objectId"]: o["axisAlignedBoundingBox"]["center"]
            for o in self.stretch_controller.controller.last_event.metadata["objects"]
        }
        for object_id in self.task_info["target_object_ids"]:
            min_dist = min(
                min_dist,
                IThorEnvironment.position_dist(
                    obj_id_to_obj_pos[object_id],
                    self.stretch_controller.get_current_agent_position(),
                ),
            )
        if min_dist == float("inf"):
            get_logger().error(
                f"No target object {self.task_info['object_type']} found"
                f" in house {self.task_info['house_name']}."
            )
            return -1.0
        return min_dist

    def dummy_distance_to_target(self) -> float: # placeholder for potential real proxy later
        return float("inf")

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(5,))

    @classmethod
    def continuous_action_dim(self):
        return 5

    def close(self) -> None:
        self.stretch_controller.stop()

    def open_fridge_according_to_action_pull(self, action, pre_move_info, post_move_info):
        end_effector_correct_position = 0
        openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]
        # openness_scale = 1/(1 - np.clip(openness, 0, 0.9))
        openness_scale = 1 if not self.openfridge_use_section_scale else 1/(1 - np.clip(openness, 0, 0.9))

        self.max_openness = max(openness, self.max_openness)
        delta = 0

        pre_ee_center = pre_move_info["end_effector_pos"]

        pre_vec_to_fridge_axis = pre_ee_center - self.fridge_corner

        pre_angle = np.arccos(np.sum(pre_vec_to_fridge_axis * self.closed_direction) / (np.linalg.norm(pre_vec_to_fridge_axis) + 1e-5))
        pre_angle = 2 * pre_angle / np.pi

        # There suppose to be only one fridge and should be opennable
        current_fridge_angle = np.pi * 0.5 * openness
        openable_direction = np.cos(current_fridge_angle) * self.closed_direction + np.sin(current_fridge_angle) * self.opened_direction
        to_open_direction = np.cos(current_fridge_angle) * self.opened_direction - np.sin(current_fridge_angle) * self.closed_direction

        end_effector_moving_direction = post_move_info["end_effector_pos"] - pre_move_info["end_effector_pos"]

        # to open has norm = 1
        self.project_vec_norm =  np.sum(end_effector_moving_direction * to_open_direction) /\
            (np.linalg.norm(end_effector_moving_direction) + 1e-5)

        if np.linalg.norm(end_effector_moving_direction) < 1e-3:
            return 0, openness_scale, end_effector_correct_position
        
        if np.sum(pre_vec_to_fridge_axis * self.opened_direction) < 0 or \
            np.sum(pre_vec_to_fridge_axis * self.closed_direction) < 0:
            return 0, openness_scale, end_effector_correct_position
    
        if pre_angle < openness:
            return 0, openness_scale, end_effector_correct_position
        
        if not self._reach_fridge:
            return 0, openness_scale, end_effector_correct_position

        # dist_to_knob = self.dist_to_knob()     
        if self.current_dist_to_knob > self.knob_reach_distance:
            return 0, openness_scale, end_effector_correct_position

        if np.linalg.norm(pre_vec_to_fridge_axis) > self.fridge_width or np.linalg.norm(pre_vec_to_fridge_axis) < 0.5 * self.fridge_width:
            return 0, openness_scale, end_effector_correct_position


        pre_dis_along = np.sum(pre_vec_to_fridge_axis * openable_direction) / np.linalg.norm(pre_vec_to_fridge_axis) * openable_direction
        pre_dis_per = pre_vec_to_fridge_axis - pre_dis_along
        pre_dis_to_fridge = np.linalg.norm(pre_dis_per) - self.fridge_size
        # print("fridge Center:", self.fridge_corner)
        # print("Pre Dis to fridge:", pre_dis_to_fridge)
        if pre_dis_to_fridge > 0.4:
            return 0, openness_scale, end_effector_correct_position


        post_ee_center = post_move_info["end_effector_pos"]
        post_vec_to_fridge_axis = post_ee_center - self.fridge_corner

        post_angle = np.arccos(np.sum(post_vec_to_fridge_axis * self.closed_direction) / (np.linalg.norm(post_vec_to_fridge_axis) + 1e-5))
        post_angle = 2 * post_angle / np.pi

        fake_angle = post_angle - 0.05
        post_openable_direction = np.cos(fake_angle) * self.closed_direction + np.sin(fake_angle) * self.opened_direction
        post_dis_along = np.sum(post_vec_to_fridge_axis * post_openable_direction) / np.linalg.norm(post_vec_to_fridge_axis) * post_openable_direction
        post_dis_per = post_vec_to_fridge_axis - post_dis_along
        post_dis_to_fridge = np.linalg.norm(post_dis_per) - self.fridge_size

        # print("Post Dis to fridge:", post_dis_to_fridge)
        if post_dis_to_fridge < 0.075:
            return 0, openness_scale, end_effector_correct_position

        post_agent_center = post_move_info["agent_pos"]
        post_agent_to_fridge_axis = post_agent_center - self.fridge_corner
        post_agent_along = np.sum(post_agent_to_fridge_axis * post_openable_direction) / np.linalg.norm(post_agent_to_fridge_axis) * post_openable_direction
        post_agent_per =  post_agent_to_fridge_axis - post_agent_along
        post_agent_dis_to_fridge = np.linalg.norm(post_agent_per) - self.fridge_size - 0.25

        # print("Post Agent Dis to fridge:", post_agent_dis_to_fridge)
        if (post_agent_dis_to_fridge < post_dis_to_fridge or post_agent_dis_to_fridge < 0.05) and (
            np.linalg.norm(post_agent_to_fridge_axis) < self.fridge_size + 0.05
        ):
            return 0, openness_scale, end_effector_correct_position

        post_top_center = np.squeeze(np.array([
            self.stretch_controller.controller.last_event.metadata["arm"]["joints"][-2]["position"]["x"],
            self.stretch_controller.controller.last_event.metadata["arm"]["joints"][-2]["position"]["z"]
        ]))
        post_top_to_fridge_axis = post_top_center - self.fridge_corner
        post_top_along = np.sum(post_top_to_fridge_axis * post_openable_direction) / np.linalg.norm(post_top_to_fridge_axis) * post_openable_direction
        post_top_per = post_top_center - post_top_along
        post_top_dis_to_fridge = np.linalg.norm(post_top_per)  - self.fridge_size

        # print("Post Top Dis to fridge:", post_top_dis_to_fridge)
        if post_top_dis_to_fridge < post_dis_to_fridge or post_top_dis_to_fridge < 0.075:
            return 0, openness_scale, end_effector_correct_position

        end_effector_correct_position = 1

        if self.project_vec_norm > 0.6:
            target_angle = min(post_angle - 0.075, openness + 0.05)
            self.stretch_controller.controller.step(
                action='OpenObject',
                objectId=self.fridge_id,
                openness=post_angle - 0.075,
                forceAction=True
            )
            delta = target_angle - openness

            openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]
            self.max_openness = max(openness, self.max_openness)
            if openness > 0.7:
                self._success = True
                self._took_end_action = True
            return delta, openness_scale, end_effector_correct_position
        else:
            return 0, openness_scale, end_effector_correct_position
        # NOTE: Here determine the relative 


    def open_fridge_according_to_action_push(self, action, mode=None):

        end_effector_correct_position = 0

        openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]

        # openness_scale = 1/(1 - np.clip(openness, 0, 0.9))
        openness_scale = 1 if not self.openfridge_use_section_scale else 1/(1 - np.clip(openness, 0, 0.9))

        self.max_openness = max(openness, self.max_openness)
        delta = 0
        end_effector_center = np.squeeze(np.array([
            self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
            self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
        ]))

        self.vec_to_fridge_axis = end_effector_center - self.fridge_corner

        angle = np.arccos(np.sum(self.vec_to_fridge_axis * self.closed_direction) / np.linalg.norm(self.vec_to_fridge_axis))
        angle = 2 * angle / np.pi

        if np.sum(self.vec_to_fridge_axis * self.opened_direction) < 0:
            angle = -angle
        
        # There suppose to be only one fridge and should be opennable
        current_angle = np.pi * 0.5 * openness
        openable_direction = np.cos(current_angle) * self.closed_direction + np.sin(current_angle) * self.opened_direction
        to_open_direction = -np.cos(current_angle) * self.closed_direction + np.sin(current_angle) * self.opened_direction
        agent_rotation = self.stretch_controller.controller.last_event.metadata[
            "agent"]["rotation"]["y"] / 360 * 2 * np.pi
        agent_moving_vec = np.array([
            np.sin(agent_rotation),
            np.cos(agent_rotation)
        ]) * action[0]

        end_effector_moving_direction =  np.array([
            np.cos(agent_rotation),
            -np.sin(agent_rotation)
        ]) * action[3] + agent_moving_vec
        # get_logger().warning(f"scores: {(np.sum(end_effector_moving_direction * openable_direction))}")
        DELTA_OPENNESS=0.05
        self.project_vec_norm =  np.sum(end_effector_moving_direction * to_open_direction) /\
            (np.linalg.norm(end_effector_moving_direction) + 1e-5)

        dis_along = np.sum(self.vec_to_fridge_axis * openable_direction) / np.linalg.norm(self.vec_to_fridge_axis) * openable_direction
        dis_per = self.vec_to_fridge_axis - dis_along
        dis_to_fridge = np.linalg.norm(dis_per) - self.fridge_size

        if np.sum(self.vec_to_fridge_axis * openable_direction) < 0:
            return 0, openness_scale, end_effector_correct_position

        agent_pos = np.squeeze(np.array([
            self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"],
            self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
        ]))
        agent_to_fridge_axis = agent_pos - self.fridge_corner
        agent_along = np.sum(agent_to_fridge_axis * openable_direction) / np.linalg.norm(agent_to_fridge_axis) * openable_direction
        agent_per = agent_to_fridge_axis - agent_along
        if np.sum(agent_per * dis_per) < 0:
            # Agent & ee different side
            return 0, openness_scale, end_effector_correct_position

        # print("EE dis to fridge:", dis_to_fridge)

        if angle > openness:
            return 0, openness_scale, end_effector_correct_position

        if not self._reach_fridge:
            return 0, openness_scale, end_effector_correct_position
                
        if np.linalg.norm(self.vec_to_fridge_axis) > self.fridge_width or np.linalg.norm(self.vec_to_fridge_axis) < 0.5 * self.fridge_width:
            return 0, openness_scale, end_effector_correct_position

        if dis_to_fridge > 0.4:
            return 0, openness_scale, end_effector_correct_position

        # dis_to_knob = self.dist_to_knob()
        # if dis_to_knob > 0.2:
        #     return 0, openness_scale, end_effector_correct_position

        end_effector_correct_position = 1
        
        if np.sum(end_effector_moving_direction * self.opened_direction) < 0:
            return 0, openness_scale, end_effector_correct_position

        if self.project_vec_norm > 0.6:
            self.stretch_controller.controller.step(
                action='OpenObject',
                objectId=self.fridge_id,
                openness=openness + DELTA_OPENNESS,
                forceAction=True
            )
            delta = DELTA_OPENNESS

            openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]
            self.max_openness = max(openness, self.max_openness)
            if openness > 0.90:
                self._success = True
                self._took_end_action = True
            return delta, openness_scale, end_effector_correct_position
        else:
            return 0, openness_scale, end_effector_correct_position
        # NOTE: Here determine the relative

    def dist_to_fridge(self):
        obj_id_to_obj_pos = {
            o["objectId"]: o["axisAlignedBoundingBox"]["center"]
            for o in self.stretch_controller.controller.last_event.metadata["objects"]
        }
        for object_id in self.task_info["target_object_ids"]:
            min_dist = min(
                min_dist,
                IThorEnvironment.position_dist(
                    obj_id_to_obj_pos[object_id],
                    self.stretch_controller.get_current_agent_position(),
                ),
            )
        return min_dist


    def _step(self, action: np.array) -> RLStepResult:
        action  = action * self.action_scale
        action = np.clip(action, - 1, 1)
        self.action_for_reward = action
        self.task_info["taken_actions"].append(action)

        self._reach_fridge = self.dist_to_target_func() <= self.fridge_reach_distance and \
            self.dist_to_target_func() > 0 and \
            self.stretch_controller.objects_is_visible_in_camera(
                self.task_info["target_object_ids"], which_camera="both"
            )
        
        self.current_dist_to_knob = self.dist_to_knob()
        self._reach_knob = self.current_dist_to_knob <= self.knob_reach_distance and self._reach_fridge

        # Push 
        self.status_opening_push, self.push_openness_scale, ee_inplace_push = self.open_fridge_according_to_action_push(action)

        pre_move_info = {
            "agent_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"],
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
            ])),
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }
        event, arm_success, movement_success = self.stretch_action_function(
            action, return_to_start=False
        )
        # print(event.metadata)

        post_move_info =  {
            "agent_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"],
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
            ])),
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }

        # Pull
        self.status_opening_pull, self.pull_openness_scale, ee_inplace_pull = self.open_fridge_according_to_action_pull(
            action, pre_move_info, post_move_info
        )

        self.ee_inplace = ee_inplace_pull + ee_inplace_push

        # NOTE: Open the fridge according to the fridge.
        self.last_action_success = bool(
            arm_success and movement_success
        )
        self.last_movement_success = movement_success

        position = self.stretch_controller.get_current_agent_position()
        self.path.append(position)
        self.task_info["followed_path"].append(position)
        self.task_info["ee_followed_path"].append(self.stretch_controller.get_arm_sphere_center())
        self.task_info["action_successes"].append(self.last_action_success)
        self.task_info["openness"].append(self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"])

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            # self.observations.append(self.stretch_controller.last_event.frame)
            pose = {}
            pose["fieldOfView"] = 50
            pose["position"] = {}
            pose["position"]["y"] = 3
            pose["position"]["x"] = self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"]
            pose["position"]["z"] = self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
            pose["orthographic"] = False
            pose["farClippingPlane"] = 50
            pose["rotation"] = {'x': 90.0, 'y': 0.0, 'z': 0.0}
            # TODO: same as above
            event = self.stretch_controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=1,
                **pose,
            )
            self.observations.append(
                np.concatenate([
                    self.stretch_controller.navigation_camera,
                    self.stretch_controller.manipulation_camera,
                    corruptions.apply_corruption_sequence(np.array(
                            self.stretch_controller.controller.last_event.third_party_camera_frames[1][..., :3]
                        ),[], []
                    )
                ], axis=1)
            )

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={
                "last_action_success": self.last_action_success, "action": action,
                "navigation_action_success": movement_success,
                "arm_action_success": arm_success,
            },
        )
        return step_result

    def render(
        self, mode: Literal["rgb", "depth"] = "rgb", *args, **kwargs
    ) -> np.ndarray:
        if mode == "rgb":
            frame = self.stretch_controller.last_event.frame.copy()
        elif mode == "depth":
            frame = self.stretch_controller.last_event.depth_frame.copy()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

        if self.mirror:
            frame = np.fliplr(frame)

        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            obj
            for obj in self.stretch_controller.last_event.metadata["objects"]
            if obj["visible"] and obj["objectType"] == self.task_info["object_type"]
        )

    def shaping(self) -> float:
        cur_distance = self.dist_to_target_func()
        self.task_info["dist_to_target"].append(cur_distance)
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        # cur_distance = self.dist_to_target_func()

        if self.distance_type == "l2":
            reward = max(self.closest_distance - cur_distance, 0)
            self.closest_distance = min(self.closest_distance, cur_distance)

            return reward * self.reward_config.shaping_weight
        else:
            # Ensuring the reward magnitude is not greater than the total distance moved
            max_reward_mag = 0.0
            if len(self.path) >= 2:
                p0, p1 = self.path[-2:]
                max_reward_mag = position_dist(p0, p1, ignore_y=True)

            if (
                self.reward_config.positive_only_reward
                and cur_distance is not None
                and cur_distance > 0.5
            ):
                reward = max(self.closest_distance - cur_distance, 0)
            elif self.last_distance is not None and cur_distance is not None:
                reward += self.last_distance - cur_distance

            self.last_distance = cur_distance
            self.closest_distance = min(self.closest_distance, cur_distance)

            return (
                max(
                    min(reward, max_reward_mag),
                    -max_reward_mag,
                )
                * self.reward_config.shaping_weight
            )

    def get_knob_pos(self):
        openness = self.stretch_controller.controller.last_event.get_object(self.fridge_id)["openness"]
        current_angle = np.pi * 0.5 * openness
        knob_pos_numpy = self.knob_offset[0] * self.fridge_width * (
            np.cos(current_angle) * self.closed_direction + np.sin(current_angle) * self.opened_direction
        )
        knob_pos = {
            "x": knob_pos_numpy[0] + self.fridge_corner[0],
            "z": knob_pos_numpy[1] + self.fridge_corner[1],
            "y": self.knob_offset[1]
        }
        
        
        # Currently Ignore the offset along vertical
        return knob_pos

    def dist_to_knob(self):
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        if self.distance_type == "realWorld":
            return -1
        knob_positions = self.get_knob_pos()
        ee_position = self.stretch_controller.get_arm_sphere_center()
        dist = position_dist(
            ee_position, knob_positions,
            ignore_y=True
        )
        return dist


    def manipulation_shaping(self) -> float:
        # cur_distance = self.dist_to_knob()
        cur_distance = self.current_dist_to_knob
        self.task_info["dist_to_knob"].append(cur_distance)
        if self.reward_config.manipulation_shaping_weight == 0.0:
            return 0, 0

        distance_moved = max(self.closest_distance_to_knob - cur_distance, 0) * \
            self.reward_config.manipulation_shaping_moving_scale
        self.closest_distance_to_knob = min(self.closest_distance_to_knob, cur_distance)
        # Encourage moving to the knob and stay there
        scale = np.exp(self.reward_config.manipulation_shaping_scale * cur_distance)
        moving_reward = distance_moved * scale * self.reward_config.manipulation_shaping_weight

        keeping_reward = scale * self.reward_config.manipulation_shaping_weight * (1 - self.project_vec_norm) * 0
        return moving_reward, keeping_reward


    def judge(self) -> float:
        """Judge the last event."""

        penalties = self.reward_config.energy_penalty * self.ee_moved_distance


        too_close_to_fridge = self.dist_to_target_func() <= 0.5
        penalties += too_close_to_fridge * self.reward_config.too_close_penalty

        penalties += self.reward_config.step_penalty
        if not self.last_movement_success: #and "Look" not in self.task_info["taken_actions"][-1]:
            penalties += self.reward_config.failed_action_penalty

        reward = penalties

        reward += (self.shaping() * (1 - self.reached_before))

        if self._reach_fridge:
            if not self.reached_before:
                reward += self.reward_config.goal_success_reward
                self.reached_before = True
                self.steps_to_target = self.num_steps_taken() + 1
        if self._reach_knob:
            if not self.reached_knob_before:
                reward += self.reward_config.knob_success_reward * self.reward_config.manipulation_shaping_weight
                self.reached_knob_before = True
                self.steps_to_knob = self.num_steps_taken() + 1

        if self._success:
            reward += self.reward_config.complete_task_reward

        manip_reward = (
            self.status_opening_pull * self.pull_openness_scale + self.status_opening_push * self.push_openness_scale
        ) * self.reward_config.open_section_reward

        if self.max_openness > self.init_openness and (not self.opened_a_bit_before):
            get_logger().warning(f"Max Openness: {self.max_openness}, Pull: {self.status_opening_pull}, Push: {self.status_opening_push}, Init: {manip_reward}")
            self.opened_a_bit_before = True
            manip_reward += self.reward_config.open_initial_reward
            get_logger().warning(f"After: {manip_reward}")

        manip_shaping_moving_reward, manip_shaping_keep_reward = self.manipulation_shaping()
        manip_reward += ((manip_shaping_moving_reward + manip_shaping_keep_reward) *  (1 - self.reached_knob_before))
        reward += manip_reward

        if self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward
            event_forward = self.stretch_controller.controller.step(action="MoveAgent", ahead=0.01)
            event_backward = self.stretch_controller.controller.step(action="MoveAgent", ahead=-0.01)
            event_left = self.stretch_controller.controller.step(action="RotateAgent", degrees=0.5)
            event_right = self.stretch_controller.controller.step(action="RotateAgent", degrees=-0.5)
            self.not_stuck = event_forward.metadata["lastActionSuccess"] or event_backward.metadata["lastActionSuccess"] or event_left.metadata["lastActionSuccess"] or event_right.metadata["lastActionSuccess"]
            self.stuck = not self.not_stuck

        self._rewards.append(float(reward))
        self._manipulation_rewards.append(float(manip_reward))
        self._penalties.append(float(penalties))
        self._manipulation_shaping_moving_rewards.append(float(manip_shaping_moving_reward))
        self._manipulation_shaping_keep_rewards.append(float(manip_shaping_keep_reward))
        
        self.task_info["rewards"].append(float(reward))
        return float(reward)

    def get_observations(self, **kwargs) -> Any:
        obs = super().get_observations()
        if self.mirror:
            for o in obs:
                if ("rgb" in o or "depth" in o) and isinstance(obs[o], np.ndarray):
                    obs[o] = np.fliplr(obs[o])
        return obs

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        metrics["init_dist_to_target"] = self.task_info["init_dist_to_target"]
        metrics["dist_to_target"] = self.dist_to_target_func()
        metrics["dist_to_knob"] = self.current_dist_to_knob
        metrics["knob_reach_distance"] = self.knob_reach_distance
        metrics["total_reward"] = np.sum(self._rewards)
        metrics["manipulation_reward"] = np.sum(self._manipulation_rewards)
        metrics["penalties"] = np.sum(self._penalties)
        metrics["manipulation_shaping_moving_reward"] = np.sum(self._manipulation_shaping_moving_rewards)
        metrics["manipulation_shaping_keep_reward"] = np.sum(self._manipulation_shaping_keep_rewards)
        metrics["navigation_reward"] = metrics["total_reward"] - metrics["manipulation_reward"]
        metrics["spl"] = spl_metric(
            success=self._success,
            optimal_distance=self.optimal_distance,
            travelled_distance=self.travelled_distance,
        )
        metrics["success"] = self._success
        metrics["progress_per_step"] = np.clip(self.max_openness / 0.7, 0, 1) / self.num_steps_taken()
        metrics["normalized_progress"] = np.clip(self.max_openness / 0.7, 0, 1)
        metrics["reach_target"] = self.reached_before
        metrics["grasped"] = self.grasped_before
        metrics["max_openness"] = self.max_openness
        metrics["stuck"] = self.stuck
        metrics["step_to_target"] = self.steps_to_target
        metrics["step_to_knob"] = self.steps_to_knob
        metrics["step_to_grasp"] = self.steps_to_grasp
        metrics["reach_knob"] = self.reached_knob_before
        self._metrics = metrics
        return metrics


class OpeningfridgeIterativeTask(OpeningfridgeTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(6,))

    @classmethod
    def continuous_action_dim(self):
        return 6

    def _step(self, action: np.array) -> RLStepResult:
        if action[-1] > 0:
            action[2] = 0
            action[3] = 0
            action[4] = 0
        else:
            action[0] = 0
            action[1] = 0
        return super()._step(action[:-1])


class OpeningfridgeGraspTask(OpeningfridgeTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(6,))

    @classmethod
    def continuous_action_dim(self):
        return 6

    def _step(self, action: np.array) -> RLStepResult:
        action = np.array(action)
        self.intend_grasp = action[-1] > 0
        if action[-1] > 0:
            action[:-1] = 0
            self.intend_grasp = True

        action = action[:-1]
    
        action  = action * self.action_scale[0]
        action = np.clip(action, -1, 1)
        self.action_for_reward = action
        self.task_info["taken_actions"].append(action)

        self._reach_fridge = self.dist_to_target_func() <= self.fridge_reach_distance and \
            self.dist_to_target_func() > 0 and \
            self.stretch_controller.objects_is_visible_in_camera(
                self.task_info["target_object_ids"], which_camera="both"
            )
        
        self.current_dist_to_knob = self.dist_to_knob()
        self._reach_knob = self.current_dist_to_knob <= self.knob_reach_distance and self._reach_fridge
        if self._reach_knob and self.intend_grasp:
            self.grasped = True
        if not self._reach_knob:
            self.grasped = False
        
        self.failed_grasp = False
        if (not self.grasped) and self.intend_grasp:
            self.failed_grasp = True

        # Push 
        self.status_opening_push, self.push_openness_scale, ee_inplace_push = self.open_fridge_according_to_action_push(action)

        pre_move_info = {
            "agent_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"],
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
            ])),
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ])),
            "full_end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["y"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }
        event, arm_success, movement_success = self.stretch_action_function(
            action, return_to_start=False
        )
        # print(event.metadata)

        post_move_info =  {
            "agent_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"],
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
            ])),
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ])),
            "full_end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["y"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }
        # print(pre_move_info)
        # print(post_move_info)
        self.ee_moved_distance = np.linalg.norm(
            post_move_info["full_end_effector_pos"] - pre_move_info["full_end_effector_pos"]
        )

        # Pull
        self.status_opening_pull, self.pull_openness_scale, ee_inplace_pull = self.open_fridge_according_to_action_pull(
            action, pre_move_info, post_move_info
        )

        self.ee_inplace = ee_inplace_pull + ee_inplace_push

        # NOTE: Open the fridge according to the fridge.
        self.last_action_success = bool(
            arm_success and movement_success
        )
        self.last_movement_success = movement_success

        position = self.stretch_controller.get_current_agent_position()
        self.path.append(position)
        self.task_info["followed_path"].append(position)
        self.task_info["ee_followed_path"].append(self.stretch_controller.get_arm_sphere_center())
        self.task_info["action_successes"].append(self.last_action_success)
        self.task_info["openness"].append(self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"])

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            # self.observations.append(self.stretch_controller.last_event.frame)
            pose = {}
            pose["fieldOfView"] = 50
            pose["position"] = {}
            pose["position"]["y"] = 3
            pose["position"]["x"] = self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"]
            pose["position"]["z"] = self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
            pose["orthographic"] = False
            pose["farClippingPlane"] = 50
            pose["rotation"] = {'x': 90.0, 'y': 0.0, 'z': 0.0}
            # TODO: same as above
            event = self.stretch_controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=1,
                **pose,
            )
            self.observations.append(
                np.concatenate([
                    self.stretch_controller.navigation_camera,
                    self.stretch_controller.manipulation_camera,
                    corruptions.apply_corruption_sequence(np.array(
                            self.stretch_controller.controller.last_event.third_party_camera_frames[1][..., :3]
                        ),[], []
                    )
                ], axis=1)
            )

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={
                "last_action_success": self.last_action_success, "action": action,
                "navigation_action_success": movement_success,
                "arm_action_success": arm_success,
            },
        )
        return step_result

    def judge(self) -> float:
        """Judge the last event."""
        penalties = self.reward_config.energy_penalty * self.ee_moved_distance

        too_close_to_fridge = self.dist_to_target_func() <= 0.5
        penalties += too_close_to_fridge * self.reward_config.too_close_penalty


        penalties += self.reward_config.step_penalty
        if not self.last_movement_success: #and "Look" not in self.task_info["taken_actions"][-1]:
            penalties += self.reward_config.failed_action_penalty

        if self.failed_grasp:
            penalties += self.reward_config.failed_action_penalty

        reward = penalties

        reward += (self.shaping() * (1 - self.reached_before))

        if self._reach_fridge:
            if not self.reached_before:
                reward += self.reward_config.goal_success_reward
                self.reached_before = True
                self.steps_to_target = self.num_steps_taken() + 1
        if self._reach_knob:
            if not self.reached_knob_before:
                reward += self.reward_config.knob_success_reward * self.reward_config.manipulation_shaping_weight
                self.reached_knob_before = True
                self.steps_to_knob = self.num_steps_taken() + 1
        if self.grasped:
            if not self.grasped_before:
                reward += self.reward_config.grasp_success_reward
                self.grasped_before = True
                self.steps_to_grasp = self.num_steps_taken() + 1

        if self._success:
            reward += self.reward_config.complete_task_reward

        manip_reward = (
            self.status_opening_pull * self.pull_openness_scale + self.status_opening_push * self.push_openness_scale
        ) * self.reward_config.open_section_reward

        if self.max_openness > self.init_openness and (not self.opened_a_bit_before):
            get_logger().warning(f"Max Openness: {self.max_openness}, Pull: {self.status_opening_pull}, Push: {self.status_opening_push}, Init: {manip_reward}")
            self.opened_a_bit_before = True
            manip_reward += self.reward_config.open_initial_reward
            get_logger().warning(f"After: {manip_reward}")

        manip_shaping_moving_reward, manip_shaping_keep_reward = self.manipulation_shaping()
        manip_reward += ((manip_shaping_moving_reward + manip_shaping_keep_reward) *  (1 - self.reached_knob_before))
        reward += manip_reward
        # get_logger().warning(reward)
        if self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward
            event_forward = self.stretch_controller.controller.step(action="MoveAgent", ahead=0.01)
            event_backward = self.stretch_controller.controller.step(action="MoveAgent", ahead=-0.01)
            event_left = self.stretch_controller.controller.step(action="RotateAgent", degrees=0.5)
            event_right = self.stretch_controller.controller.step(action="RotateAgent", degrees=-0.5)
            self.not_stuck = event_forward.metadata["lastActionSuccess"] or event_backward.metadata["lastActionSuccess"] or event_left.metadata["lastActionSuccess"] or event_right.metadata["lastActionSuccess"]
            self.stuck = not self.not_stuck

        self._rewards.append(float(reward))
        self._manipulation_rewards.append(float(manip_reward))
        self._penalties.append(float(penalties))
        self._manipulation_shaping_moving_rewards.append(float(manip_shaping_moving_reward))
        self._manipulation_shaping_keep_rewards.append(float(manip_shaping_keep_reward))
        
        self.task_info["rewards"].append(float(reward))
        return float(reward)

    def open_fridge_according_to_action_pull(self, action, pre_move_info, post_move_info):
        end_effector_correct_position = 0
        openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]
        # openness_scale = 1/(1 - np.clip(openness, 0, 0.9))
        openness_scale = 1 if not self.openfridge_use_section_scale else 1/(1 - np.clip(openness, 0, 0.9))

        self.max_openness = max(openness, self.max_openness)
        delta = 0

        pre_ee_center = pre_move_info["end_effector_pos"]

        pre_vec_to_fridge_axis = pre_ee_center - self.fridge_corner

        pre_angle = np.arccos(np.sum(pre_vec_to_fridge_axis * self.closed_direction) / np.linalg.norm(pre_vec_to_fridge_axis))
        pre_angle = 2 * pre_angle / np.pi

        # There suppose to be only one fridge and should be opennable
        current_fridge_angle = np.pi * 0.5 * openness
        openable_direction = np.cos(current_fridge_angle) * self.closed_direction + np.sin(current_fridge_angle) * self.opened_direction
        to_open_direction = np.cos(current_fridge_angle) * self.opened_direction - np.sin(current_fridge_angle) * self.closed_direction

        end_effector_moving_direction = post_move_info["end_effector_pos"] - pre_move_info["end_effector_pos"]

        self.project_vec_norm =  np.sum(end_effector_moving_direction * to_open_direction) /\
            (np.linalg.norm(end_effector_moving_direction) + 1e-5)

        if np.linalg.norm(end_effector_moving_direction) < 1e-3:
            return 0, openness_scale, end_effector_correct_position
        
        if np.sum(pre_vec_to_fridge_axis * self.opened_direction) < 0 or \
            np.sum(pre_vec_to_fridge_axis * self.closed_direction) < 0:
            return 0, openness_scale, end_effector_correct_position
    
        if pre_angle < openness:
            return 0, openness_scale, end_effector_correct_position
        
        if not self._reach_fridge:
            return 0, openness_scale, end_effector_correct_position


        if not self._reach_knob:
            return 0, openness_scale, end_effector_correct_position

        if self.push_need_grasp:
            if not self.grasped:
                return 0, openness_scale, end_effector_correct_position


        # dist_to_knob = self.dist_to_knob()     
        if self.current_dist_to_knob > self.knob_reach_distance:
            return 0, openness_scale, end_effector_correct_position

        if np.linalg.norm(pre_vec_to_fridge_axis) > self.fridge_width or np.linalg.norm(pre_vec_to_fridge_axis) < 0.5 * self.fridge_width:
            return 0, openness_scale, end_effector_correct_position


        pre_dis_along = np.sum(pre_vec_to_fridge_axis * openable_direction) / np.linalg.norm(pre_vec_to_fridge_axis) * openable_direction
        pre_dis_per = pre_vec_to_fridge_axis - pre_dis_along
        pre_dis_to_fridge = np.linalg.norm(pre_dis_per) - self.fridge_size
        # print("fridge Center:", self.fridge_corner)
        # print("Pre Dis to fridge:", pre_dis_to_fridge)
        if pre_dis_to_fridge > 0.4:
            return 0, openness_scale, end_effector_correct_position


        post_ee_center = post_move_info["end_effector_pos"]
        post_vec_to_fridge_axis = post_ee_center - self.fridge_corner

        post_angle = np.arccos(np.sum(post_vec_to_fridge_axis * self.closed_direction) / np.linalg.norm(post_vec_to_fridge_axis))
        post_angle = 2 * post_angle / np.pi

        fake_angle = post_angle - 0.05
        post_openable_direction = np.cos(fake_angle) * self.closed_direction + np.sin(fake_angle) * self.opened_direction
        post_dis_along = np.sum(post_vec_to_fridge_axis * post_openable_direction) / np.linalg.norm(post_vec_to_fridge_axis) * post_openable_direction
        post_dis_per = post_vec_to_fridge_axis - post_dis_along
        post_dis_to_fridge = np.linalg.norm(post_dis_per) - self.fridge_size

        # print("Post Dis to fridge:", post_dis_to_fridge)
        if post_dis_to_fridge < 0.075:
            return 0, openness_scale, end_effector_correct_position

        post_agent_center = post_move_info["agent_pos"]
        post_agent_to_fridge_axis = post_agent_center - self.fridge_corner
        post_agent_along = np.sum(post_agent_to_fridge_axis * post_openable_direction) / np.linalg.norm(post_agent_to_fridge_axis) * post_openable_direction
        post_agent_per =  post_agent_to_fridge_axis - post_agent_along
        post_agent_dis_to_fridge = np.linalg.norm(post_agent_per) - self.fridge_size - 0.25

        # print("Post Agent Dis to fridge:", post_agent_dis_to_fridge)
        if post_agent_dis_to_fridge < post_dis_to_fridge or post_agent_dis_to_fridge < 0.05:
            return 0, openness_scale, end_effector_correct_position

        post_top_center = np.squeeze(np.array([
            self.stretch_controller.controller.last_event.metadata["arm"]["joints"][-2]["position"]["x"],
            self.stretch_controller.controller.last_event.metadata["arm"]["joints"][-2]["position"]["z"]
        ]))
        post_top_to_fridge_axis = post_top_center - self.fridge_corner
        post_top_along = np.sum(post_top_to_fridge_axis * post_openable_direction) / np.linalg.norm(post_top_to_fridge_axis) * post_openable_direction
        post_top_per = post_top_center - post_top_along
        post_top_dis_to_fridge = np.linalg.norm(post_top_per)  - self.fridge_size

        # print("Post Top Dis to fridge:", post_top_dis_to_fridge)
        if post_top_dis_to_fridge < post_dis_to_fridge or post_top_dis_to_fridge < 0.075:
            return 0, openness_scale, end_effector_correct_position

        end_effector_correct_position = 1

        if self.project_vec_norm > 0.6:
            target_angle = min(post_angle - 0.075, openness + 0.05)
            self.stretch_controller.controller.step(
                action='OpenObject',
                objectId=self.fridge_id,
                openness=post_angle - 0.075,
                forceAction=True
            )
            delta = target_angle - openness

            openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]
            self.max_openness = max(openness, self.max_openness)
            if openness > 0.7:
                self._success = True
                self._took_end_action = True
            return delta, openness_scale, end_effector_correct_position
        else:
            return 0, openness_scale, end_effector_correct_position
        # NOTE: Here determine the relative 


    def open_fridge_according_to_action_push(self, action, mode=None):

        end_effector_correct_position = 0

        openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]

        # openness_scale = 1/(1 - np.clip(openness, 0, 0.9))
        openness_scale = 1 if not self.openfridge_use_section_scale else 1/(1 - np.clip(openness, 0, 0.9))

        self.max_openness = max(openness, self.max_openness)
        delta = 0
        end_effector_center = np.squeeze(np.array([
            self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
            self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
        ]))

        self.vec_to_fridge_axis = end_effector_center - self.fridge_corner

        angle = np.arccos(np.sum(self.vec_to_fridge_axis * self.closed_direction) / np.linalg.norm(self.vec_to_fridge_axis))
        angle = 2 * angle / np.pi

        if np.sum(self.vec_to_fridge_axis * self.opened_direction) < 0:
            angle = -angle
        
        # There suppose to be only one fridge and should be opennable
        current_angle = np.pi * 0.5 * openness
        openable_direction = np.cos(current_angle) * self.closed_direction + np.sin(current_angle) * self.opened_direction
        to_open_direction = -np.cos(current_angle) * self.closed_direction + np.sin(current_angle) * self.opened_direction
        agent_rotation = self.stretch_controller.controller.last_event.metadata[
            "agent"]["rotation"]["y"] / 360 * 2 * np.pi
        agent_moving_vec = np.array([
            np.sin(agent_rotation),
            np.cos(agent_rotation)
        ]) * action[0]

        end_effector_moving_direction =  np.array([
            np.cos(agent_rotation),
            -np.sin(agent_rotation)
        ]) * action[3] + agent_moving_vec
        # get_logger().warning(f"scores: {(np.sum(end_effector_moving_direction * openable_direction))}")
        DELTA_OPENNESS=0.05
        self.project_vec_norm =  np.sum(end_effector_moving_direction * to_open_direction) /\
            (np.linalg.norm(end_effector_moving_direction) + 1e-5)

        dis_along = np.sum(self.vec_to_fridge_axis * openable_direction) / np.linalg.norm(self.vec_to_fridge_axis) * openable_direction
        dis_per = self.vec_to_fridge_axis - dis_along
        dis_to_fridge = np.linalg.norm(dis_per) - self.fridge_size

        if np.sum(self.vec_to_fridge_axis * openable_direction) < 0:
            return 0, openness_scale, end_effector_correct_position

        agent_pos = np.squeeze(np.array([
            self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"],
            self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
        ]))
        agent_to_fridge_axis = agent_pos - self.fridge_corner
        agent_along = np.sum(agent_to_fridge_axis * openable_direction) / np.linalg.norm(agent_to_fridge_axis) * openable_direction
        agent_per = agent_to_fridge_axis - agent_along
        if np.sum(agent_per * dis_per) < 0:
            # Agent & ee different side
            return 0, openness_scale, end_effector_correct_position

        if angle > openness:
            return 0, openness_scale, end_effector_correct_position

        if not self._reach_fridge:
            return 0, openness_scale, end_effector_correct_position

        if self.push_need_close_knob:
            if not self._reach_knob:
                return 0, openness_scale, end_effector_correct_position
        
        if self.push_need_grasp:
            if not self.grasped:
                return 0, openness_scale, end_effector_correct_position
        
        if np.linalg.norm(self.vec_to_fridge_axis) > self.fridge_width or np.linalg.norm(self.vec_to_fridge_axis) < 0.5 * self.fridge_width:
            return 0, openness_scale, end_effector_correct_position

        if dis_to_fridge > 0.4:
            return 0, openness_scale, end_effector_correct_position

        end_effector_correct_position = 1
        
        if np.sum(end_effector_moving_direction * self.opened_direction) < 0:
            return 0, openness_scale, end_effector_correct_position

        if self.project_vec_norm > 0.6:
            self.stretch_controller.controller.step(
                action='OpenObject',
                objectId=self.fridge_id,
                openness=openness + DELTA_OPENNESS,
                forceAction=True
            )
            delta = DELTA_OPENNESS

            openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]
            self.max_openness = max(openness, self.max_openness)
            if openness > 0.90:
                self._success = True
                self._took_end_action = True
            return delta, openness_scale, end_effector_correct_position
        else:
            return 0, openness_scale, end_effector_correct_position
        # NOTE: Here determine the relative

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        metrics["init_dist_to_target"] = self.task_info["init_dist_to_target"]
        metrics["dist_to_target"] = self.dist_to_target_func()
        metrics["dist_to_knob"] = self.current_dist_to_knob
        metrics["total_reward"] = np.sum(self._rewards)
        metrics["manipulation_reward"] = np.sum(self._manipulation_rewards)
        metrics["penalties"] = np.sum(self._penalties)
        metrics["manipulation_shaping_moving_reward"] = np.sum(self._manipulation_shaping_moving_rewards)
        metrics["manipulation_shaping_keep_reward"] = np.sum(self._manipulation_shaping_keep_rewards)
        metrics["navigation_reward"] = metrics["total_reward"] - metrics["manipulation_reward"]
        metrics["spl"] = spl_metric(
            success=self._success,
            optimal_distance=self.optimal_distance,
            travelled_distance=self.travelled_distance,
        )
        metrics["success"] = self._success
        metrics["reach_target"] = self.reached_before
        metrics["grasped"] = self.grasped_before
        metrics["max_openness"] = self.max_openness
        metrics["stuck"] = self.stuck
        metrics["step_to_target"] = self.steps_to_target
        metrics["step_to_knob"] = self.steps_to_knob
        metrics["step_to_grasp"] = self.steps_to_grasp
        metrics["reach_knob"] = self.reached_knob_before
        metrics["progress_per_step"] = np.clip(self.max_openness / 0.7, 0, 1) / self.num_steps_taken()
        metrics["normalized_progress"] = np.clip(self.max_openness / 0.7, 0, 1)
        self._metrics = metrics
        return metrics


class OpeningfridgeNWGraspTask(OpeningfridgeGraspTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(5,))

    @property
    def stretch_action_function(self,):
        return self.stretch_controller.continuous_agent_step_nw

    @classmethod
    def continuous_action_dim(self):
        return 5


class OpeningfridgeNWGraspIterativeTask(OpeningfridgeNWGraspTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(6,))

    @classmethod
    def continuous_action_dim(self):
        return 6

    def _step(self, action: np.array) -> RLStepResult:
        if action[-1] > 0:
            action[2] = 0
            action[3] = 0
            # action[4] = 0
        else:
            action[0] = 0
            action[1] = 0
        return super()._step(action[:-1])


class OpeningfridgeGraspIterativeTask(OpeningfridgeGraspTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    @classmethod
    def continuous_action_dim(self):
        return 7

    def _step(self, action: np.array) -> RLStepResult:
        if action[-1] > 0:
            action[2] = 0
            action[3] = 0
            action[4] = 0
        else:
            action[0] = 0
            action[1] = 0
        return super()._step(action[:-1])


class OpeningfridgeGraspSubgoalTask(OpeningfridgeGraspTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    @classmethod
    def continuous_action_dim(self):
        return 8

    def _step(self, action: np.array) -> RLStepResult:
        action = np.array(action)
        self.intend_grasp = action[-1] > 0

        # 0 -> Move Base
        # 1 -> Move Arm
        # 2 -> Do not move
        if action[-2] > 0:
            action[-2] = 0
        else:
            action[-2] = 1

        if action[-1] > 0:
            action[:-2] = 0
            self.intend_grasp = True
            action[-2] = 2

        action = action[:-1]

        action = np.array(action) * self.action_scale[0]
        action = np.clip(action, -1, 1)
        self.action_for_reward = action

        action  = action * self.action_scale
        action = np.clip(action, -1, 1)
        self.action_for_reward = action
        self.task_info["taken_actions"].append(action)

        self._reach_fridge = self.dist_to_target_func() <= self.fridge_reach_distance and \
            self.dist_to_target_func() > 0 and \
            self.stretch_controller.objects_is_visible_in_camera(
                self.task_info["target_object_ids"], which_camera="both"
            )

        self.current_dist_to_knob = self.dist_to_knob()
        self._reach_knob = self.current_dist_to_knob <= self.knob_reach_distance and self._reach_fridge
        if self._reach_knob and self.intend_grasp:
            self.grasped = True
        if not self._reach_knob:
            self.grasped = False

        # Push 
        self.status_opening_push, self.push_openness_scale, ee_inplace_push = self.open_fridge_according_to_action_push_subgoal(
            action, self.action[-2]
        )

        pre_move_info = {
            "agent_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"],
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
            ])),
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }
        event, arm_success, movement_success = self.stretch_action_function(
            action, return_to_start=False
        )
        # print(event.metadata)

        post_move_info =  {
            "agent_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"],
                self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
            ])),
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }

        # Pull
        self.status_opening_pull, self.pull_openness_scale, ee_inplace_pull = self.open_fridge_according_to_action_pull(
            action, pre_move_info, post_move_info
        )

        self.ee_inplace = ee_inplace_pull + ee_inplace_push

        # NOTE: Open the fridge according to the fridge.
        self.last_action_success = bool(
            arm_success and movement_success
        )
        self.last_movement_success = movement_success

        position = self.stretch_controller.get_current_agent_position()
        self.path.append(position)
        self.task_info["followed_path"].append(position)
        self.task_info["ee_followed_path"].append(self.stretch_controller.get_arm_sphere_center())
        self.task_info["action_successes"].append(self.last_action_success)
        self.task_info["openness"].append(self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"])

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            # self.observations.append(self.stretch_controller.last_event.frame)
            pose = {}
            pose["fieldOfView"] = 50
            pose["position"] = {}
            pose["position"]["y"] = 3
            pose["position"]["x"] = self.stretch_controller.controller.last_event.metadata["agent"]["position"]["x"]
            pose["position"]["z"] = self.stretch_controller.controller.last_event.metadata["agent"]["position"]["z"]
            pose["orthographic"] = False
            pose["farClippingPlane"] = 50
            pose["rotation"] = {'x': 90.0, 'y': 0.0, 'z': 0.0}
            # TODO: same as above
            event = self.stretch_controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=1,
                **pose,
            )
            self.observations.append(
                np.concatenate([
                    self.stretch_controller.navigation_camera,
                    self.stretch_controller.manipulation_camera,
                    corruptions.apply_corruption_sequence(np.array(
                            self.stretch_controller.controller.last_event.third_party_camera_frames[1][..., :3]
                        ),[], []
                    )
                ], axis=1)
            )

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={
                "last_action_success": self.last_action_success, "action": action,
                "navigation_action_success": movement_success,
                "arm_action_success": arm_success,
            },
        )
        return step_result


    def open_fridge_according_to_action_push_subgoal(self, action, mode=None):
        end_effector_correct_position = 0
        openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]

        # openness_scale = 1/(1 - np.clip(openness, 0, 0.9))
        openness_scale = 1 if not self.openfridge_use_section_scale else 1/(1 - np.clip(openness, 0, 0.9))

        self.max_openness = max(openness, self.max_openness)
        delta = 0
        end_effector_center = np.squeeze(np.array([
            self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
            self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
        ]))

        self.vec_to_fridge_axis = end_effector_center - self.fridge_corner

        angle = np.arccos(np.sum(self.vec_to_fridge_axis * self.closed_direction) / np.linalg.norm(self.vec_to_fridge_axis))
        angle = 2 * angle / np.pi

        if np.sum(self.vec_to_fridge_axis * self.opened_direction) < 0:
            angle = -angle

        # There suppose to be only one fridge and should be opennable
        current_angle = np.pi * 0.5 * openness
        openable_direction = np.cos(current_angle) * self.closed_direction + np.sin(current_angle) * self.opened_direction
        to_open_direction = -np.cos(current_angle) * self.closed_direction + np.sin(current_angle) * self.opened_direction
        agent_rotation = self.stretch_controller.controller.last_event.metadata[
            "agent"]["rotation"]["y"] / 360 * 2 * np.pi
        agent_moving_vec = np.array([
            np.sin(agent_rotation),
            np.cos(agent_rotation)
        ]) * action[0]

        end_effector_moving_direction =  np.array([
            np.cos(agent_rotation),
            -np.sin(agent_rotation)
        ]) * action[3] + agent_moving_vec
        # get_logger().warning(f"scores: {(np.sum(end_effector_moving_direction * openable_direction))}")
        DELTA_OPENNESS=0.05
        self.project_vec_norm =  np.sum(end_effector_moving_direction * to_open_direction) /\
            (np.linalg.norm(end_effector_moving_direction) + 1e-5)

        dis_along = np.sum(self.vec_to_fridge_axis * openable_direction) / np.linalg.norm(self.vec_to_fridge_axis) * openable_direction
        dis_per = self.vec_to_fridge_axis - dis_along
        dis_to_fridge = np.linalg.norm(dis_per) - self.fridge_size

        if mode == 2:
            return 0, openness_scale, end_effector_correct_position


        if np.sum(self.vec_to_fridge_axis * openable_direction) < 0:
            return 0, openness_scale, end_effector_correct_position

        if angle > openness:
            return 0, openness_scale, end_effector_correct_position

        if not self._reach_fridge:
            return 0, openness_scale, end_effector_correct_position

        if self.push_need_close_knob:
            if not self._reach_knob:
                return 0, openness_scale, end_effector_correct_position
        
        if self.push_need_grasp:
            if not self.grasped:
                return 0, openness_scale, end_effector_correct_position
        
        if np.linalg.norm(self.vec_to_fridge_axis) > self.fridge_width or np.linalg.norm(self.vec_to_fridge_axis) < 0.5 * self.fridge_width:
            return 0, openness_scale, end_effector_correct_position

        if dis_to_fridge > 0.4:
            return 0, openness_scale, end_effector_correct_position


        end_effector_correct_position = 1
        
        if np.sum(end_effector_moving_direction * self.opened_direction) < 0:
            return 0, openness_scale, end_effector_correct_position

        if self.project_vec_norm > 0.6:
            self.stretch_controller.controller.step(
                action='OpenObject',
                objectId=self.fridge_id,
                openness=openness + DELTA_OPENNESS,
                forceAction=True
            )
            delta = DELTA_OPENNESS

            openness = self.stretch_controller.controller.last_event.metadata["objects"][0]["openness"]
            self.max_openness = max(openness, self.max_openness)
            if openness > 0.90:
                self._success = True
                self._took_end_action = True
            return delta, openness_scale, end_effector_correct_position
        else:
            return 0, openness_scale, end_effector_correct_position
        # NOTE: Here determine the relative


class OpeningfridgeGraspTwoStageTask(OpeningfridgeGraspTask):
    def __init__(self, **kwargs):
        super().__init__(**prepare_locals_for_super(locals()))
        # 0 for navigation
        # 1 for manipulation
        self.current_stage = 0

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    @classmethod
    def continuous_action_dim(self):
        return 7

    def _step(self, action: np.array) -> RLStepResult:
        if action[-1] > 0:
            self.current_stage = 1
    
        if self.current_stage == 0:
            action[2] = 0
            action[3] = 0
            action[4] = 0
        else:
            action[0] = 0
            # action[1] = 0
        return super()._step(action[:-1])


class OpeningfridgeNWGraspTwoStageTask(OpeningfridgeNWGraspTask):
    def __init__(self, **kwargs):
        super().__init__(**prepare_locals_for_super(locals()))
        # 0 for navigation
        # 1 for manipulation
        self.current_stage = 0

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(6,))

    @classmethod
    def continuous_action_dim(self):
        return 6

    def _step(self, action: np.array) -> RLStepResult:
        if action[-1] > 0:
            self.current_stage = 1
    
        if self.current_stage == 0:
            action[2] = 0
            action[3] = 0
            # action[4] = 0
        else:
            action[0] = 0
            # action[1] = 0
        return super()._step(action[:-1])


class OpeningfridgeGraspTwoStageHalfTask(OpeningfridgeGraspTask):
    def __init__(self, **kwargs):
        super().__init__(**prepare_locals_for_super(locals()))
        # 0 for navigation
        # 1 for manipulation
        self.current_stage = 0

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    @classmethod
    def continuous_action_dim(self):
        return 7

    def _step(self, action: np.array) -> RLStepResult:
        if action[-1] > 0:
            self.current_stage = 1
    
        down_action = np.clip(
            action[:-1],
            -1 / self.action_scale[0],
            1 / self.action_scale[0]
        )

        if self.current_stage == 0:
            down_action[2] = 0
            down_action[3] = 0
            down_action[4] = 0
            # down_action[0] = 0
        else:
            down_action[0] = down_action[0] * 0.5
            down_action[1] = down_action[1] * 0.5
        return super()._step(down_action)


class OpeningfridgeGraspTwoStageFullTask(OpeningfridgeGraspTask):
    def __init__(self, **kwargs):
        super().__init__(**prepare_locals_for_super(locals()))
        # 0 for navigation
        # 1 for manipulation
        self.current_stage = 0

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    @classmethod
    def continuous_action_dim(self):
        return 7

    def _step(self, action: np.array) -> RLStepResult:
        if action[-1] > 0:
            self.current_stage = 1
    
        down_action = np.clip(
            action[:-1],
            -1 / self.action_scale[0],
            1 / self.action_scale[0]
        )

        if self.current_stage == 0:
            down_action[2] = 0
            down_action[3] = 0
            down_action[4] = 0
        # else:
        #     down_action[0] = down_action[0] * 0.5
        #     down_action[1] = down_action[1] * 0.5
        return super()._step(down_action)


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 


class OpeningfridgeGraspSigmoidTask(OpeningfridgeGraspTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    @classmethod
    def continuous_action_dim(self):
        return 7

    def _step(self, action: np.array) -> RLStepResult:
        # action[-1] > 0: Navigation
        # action[-1] < 0: Manipulation
        # Assue Action Scale are all the same
        down_action = np.clip(
            action[:-1],
            -1 / self.action_scale[0],
            1 / self.action_scale[0]
        )

        manip_scale = sigmoid(action[-1])
        down_action[0] = down_action[0] * (1 - manip_scale)
        down_action[1] = down_action[1] * (1 - manip_scale)
        down_action[2] = down_action[2] * manip_scale
        down_action[3] = down_action[3] * manip_scale
        down_action[4] = down_action[4] * manip_scale
        return super()._step(down_action)


class OpeningfridgeASCTask(OpeningfridgeTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    @classmethod
    def continuous_action_dim(self):
        return 7

    def _step(self, action: List) -> RLStepResult:
        if action[-2] < 0:
            action[0] = 0
            action[1] = 0
        if action[-1] < 0:
            action[2] = 0
            action[3] = 0
            action[4] = 0
        # else:
        #     action[:2] = 0
        return super()._step(action[:-2])


class OpeningfridgeRealTask(OpeningfridgeTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        self.directory = f'output/saved_images/{time_now}'
        os.makedirs(self.directory, exist_ok=True)
        self.bonks = None
        self.visualize_every_frame = True

    def _is_goal_in_range(self) -> bool:
        success = False
        print('I think I found a(n)', self.task_info['object_type'], '. Was I correct? Set success and self.bonks in trace (default false).' )
        ForkedPdb().set_trace()
        return success

    def _step(self, action: int) -> RLStepResult:
        end_early = False

        if self.visualize_every_frame:
            last_frame = self.stretch_controller.controller.last_event.frame
            path2img = os.path.join(self.directory, datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '_nav.png')
            import matplotlib.pyplot as plt; plt.imsave(path2img, last_frame)

            last_manip_frame = self.stretch_controller.controller.last_event.third_party_camera_frames[0]
            path2manipimg = os.path.join(self.directory, datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '_manip.png')
            import matplotlib.pyplot as plt; plt.imsave(path2manipimg, last_manip_frame)


            last_frame_obs = self.stretch_controller.navigation_camera
            path2img = os.path.join(self.directory, datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '_nav_obs.png')
            import matplotlib.pyplot as plt; plt.imsave(path2img, last_frame_obs)

            last_manip_frame_obs = self.stretch_controller.manipulation_camera
            path2manipimg = os.path.join(self.directory, datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '_manip_obs.png')
            import matplotlib.pyplot as plt; plt.imsave(path2manipimg, last_manip_frame_obs)


        if self.num_steps_taken() == self.max_steps - 1:
            print('Reached horizon. Did I run into things a lot? Set self.bonks in trace (default zero).' )
            ForkedPdb().set_trace()
        
        if self.num_steps_taken() % 50 == 0 and self.num_steps_taken() > 0:
            print('Have I run into things more than 10 times? Set end_early=True and self.bonks in trace (default zero).' )
            ForkedPdb().set_trace()

        if self.num_steps_taken() % 50 == 0 and self.num_steps_taken() > 0:
            print('Have I run into things more than 10 times? Set end_early=True and self.bonks in trace (default zero).' )
            ForkedPdb().set_trace()


        action  = action * self.action_scale
        action = np.clip(action, -1, 1)
        get_logger().warning(action)
        self.task_info["taken_actions"].append(action)

        event, arm_success, movement_success = self.stretch_action_function(
            action, return_to_start=False
        )
        # print(event.metadata)

        # NOTE: Open the fridge according to the fridge.
        self.last_action_success = bool(
            arm_success and movement_success
        )
        self.last_movement_success = movement_success

        position = self.stretch_controller.get_current_agent_position()
        self.path.append(position)
        self.task_info["followed_path"].append(position)
        self.task_info["ee_followed_path"].append(self.stretch_controller.get_arm_sphere_center())
        self.task_info["action_successes"].append(self.last_action_success)
        self.task_info["openness"].append(0)

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            # self.observations.append(self.stretch_controller.last_event.frame)
            self.observations.append(
                np.concatenate([
                    self.stretch_controller.navigation_camera,
                    self.stretch_controller.manipulation_camera,
                ], axis=1)
            )

        sr = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={
                "last_action_success": self.last_action_success, "action": action,
                "navigation_action_success": movement_success,
                "arm_action_success": arm_success,
            },
        )

        if end_early:
            self._took_end_action = True
            print('Ending early for too many environment run-ins')

        self.task_info["number_of_interventions"] = self.bonks
        print(str(self.num_steps_taken()))
        return sr
    
    def judge(self) -> float:
        reward = 0.0
        self._rewards.append(reward)
        cur_distance = self.dist_to_target_func()
        self.task_info["dist_to_target"].append(cur_distance)
        self.task_info["rewards"].append(float(reward))
        return reward
    
    # possibly also metrics

class OpeningfridgeTaskSampler(TaskSampler):
    def __init__(
        self,
        args: TaskSamplerArgs,
        opening_fridge_task_type: Type = OpeningfridgeTask,
    ) -> None:
        self.args = args
        # self.rescale_rewards()
        self.opening_fridge_task_type = opening_fridge_task_type
        random.shuffle(self.args.house_inds)

        self.stretch_controller: Optional[StretchController] = None
        self.distance_cache = DynamicDistanceCache(rounding=1)

        # get the total number of tasks assigned to this process
        self.reset_tasks = self.args.max_tasks

        self.resample_same_scene_freq = args.resample_same_scene_freq
        """The number of times to resample the same houses before moving to the next on."""

        self.house_inds_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[OpeningfridgeTask] = None

        self.reachable_positions_map: Dict[int, Vector3] = dict()
        self.starting_positions_map: Dict[int, Vector3] = dict()
        self.objects_in_scene_map: Dict[str, List[str]] = dict()

        self.visible_objects_cache = dict()

        rotate_step_degrees = self.args.controller_args["rotateStepDegrees"]
        self.valid_rotations = np.arange(
            start=0, stop=360, step=rotate_step_degrees
        ).tolist()

        if args.seed is not None:
            self.set_seed(args.seed)

        if args.deterministic_cudnn:
            set_deterministic_cudnn()

        self.target_object_types_set = set(self.args.target_object_types)
        self.obj_type_counter = Counter(
            {obj_type: 0 for obj_type in self.args.target_object_types}
        )

        self.spawn_range_min = self.args.extra_task_spec["spawn_range_min"]
        self.spawn_range_max = self.args.extra_task_spec["spawn_range_max"]
        # self.sample_opening_type = self.args.extra_task_spec["sample_opening_type"]
        self.spawn_range_curriculum_step = self.args.extra_task_spec["spawn_range_curriculum_step"]

        self.knob_reach_distance_min = self.args.extra_task_spec["knob_reach_distance_min"]
        self.knob_reach_distance_max = self.args.extra_task_spec["knob_reach_distance_max"]
        self.knob_dist_curriculum_step = self.args.extra_task_spec["knob_dist_curriculum_step"]
        self.knob_dist_enable_curriculum = self.args.extra_task_spec["knob_dist_enable_curriculum"]

        self.reset()

    def rescale_rewards(self):
        for attribute_name in dir(self.args.reward_config):
            if not attribute_name.startswith('__'):  # Optional: ignore names starting with double underscore
                attribute_value = getattr(self.args.reward_config, attribute_name)
                if not callable(attribute_value):  # Ignore methods
                    print(f'{attribute_name}: {attribute_value}')
                setattr(self.args.reward_config, attribute_name, attribute_value / self.args.max_steps)

    def set_seed(self, seed: int):
        set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.
        # Returns
        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return self.args.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[OpeningfridgeTask]:
        # NOTE: This book-keeping should be done in TaskSampler...
        return self._last_sampled_task

    def close(self) -> None:
        if self.stretch_controller is not None:
            self.stretch_controller.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.
        # Returns
        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def get_nearest_positions(self, world_position: Vector3) -> List[Vector3]:
        """Get the n reachable positions that are closest to the world_position."""
        self.reachable_positions.sort(
            key=lambda p: sum((p[k] - world_position[k]) ** 2 for k in ["x", "z"])
        )
        return self.reachable_positions[
            : min(
                len(self.reachable_positions),
                cfg.training.object_selection.max_agent_positions,
            )
        ]

    def get_nearest_agent_height(self, y_coordinate: float) -> float:
        """Get the nearest valid agent height to a y_coordinate."""
        if len(cfg.agent.valid_agent_heights) == 1:
            return cfg.agent.valid_agent_heights[0]

        min_distance = float("inf")
        out = None
        for height in cfg.agent.valid_agent_heights:
            dist = abs(y_coordinate - height)
            if dist < min_distance:
                min_distance = dist
                out = height
        return out

    @property
    def house_index(self) -> int:
        return self.args.house_inds[self.house_inds_index]

    def is_object_visible(self, object_id: str) -> bool:
        """Return True if object_id is visible without any interaction in the scene.

        This method makes an approximation based on checking if the object
        is hit with a raycast from nearby reachable positions.
        """
        # NOTE: Check the cached visible objects first.
        if (
            self.house_index in self.visible_objects_cache
            and object_id in self.visible_objects_cache[self.house_index]
        ):
            return self.visible_objects_cache[self.house_index][object_id]
        elif self.house_index not in self.visible_objects_cache:
            self.visible_objects_cache[self.house_index] = dict()

        # NOTE: Get the visibility points on the object
        visibility_points = self.stretch_controller.step(
            action="GetVisibilityPoints", objectId=object_id, raise_for_failure=True
        ).metadata["actionReturn"]

        # NOTE: Randomly sample visibility points
        for vis_point in random.sample(
            population=visibility_points,
            k=min(len(visibility_points), cfg.training.object_selection.max_vis_points),
        ):
            # NOTE: Get the nearest reachable agent positions to the target object.
            agent_positions = self.get_nearest_positions(world_position=vis_point)
            for agent_pos in agent_positions:
                agent_pos = agent_pos.copy()
                agent_pos["y"] = self.get_nearest_agent_height(
                    y_coordinate=vis_point["y"]
                )
                event = self.stretch_controller.step(
                    action="PerformRaycast",
                    origin=agent_pos,
                    destination=vis_point,
                )
                hit = event.metadata["actionReturn"]
                if (
                    event.metadata["lastActionSuccess"]
                    and hit["objectId"] == object_id
                    and hit["hitDistance"] < cfg.agent.visibility_distance
                ):
                    self.visible_objects_cache[self.house_index][object_id] = True
                    return True

        self.visible_objects_cache[self.house_index][object_id] = False
        return False

    @property
    def target_objects_in_scene(self) -> Dict[str, List[str]]:
        """Return a map from the object type to the objectIds in the scene."""
        if self.house_index in self.objects_in_scene_map:
            return self.objects_in_scene_map[self.house_index]

        event = self.stretch_controller.step(action="ResetObjectFilter", raise_for_failure=True)
        objects = event.metadata["objects"]
        out = {}
        for obj in objects:
            if obj["objectType"] in self.target_object_types_set:
                if obj["objectType"] not in out:
                    out[obj["objectType"]] = []
                out[obj["objectType"]].append(obj["objectId"])
        self.objects_in_scene_map[self.house_index] = out
        return out

    def sample_target_object_ids(self) -> Tuple[str, List[str]]:
        """Sample target objects.

        Objects returned will all be of the same objectType. Only considers visible
        objects in the house.
        """
        if random.random() < cfg.training.object_selection.p_greedy_target_object:
            for obj_type, count in reversed(self.obj_type_counter.most_common()):
                instances_of_type = self.target_objects_in_scene.get(obj_type, [])

                # NOTE: object type doesn't appear in the scene.
                if not instances_of_type:
                    continue

                visible_ids = []
                for object_id in instances_of_type:
                    if self.is_object_visible(object_id=object_id):
                        visible_ids.append(object_id)

                if visible_ids:
                    self.obj_type_counter[obj_type] += 1
                    return obj_type, visible_ids
        else:
            candidates = dict()
            for obj_type, object_ids in self.target_objects_in_scene.items():
                visible_ids = []
                for object_id in object_ids:
                    if self.is_object_visible(object_id=object_id):
                        visible_ids.append(object_id)

                if visible_ids:
                    candidates[obj_type] = visible_ids

            if candidates:
                return random.choice(list(candidates.items()))

        raise ValueError(f"No target objects in house {self.house_index}.")

    @property
    def reachable_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.reachable_positions_map[self.house_index]

    @property
    def starting_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.starting_positions_map[self.house_index]


    def filter_starting_positions_by_distance(self, fridge_pos, reachable_positions):
        # NOTE: may return -1 if the object is unreachable.
        filtered_reachable_positions = []
        for pos in reachable_positions:
            dist = IThorEnvironment.position_dist(
                fridge_pos,
                pos,
                ignore_y = True
            )
            if dist <= self.spawn_range_max and dist >= self.spawn_range_min:
            # if dist <= self.spawn_range_max:
                filtered_reachable_positions.append(pos)
        return filtered_reachable_positions

    def get_fridge_position(self, house):
        for item in house["objects"]:
            if "Fridge" in item["assetId"]:
                return item["position"]

    def get_reachable_position_in_room_with_fridge(self, reachable_positions, house):
        room_poly_map, room_type_dict = get_rooms_polymap_and_type(house)
        fridge_position = self.get_fridge_position(house)
        room_id = get_room_id_from_location(room_poly_map, fridge_position)

        reachable_position_in_room = []
        room_poly = room_poly_map[room_id]
        for position in reachable_positions:
            point = Point(position["x"], position["z"])
            if room_poly.contains(point):
                reachable_position_in_room.append(position)
        return reachable_position_in_room

    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.increment_scene_index()

        self.house = self.args.houses[self.house_index]

        self.fridge_property = get_fridge_property(self.house)

        event = self.stretch_controller.reset(self.house)

        # Update
        # NOTE: Set reachable positions
        if self.house_index not in self.reachable_positions_map:

            if not event:
                get_logger().warning(f"Initial teleport failing in {self.house_index}.")
                return False

        rp_event = self.stretch_controller.step(action="GetReachablePositions")
        if not rp_event:
            # NOTE: Skip scenes where GetReachablePositions fails
            get_logger().warning(
                f"GetReachablePositions failed in {self.house_index}"
            )
            return False

        fridge_pos = self.get_fridge_position(self.house)
        reachable_positions = rp_event.metadata["actionReturn"]
        fridge_pos = {
            "x": self.fridge_property["fridge_closed_center"][0],
            "z": self.fridge_property["fridge_closed_center"][1]
        }


        reachable_positions_in_room = self.get_reachable_position_in_room_with_fridge(reachable_positions, self.house)
        starting_positions = self.filter_starting_positions_by_distance(fridge_pos, reachable_positions_in_room)
        get_logger().warning(f"# Starting Position: {len(starting_positions)} in House # {self.house_index}")

        self.reachable_positions_map[self.house_index] = reachable_positions
        self.starting_positions_map[self.house_index] = starting_positions
        return True

    def increment_scene_index(self):
        self.house_inds_index = (self.house_inds_index + 1) % len(self.args.house_inds)

    def close_all_fridges(self, fridge_ids, init_openness):
        for fridge_id in fridge_ids:
            self.stretch_controller.controller.step(
                action='OpenObject',
                objectId=fridge_id,
                openness=init_openness,
                forceAction=True
            )
    
    def reset_arm_pos(self):
        self.stretch_controller.step(
            **dict(action='MoveArm', position=dict(x=0, y=0.5, z=0.0),)
        )
        curr_wrist = self.stretch_controller.get_arm_wrist_raw_rotation()
        rotate_angle = -curr_wrist
        if curr_wrist < 181 and curr_wrist > 70:
            rotate_angle += 360
        action_dict = dict(
            action="RotateWristRelative", yaw=rotate_angle, # **ADDITIONAL_ARM_ARGS,
            renderImage=True,
            # returnToStart = return_to_start
        )
        event = self.stretch_controller.step(**action_dict)

    def randomize_lighting_and_materials(self):
        randomize_wall_and_floor_materials(self.house)
        randomize_lighting(self.house)
        self.stretch_controller.reset(scene=self.house)
        return True

    def next_task(self, force_advance_scene: bool = False) -> Optional[OpeningfridgeTask]:
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

        # NOTE: Setup the Controller
        if self.stretch_controller is None:
            # self.stretch_controller = Controller(**self.args.controller_args)
            self.stretch_controller = StretchController(**self.args.controller_args)
            get_logger().info(
                f"Using Controller commit id: {self.stretch_controller.controller._build.commit_id}"
            )
            while not self.increment_scene():
                pass

        # NOTE: determine if the house should be changed.
        if (
            force_advance_scene
            or (
                self.resample_same_scene_freq > 0
                and self.episode_index % self.resample_same_scene_freq == 0
            )
            or self.episode_index == 0
        ):
            while not self.increment_scene():
                pass

        # NOTE: No target Object in Openfridge
        while True:
            try:
                # NOTE: The loop avoid a very rare edge case where the agent
                # starts out trapped in some part of the room.
                ""
                target_object_type, target_object_ids = self.sample_target_object_ids()
                break
            except ValueError:
                while not self.increment_scene():
                    pass

        if random.random() < cfg.procthor.p_randomize_materials:
            self.randomize_lighting_and_materials()
            self.stretch_controller.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.stretch_controller.controller.step(action="ResetMaterials", raise_for_failure=True)

        fridge_ids = [self.fridge_property["fridge_id"]]
        self.stretch_controller.step(
            action="SetObjectFilter",
            objectIds=fridge_ids,
            raise_for_failure=True,
        )

        fridge_size = 0.025

        self.fridge_property["fridge_corner"] = self.fridge_property["fridge_corner"] + fridge_size * self.fridge_property["closed_direction"]
        self.fridge_property["fridge_width"] = self.fridge_property["fridge_width"] - fridge_size
        self.fridge_property["fridge_size"] = fridge_size


        self.close_all_fridges(fridge_ids, init_openness=self.args.extra_task_spec["init_openness"])

        # NOTE: Set agent pose
        standing = (
            {}
            if self.args.controller_args["agentMode"] == "locobot"
            else {"standing": True}
        )
        for _ in range(10):
            # Retry ten times to make it reasonable
            starting_pose = AgentPose(
                position=random.choice(self.starting_positions),
                rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
                horizon=0,
                **standing,
            )
            self.reset_arm_pos()
            event = self.stretch_controller.step(action="TeleportFull", **starting_pose)
            # self.stretch_controller.reset_obs_his()
            if not event:
                get_logger().warning(
                     f"Teleport failing in {self.house_index} at {starting_pose}"
                )
            else:
                break
        #     continue
        # break
        self.reset_arm_pos()
        self.stretch_controller.reset_obs_his()

        self.episode_index += 1
        self.args.max_tasks -= 1

        task_id = (f'opening_fridge_{target_object_type}'
            f'__proc_{str(self.args.process_ind)}'
            f'__epidx_{str(self.episode_index)}')

        self.current_knob_reach_distance = self.args.extra_task_spec["knob_reach_distance"]
        if self.knob_dist_enable_curriculum:
            self.current_knob_reach_distance = self.knob_reach_distance_max
            curriculum_idx = 0
            while curriculum_idx < len(self.knob_dist_curriculum_step) - 1 and\
                  self.episode_index > self.knob_dist_curriculum_step[curriculum_idx]:
                self.current_knob_reach_distance -= 0.05
                curriculum_idx += 1
            self.current_knob_reach_distance = max(
                self.current_knob_reach_distance, 
                self.knob_reach_distance_min
            )

        self._last_sampled_task = self.opening_fridge_task_type(
            controller=self.stretch_controller,
            action_scale=self.args.action_scale,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type="l2",
            house=self.house,
            distance_cache=self.distance_cache,
            visualize= self.args.visualize if self.args.visualize else None, 
            task_info={
                "mode": self.args.mode,
                "process_ind": self.args.process_ind,
                "house_name": str(self.house_index),
                "rooms": len(self.house["rooms"]),
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
                "id": task_id,
                "fridge_reach_distance": self.args.extra_task_spec["fridge_reach_distance"],
                "knob_reach_distance": self.current_knob_reach_distance,
                "openfridge_use_section_scale": self.args.extra_task_spec["openfridge_use_section_scale"],
                "init_openness": self.args.extra_task_spec["init_openness"],
                "push_need_grasp": self.args.extra_task_spec["push_need_grasp"],
                "push_need_close_knob": self.args.extra_task_spec["push_need_close_knob"],
                "pull_need_grasp": self.args.extra_task_spec["pull_need_grasp"],
                **self.fridge_property
            },
        )
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.args.max_tasks = self.reset_tasks
        self.house_inds_index = 0



def get_fake_fridge_property():
    return {
        "fridge_id": "Realfridge",
        "fridge_corner": np.array([0,0]),
        "fridge_width": 1.0,
        "fridge_size": 0.025,
        "closed_direction": np.array([0,0]),
        "opened_direction": np.array([0,0]),
        "fridge_closed_center":  np.array([0,0])
    }

class RealOpeningfridgeTaskSampler(OpeningfridgeTaskSampler):
    def __init__(self, args: TaskSamplerArgs, opening_fridge_task_type: Type = OpeningfridgeRealTask) -> None:
        super().__init__(args, opening_fridge_task_type)
        assert cfg.eval

        self.num_starting_positions = 3

        self.real_object_index = 0
        self.starting_position_index = 0
        self.host = cfg.real.host 
        get_logger().debug('Initializing on host '+ self.host+ '. Verify that is the correct platform before continuing.')
        ForkedPdb().set_trace()
        self.args.controller_args = {
            "host":self.host, #+".corp.ai2"
            "port":9000, "width": 384, "height":224,
            "reverse_at_boundary": cfg.mdp.reverse_at_boundary,
            "smaller_action": cfg.mdp.smaller_action
        }
        self.object_set = ['fridgeway'] # base set - robothor and # self.args.target_object_types
        self.args.max_tasks = len(self.object_set) * self.num_starting_positions


    def reset_arm_pos(self):
        arm_proprio = self.stretch_controller.get_arm_proprioception()
        y = 0.5 - (arm_proprio["position"]["y"]  + 0.16297650337219238)
        # y = 0.8 - (arm_proprio["position"]["y"]  + 0.16297650337219238)
        w_dist_to_init = arm_proprio["rotation"]["y"]
        rotation_value = 0
        if 0 <= w_dist_to_init <= 180:
            rotation_value = 180 - w_dist_to_init
        else:
            rotation_value = -180 - w_dist_to_init
        arm_and_move_actions = [
            {"action": "MoveArmBase", "args": {"move_scalar": y}},
            {"action": "MoveArmExtension", "args": {"move_scalar": -arm_proprio[2]}},
            # {"action": "MoveWrist", "args": {"move_scalar": rotation_value}},
        ]

        self.stretch_controller.controller.step({
            "action":arm_and_move_actions
        })
        # return 

    def next_task(self) -> Optional[OpeningfridgeRealTask]:
        if self.args.max_tasks <= 0:
            return None
        
        # identify which model is being evaluated
        eval_checkpoint_name = "evaluation_model_not_specified"
        if cfg.checkpoint is None and cfg.pretrained_model.name is not None:
            eval_checkpoint_name = cfg.pretrained_model.name
        elif cfg.checkpoint is not None:
            eval_checkpoint_name = cfg.checkpoint.split('/')[-1]
        eval_checkpoint_name = eval_checkpoint_name.split('.')[0]

        # NOTE: Setup the Controller
        if self.stretch_controller is None:
            self.stretch_controller = StretchRealController(**self.args.controller_args)
            self.stretch_controller.controller.step("Pass")
        
        # pick object and initialize task
        while True:
            if cfg.real.specific_object is not None:
                target_object = {"object_type":cfg.real.specific_object}
            else:
                target_object = {"object_type":self.object_set[self.real_object_index]}
            target_object["object_id"] = target_object['object_type'].strip() + "|1|1|1"
            skip=False
            print('I am now seeking a(n)', target_object['object_type'].strip(), 
                    'from starting potiion ',str(self.starting_position_index+1),
                    '. Continue when ready or set skip=True.')
            ForkedPdb().set_trace()

            if skip==True:
                self.real_object_index += 1
                if self.real_object_index == len(self.object_set):
                    self.real_object_index = 0
                    self.starting_position_index +=1 
                continue
            # do this to reset the camera/sensors after moving the robot/resetting environment
            self.stretch_controller.controller.step({"action": "Pass"})

            self.reset_arm_pos()

            task_start = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            task_id = (f'realRobothor__{target_object["object_type"]}'
                        f'__starting_position_{str(self.starting_position_index+1)}'
                        f'__model_{eval_checkpoint_name}'
                        f'__task_start_{task_start}')

            self.args.max_tasks -= 1
            self.episode_index += 1
            
            self._last_sampled_task = self.opening_fridge_task_type(
                visualize=True,
                controller=self.stretch_controller,
                action_scale=self.args.action_scale,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type="realWorld",
                distance_cache=self.distance_cache,
                task_info={
                    "mode": self.args.mode,
                    "house_name": 'Real Back Apt',
                    "target_object_ids": [target_object['object_id']],
                    "object_type": target_object['object_type'],
                    "starting_pose": {},
                    "mirrored": False,
                    "id": task_id,
                    "fridge_reach_distance": 0,
                    "knob_reach_distance": 0,
                    "openfridge_use_section_scale": 0,
                    "init_openness": 0,
                    **get_fake_fridge_property()
                },
            )
            self.real_object_index += 1
            if self.real_object_index == len(self.object_set):
                self.real_object_index = 0
                self.starting_position_index +=1 

            return self._last_sampled_task
        

class OpeningfridgeRealGraspTask(OpeningfridgeRealTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(6,))

    @classmethod
    def continuous_action_dim(self):
        return 6

    def _step(self, action: np.array) -> RLStepResult:
        action = np.array(action)
        self.intend_grasp = action[-1] > 0
        if action[-1] > 0:
            action[:-1] = 0
            self.intend_grasp = True
            # print('Intended to Grasp')
            
            ag = False
            get_logger().warning("Actual Grasp Or Not?")
            ForkedPdb().set_trace()
            if ag:
                self.stretch_controller.grasp(self.directory)

        action = action[:-1]
        return super()._step(action)



class RealOpeningfridgeGraspTaskSampler(OpeningfridgeTaskSampler):
    def __init__(self, args: TaskSamplerArgs, opening_fridge_task_type: Type = OpeningfridgeRealGraspTask) -> None:
        super().__init__(args, opening_fridge_task_type)
        assert cfg.eval

        self.num_starting_positions = 3

        self.real_object_index = 0
        self.starting_position_index = 0
        self.host = cfg.real.host 
        get_logger().debug('Initializing on host '+ self.host+ '. Verify that is the correct platform before continuing.')
        ForkedPdb().set_trace()
        self.args.controller_args = {
            "host":self.host, #+".corp.ai2"
            "port":50051, "width": 396, "height":224,
            "reverse_at_boundary": cfg.mdp.reverse_at_boundary,
            "smaller_action": cfg.mdp.smaller_action
        }
        self.object_set = ['fridgeway'] # base set - robothor and # self.args.target_object_types
        self.args.max_tasks = len(self.object_set) * self.num_starting_positions


    def reset_arm_pos(self):
        arm_proprio = self.stretch_controller.get_arm_proprioception()
        y = 0.5 - (arm_proprio[1]  + 0.16297650337219238)
        # y = 0.8 - (arm_proprio["position"]["y"]  + 0.16297650337219238)
        w_dist_to_init = arm_proprio[-1]
        rotation_value = 0
        if 0 <= w_dist_to_init <= 180:
            rotation_value = 180 - w_dist_to_init
        else:
            rotation_value = -180 - w_dist_to_init
        arm_and_move_actions = [
            {"action": "MoveArmBase", "args": {"move_scalar": y}},
            {"action": "MoveArmExtension", "args": {"move_scalar": -arm_proprio[2]}},
            # {"action": "MoveWrist", "args": {"move_scalar": rotation_value}},
        ]

        self.stretch_controller.controller.step({
            "action":arm_and_move_actions
        })
        # return 

    def next_task(self) -> Optional[OpeningfridgeRealTask]:
        if self.args.max_tasks <= 0:
            return None
        
        # identify which model is being evaluated
        eval_checkpoint_name = "evaluation_model_not_specified"
        if cfg.checkpoint is None and cfg.pretrained_model.name is not None:
            eval_checkpoint_name = cfg.pretrained_model.name
        elif cfg.checkpoint is not None:
            eval_checkpoint_name = cfg.checkpoint.split('/')[-1]
        eval_checkpoint_name = eval_checkpoint_name.split('.')[0]

        # NOTE: Setup the Controller
        if self.stretch_controller is None:
            self.stretch_controller = StretchRealController(**self.args.controller_args)
            self.stretch_controller.controller.step("Pass")
        
        # pick object and initialize task
        while True:
            if cfg.real.specific_object is not None:
                target_object = {"object_type":cfg.real.specific_object}
            else:
                target_object = {"object_type":self.object_set[self.real_object_index]}
            target_object["object_id"] = target_object['object_type'].strip() + "|1|1|1"
            skip=False
            print('I am now seeking a(n)', target_object['object_type'].strip(), 
                    'from starting potiion ',str(self.starting_position_index+1),
                    '. Continue when ready or set skip=True.')
            ForkedPdb().set_trace()

            if skip==True:
                self.real_object_index += 1
                if self.real_object_index == len(self.object_set):
                    self.real_object_index = 0
                    self.starting_position_index +=1 
                continue
            # do this to reset the camera/sensors after moving the robot/resetting environment
            self.stretch_controller.controller.step("Pass")

            self.reset_arm_pos()

            task_start = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            task_id = (f'realRobothor__{target_object["object_type"]}'
                        f'__starting_position_{str(self.starting_position_index+1)}'
                        f'__model_{eval_checkpoint_name}'
                        f'__task_start_{task_start}')

            self.args.max_tasks -= 1
            self.episode_index += 1
            
            self._last_sampled_task = self.opening_fridge_task_type(
                visualize=True,
                controller=self.stretch_controller,
                action_scale=self.args.action_scale,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type="realWorld",
                distance_cache=self.distance_cache,
                task_info={
                    "mode": self.args.mode,
                    "house_name": 'Real Back Apt',
                    "target_object_ids": [target_object['object_id']],
                    "object_type": target_object['object_type'],
                    "starting_pose": {},
                    "mirrored": False,
                    "id": task_id,
                    "fridge_reach_distance": 0,
                    "knob_reach_distance": 0,
                    "openfridge_use_section_scale": 0,
                    "init_openness": 0,        
                    "push_need_grasp": self.args.extra_task_spec["push_need_grasp"],
                    "push_need_close_knob": self.args.extra_task_spec["push_need_close_knob"],
                    "pull_need_grasp": self.args.extra_task_spec["pull_need_grasp"],
                    **get_fake_fridge_property()
                },
            )
            self.real_object_index += 1
            if self.real_object_index == len(self.object_set):
                self.real_object_index = 0
                self.starting_position_index +=1 

            return self._last_sampled_task
        