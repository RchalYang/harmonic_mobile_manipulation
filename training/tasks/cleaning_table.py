import os
import sys
import pdb
import random
import copy
import time
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


class CleaningTableTask(Task[Controller]):
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
        self.task_info["dist_to_table"] = []
        self.task_info["target_locations"] = []

        # self.task_info["success_progress"] = task_info["success_progress"]
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

        self.task_info["init_dist_to_target"] = self.last_distance

        if self.task_info["init_dist_to_target"] <= 1:
            dist_cat = "<1"
        elif self.task_info["init_dist_to_target"] < 2:
            dist_cat = "1.-2."
        elif self.task_info["init_dist_to_target"] < 3:
            dist_cat = "2.-3."
        elif self.task_info["init_dist_to_target"] < 4:
            dist_cat = "3.-4."
        else:
            dist_cat = ">4."
        self.task_info["init_dist_category"] = dist_cat

        # else:
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
        self._metrics = None

        self.reached_before = False
        self.reached_table_before = False
        self.steps_to_table = self.max_steps
        if self.distance_type != "realWorld":
            self.last_distance_to_table = self.dist_to_table()
            self.closest_distance_to_table = self.last_distance_to_table
            self.task_info["dist_to_table"].append(self.last_distance_to_table)
        self.stuck = False

        #self.agent_table_reach_distance = self.task_info["agent_table_reach_distance"]
        #self.ee_table_reach_distance = self.task_info["ee_table_reach_distance"]
        self.previous_num_dirt = self.task_info["init_dirt_points"]
        self.table_reach_distance = self.task_info["table_reach_distance"]

    def min_approx_geo_distance_to_target(self) -> float:
        agent_position = self.stretch_controller.last_event.metadata["agent"]["position"]
        room_i_with_agent = nearest_room_to_point(
            point=agent_position, room_polygons=self.room_polygons
        )
        room_id_with_agent = int(
            self.house["rooms"][room_i_with_agent]["id"].split("|")[-1]
        )
        if room_id_with_agent in self.room_id_to_open_key:
            room_id_with_agent = self.room_id_to_open_key[room_id_with_agent]

        return get_approx_geo_dist(
            target_object_type=self.task_info["object_type"],
            agent_position=agent_position,
            house=self.house,
            controller=self.stretch_controller,
            room_polygons=self.room_polygons,
            room_connection_graph=self.room_connection_graph,
            room_id_to_open_key=self.room_id_to_open_key,
            room_id_with_agent=room_id_with_agent,
            house_name=self.task_info["house_name"],
        )

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

    def min_geo_distance_to_target(self) -> float:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = None
        for object_id in self.task_info["target_object_ids"]:
            geo_dist = distance_to_object_id(
                controller=self.stretch_controller,
                distance_cache=self.distance_cache,
                object_id=object_id,
                house_name=self.task_info["house_name"],
            )
            if (min_dist is None and geo_dist >= 0) or (
                geo_dist >= 0 and geo_dist < min_dist
            ):
                min_dist = geo_dist
        if min_dist is None:
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

    def _step(self, action: np.array) -> RLStepResult:

        action  = action * self.action_scale
        action = np.clip(action, -1, 1)
        self.action_for_reward = action
        self.task_info["taken_actions"].append(action)

        ### for the Kiana Experiment (tm)

        # # self._success = self._is_goal_in_range()
        self._reach_door = self.dist_to_target_func() <= self.table_reach_distance and \
            self.dist_to_target_func() > 0 and \
            self.stretch_controller.objects_is_visible_in_camera(
                self.task_info["target_object_ids"], which_camera="both"
            )

        self._ee_reach_table = self.dist_to_table() <= 0.4 and \
            self.dist_to_table() > 0 # and \

        pre_move_info = {
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["y"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }

        event, arm_success, movement_success= self.stretch_controller.continuous_agent_step(
            action, return_to_start=False
        )

        post_move_info =  {
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["y"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }

        self.ee_moved_distance = np.linalg.norm(
            post_move_info["end_effector_pos"] - pre_move_info["end_effector_pos"]
        )

        # NOTE: Open the door according to the door.
        self.last_action_success = bool(
            arm_success and movement_success
        )
        self.last_movement_success = movement_success

        position = self.stretch_controller.get_current_agent_position()
        self.path.append(position)
        self.task_info["followed_path"].append(position)
        self.task_info["ee_followed_path"].append(self.stretch_controller.get_arm_sphere_center())
        self.task_info["action_successes"].append(self.last_action_success)

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

    def get_cleaning_table_reward(self) -> float:
        event=self.stretch_controller.step(
            action="GetDirtMeta"
        )
        if isinstance(event.metadata['actionReturn'], list):
            current_num_dirt = len(event.metadata['actionReturn'])
        elif isinstance(event.metadata['actionReturn'], str):
            current_num_dirt = 0
        num_reduced = self.previous_num_dirt - current_num_dirt
        self.previous_num_dirt = current_num_dirt
        # self._success = current_num_dirt < 0.1
        self.progress = 1 - current_num_dirt / self.task_info["init_dirt_points"]
        self._success = self.progress > self.task_info["success_progress"]
        self._took_end_action = self._success
        return num_reduced * self.reward_config.per_dirt_reward

    def get_table_pos(self):
        table_pos = self.stretch_controller.get_object_position("TableToClean")
        # Currently Ignore the offset along vertical
        return table_pos

    def dist_to_table(self):
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        table_pos = self.get_table_pos()
        ee_position = self.stretch_controller.get_arm_sphere_center()
        dist = position_dist(
            ee_position, table_pos, ignore_y=False
        )
        return dist

    def manipulation_shaping(self) -> float:
        cur_distance = self.dist_to_table()
        self.task_info["dist_to_table"].append(cur_distance)
        if self.reward_config.manipulation_shaping_weight == 0.0:
            return 0

        arm_pose = self.stretch_controller.get_relative_stretch_current_arm_state()
        arm_y = arm_pose["y"]
        distance_moved = max(self.closest_distance_to_table - cur_distance, 0) * \
            self.reward_config.manipulation_shaping_moving_scale
        self.closest_distance_to_table = min(self.closest_distance_to_table, cur_distance)
        # Encourage moving to the Table
        #scale = np.exp(self.reward_config.manipulation_shaping_scale * cur_distance)
        moving_reward = (arm_y > 0.67) * distance_moved * self.reward_config.manipulation_shaping_weight
        return moving_reward

    def judge(self) -> float:
        """Judge the last event."""
        penalties = self.reward_config.energy_penalty * self.ee_moved_distance

        too_close_to_table = self.dist_to_target_func() <= 0.5
        penalties += too_close_to_table * self.reward_config.too_close_penalty

        penalties += self.reward_config.step_penalty
        if not self.last_movement_success: #and "Look" not in self.task_info["taken_actions"][-1]:
            penalties += self.reward_config.failed_action_penalty

        reward = penalties

        reward += self.shaping()

        if self._reach_door:
            if not self.reached_before:
                reward += self.reward_config.goal_success_reward
                self.reached_before = True

        manip_reward = self.reward_config.cleaning_reward * self.get_cleaning_table_reward()
        if self._success:
            reward += self.reward_config.complete_task_reward

        manip_shaping_moving_reward = self.manipulation_shaping()
            
        if self._ee_reach_table:
            if not self.reached_table_before:
                manip_reward += self.reward_config.table_success_reward * self.reward_config.manipulation_shaping_weight
                self.reached_table_before = True
                self.steps_to_table = self.num_steps_taken() + 1

        manip_reward += (manip_shaping_moving_reward *  (1 - self.reached_table_before))
        reward += manip_reward
        # get_logger().warning(reward)
        if self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
        self._manipulation_rewards.append(float(manip_reward))
        self._penalties.append(float(penalties))
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
        metrics["dist_to_table"] = self.dist_to_table()
        metrics["total_reward"] = np.sum(self._rewards)
        metrics["manipulation_reward"] = np.sum(self._manipulation_rewards)
        metrics["penalties"] = np.sum(self._penalties)
        metrics["navigation_reward"] = metrics["total_reward"] - metrics["manipulation_reward"]
        metrics["spl"] = spl_metric(
            success=self._success,
            optimal_distance=self.optimal_distance,
            travelled_distance=self.travelled_distance,
        )
        metrics["success"] = self._success
        metrics["reach_target"] = self.reached_before
        metrics["cleaning_progress"] = self.progress

        metrics["progress_per_step"] = np.clip(self.progress / self.task_info["success_progress"], 0, 1) / self.num_steps_taken()
        metrics["normalized_progress"] = np.clip(self.progress / self.task_info["success_progress"], 0, 1)

        self._metrics = metrics
        return metrics


class CleaningTableNWTask(CleaningTableTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(4,))

    @classmethod
    def continuous_action_dim(self):
        return 4

    def _step(self, action: np.array) -> RLStepResult:
        action  = np.array(action) * self.action_scale[0]
        action = np.clip(action, -1, 1)
        self.action_for_reward = action
        self.task_info["taken_actions"].append(action)

        self._reach_door = self.dist_to_target_func() <= self.table_reach_distance and \
            self.dist_to_target_func() > 0 and \
            self.stretch_controller.objects_is_visible_in_camera(
                self.task_info["target_object_ids"], which_camera="both"
            )

        self._ee_reach_table = self.dist_to_table() <= 0.4 and \
            self.dist_to_table() > 0 # and \
        
        pre_move_info = {
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["y"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }

        event, arm_success, movement_success= self.stretch_controller.continuous_agent_step_nw(
            action, return_to_start=False
        )

        post_move_info =  {
            "end_effector_pos": np.squeeze(np.array([
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["y"],
                self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
            ]))
        }

        self.ee_moved_distance = np.linalg.norm(
            post_move_info["end_effector_pos"] - pre_move_info["end_effector_pos"]
        )

        # NOTE: Open the door according to the door.
        self.last_action_success = bool(
            arm_success and movement_success
        )
        self.last_movement_success = movement_success

        position = self.stretch_controller.get_current_agent_position()
        self.path.append(position)
        self.task_info["followed_path"].append(position)
        self.task_info["ee_followed_path"].append(self.stretch_controller.get_arm_sphere_center())
        self.task_info["action_successes"].append(self.last_action_success)

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
            # pose["ee_position"] = {}
            # pose["ee_position"]["y"] = 3
            # pose["ee_position"]["x"] = self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["x"]
            # pose["ee_position"]["z"] = self.stretch_controller.controller.last_event.metadata["arm"]["handSphereCenter"]["z"]
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



class CleaningTableIterativeTask(CleaningTableTask):
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

class CleaningTableNWIterativeTask(CleaningTableNWTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(5,))

    @classmethod
    def continuous_action_dim(self):
        return 5

    def _step(self, action: np.array) -> RLStepResult:
        if action[-1] > 0:
            action[2] = 0
            action[3] = 0
            # action[4] = 0
        else:
            action[0] = 0
            action[1] = 0
        return super()._step(action[:-1])


class CleaningTableSubgoalTask(CleaningTableTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    @classmethod
    def continuous_action_dim(self):
        return 7


    def _step(self, action: np.array) -> RLStepResult:
        # action_scale = np.concatenate([
            # self.action_scale, 
        action = np.array(action) * self.action_scale[0]
        action = np.clip(action, -1, 1)
        self.action_for_reward = action
        self.task_info["taken_actions"].append(action)

        self._reach_door = self.dist_to_target_func() <= self.table_reach_distance and \
            self.dist_to_target_func() > 0 and \
            self.stretch_controller.objects_is_visible_in_camera(
                self.task_info["target_object_ids"], which_camera="both"
            )

        self._ee_reach_table = self.dist_to_table() <= 0.4 and \
            self.dist_to_table() > 0 # and \

        event, arm_success, movement_success= self.stretch_controller.continuous_agent_step_with_subgoal(
            action, return_to_start=False
        )
        # Pull

        # NOTE: Open the door according to the door.
        self.last_action_success = bool(
            arm_success and movement_success
        )
        self.last_movement_success = movement_success

        position = self.stretch_controller.get_current_agent_position()
        self.path.append(position)
        self.task_info["followed_path"].append(position)
        self.task_info["ee_followed_path"].append(self.stretch_controller.get_arm_sphere_center())
        self.task_info["action_successes"].append(self.last_action_success)

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


class CleaningTableTwoStageTask(CleaningTableTask):
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
            action[4] = 0
        else:
            action[0] = 0
            # action[1] = 0
        return super()._step(action[:-1])


class CleaningTableNWTwoStageTask(CleaningTableNWTask):
    def __init__(self, **kwargs):
        super().__init__(**prepare_locals_for_super(locals()))
        # 0 for navigation
        # 1 for manipulation
        self.current_stage = 0

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(5,))

    @classmethod
    def continuous_action_dim(self):
        return 5

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


class CleaningTableTwoStageHalfTask(CleaningTableTask):
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

        down_action = np.clip(
            action[:-1],
            -1 / self.action_scale,
            1 / self.action_scale
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


class CleaningTableTwoStageFullTask(CleaningTableTask):
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

        down_action = np.clip(
            action[:-1],
            -1 / self.action_scale,
            1 / self.action_scale
        )

        if self.current_stage == 0:
            down_action[2] = 0
            down_action[3] = 0
            down_action[4] = 0
        return super()._step(down_action)


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 


class CleaningTableSigmoidTask(CleaningTableTask):
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(6,))

    @classmethod
    def continuous_action_dim(self):
        return 6

    def _step(self, action: np.array) -> RLStepResult:
        # action[-1] > 0: Navigation
        # action[-1] < 0: Manipulation

        down_action = np.clip(
            action[:-1],
            -1 / self.action_scale[0],
            -1 / self.action_scale[0]
        )

        manip_scale = sigmoid(action[-1])
        down_action[0] = down_action[0] * (1 - manip_scale)
        down_action[1] = down_action[1] * (1 - manip_scale)
        down_action[2] = down_action[2] * manip_scale
        down_action[3] = down_action[3] * manip_scale
        down_action[4] = down_action[4] * manip_scale
        return super()._step(down_action)


class CleaningTableTaskSampler(TaskSampler):
    def __init__(
        self,
        args: TaskSamplerArgs,
        cleaning_table_task_type: Type = CleaningTableTask,
    ) -> None:
        self.args = args
        self.cleaning_table_task_type = cleaning_table_task_type
        random.shuffle(self.args.house_inds)

        self.stretch_controller: Optional[StretchController] = None
        self.distance_cache = DynamicDistanceCache(rounding=1)

        # get the total number of tasks assigned to this process
        self.reset_tasks = self.args.max_tasks

        self.resample_same_scene_freq = args.resample_same_scene_freq
        """The number of times to resample the same houses before moving to the next on."""

        self.house_inds_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[CleaningTableTask] = None

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

        self.reset()

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
    def last_sampled_task(self) -> Optional[CleaningTableTask]:
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

    def get_table_position(self, house):
        for item in house["objects"]:
            if "CleaningTable" in item["assetId"]:
                return item["position"]

    def get_reachable_position_in_room_with_table(self, reachable_positions, house):
        room_poly_map, room_type_dict = get_rooms_polymap_and_type(house)
        cleaning_table_position = self.get_table_position(house)
        room_id = get_room_id_from_location(room_poly_map, cleaning_table_position)

        reachable_position_in_room = []
        room_poly = room_poly_map[room_id]
        for position in reachable_positions:
            point = Point(position["x"], position["z"])
            if room_poly.contains(point):
                reachable_position_in_room.append(position)
        return reachable_position_in_room

    def filter_starting_positions_by_distance(self, table_pos, reachable_positions):
        # NOTE: may return -1 if the object is unreachable.
        filtered_reachable_positions = []
        for pos in reachable_positions:
            dist = IThorEnvironment.position_dist(
                table_pos,
                pos,
                ignore_y = True
            )
            if dist <= self.spawn_range_max and dist >= self.spawn_range_min:
            # if dist <= self.spawn_range_max:
                filtered_reachable_positions.append(pos)
        return filtered_reachable_positions

    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.increment_scene_index()

        self.house = self.args.houses[self.house_index]

        event = self.stretch_controller.reset(self.house)

        # NOTE: Set reachable positions
        if self.house_index not in self.reachable_positions_map:
            # pose = self.house["metadata"]["agent"].copy()
            # if self.args.controller_args["agentMode"] == "locobot":
            #     del pose["standing"]
            # event = self.stretch_controller.step(action="TeleportFull", **pose)
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
        table_pos = self.get_table_position(self.house)
        # table_pos = {
        #     "x": self.door_property["door_closed_center"][0],
        #     "z": self.door_property["door_closed_center"][1]
        # }
        reachable_positions = rp_event.metadata["actionReturn"]
        reachable_positions_in_room = self.get_reachable_position_in_room_with_table(reachable_positions, self.house)
        starting_positions = self.filter_starting_positions_by_distance(table_pos, reachable_positions_in_room)
        get_logger().warning(f"# Starting Position: {len(starting_positions)} in House # {self.house_index}")
        self.reachable_positions_map[self.house_index] = reachable_positions_in_room
        self.starting_positions_map[self.house_index] = starting_positions
        return True

    def increment_scene_index(self):
        self.house_inds_index = (self.house_inds_index + 1) % len(self.args.house_inds)

    def close_all_doors(self, door_ids):
        for door_id in door_ids:
            self.stretch_controller.controller.step(
                action='OpenObject',
                objectId=door_id,
                openness=0,
                forceAction=True
            )

    def reset_arm_pos(self):
        self.stretch_controller.step(
            **dict(action='MoveArm', position=dict(x=0, y=0.8, z=0.0),)
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

    def next_task(self, force_advance_scene: bool = False) -> Optional[CleaningTableTask]:
        # NOTE: Stopping condition
        # force_advance_scene = True
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

        # NOTE: No target Object in Opendoor
        while True:
            try:
                # NOTE: The loop avoid a very rare edge case where the agent
                # starts out trapped in some part of the room.
                target_object_type, target_object_ids = self.sample_target_object_ids()
                assert len(target_object_ids) == 1 and target_object_ids[0] == "TableToClean", f"Wrong Object Sampled: {target_object_ids}"
                break
            except ValueError:
                while not self.increment_scene():
                    pass

        # event = self.stretch_controller.reset(self.house)

        if random.random() < cfg.procthor.p_randomize_materials:
            self.randomize_lighting_and_materials()
            # num_randomize = np.random.randint(5)
            # for _ in range(num_randomize):
            self.stretch_controller.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.stretch_controller.controller.step(action="ResetMaterials", raise_for_failure=True)

        door_ids = [door["id"] for door in self.house["doors"]]
        self.stretch_controller.step(
            action="SetObjectFilter",
            objectIds=door_ids + ["TableToClean"],
            raise_for_failure=True,
        )

        self.reset_arm_pos()
        # NOTE: Set agent pose

        for _ in range(10):
            standing = (
                {}
                if self.args.controller_args["agentMode"] == "locobot"
                else {"standing": True}
            )
            # while True:
            starting_pose = AgentPose(
                position=random.choice(self.starting_positions),
                rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
                horizon=0,
                **standing,
            )
            self.reset_arm_pos()
            event = self.stretch_controller.step(action="TeleportFull", **starting_pose)
            self.stretch_controller.reset_obs_his()
            if not event:
                get_logger().warning(
                    # f"Teleport failing in {self.house_index} at {starting_pose}, {event.metadata['errorMessage']}"
                    f"Teleport failing in {self.house_index} at {starting_pose}"
                )
                #     continue
                # break

        self.reset_arm_pos()
        self.stretch_controller.step(
            action="ClearAllDirt"
        )
        #Spawns dirt on the specific cleaning table set up
        self.stretch_controller.step(
            action="SpawnDirt",
            objectId= "TableToClean",
            howManyDirt = self.args.extra_task_spec["init_dirt_points"], #total number of dirt spots, around 200 is pretty good as a start
            # randomSeed = 2345, #int seed to replicate dirt spawns if you need to
            randomSeed=time.time()
        )
        # event=self.stretch_controller.step(
        #     action="GetDirtMeta"
        # )
        #Turns on the sponge attachment, disabling the pen attachment and the default hand sphere
        self.stretch_controller.step(
            action="ActivateSponge"
        )
        # self.close_all_doors(door_ids)

        self.episode_index += 1
        self.args.max_tasks -= 1

        task_id = (f'cleaning_table_{target_object_type}'
            f'__proc_{str(self.args.process_ind)}'
            f'__epidx_{str(self.episode_index)}')

        self._last_sampled_task = self.cleaning_table_task_type(
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
                "init_dirt_points": self.args.extra_task_spec["init_dirt_points"],
                "success_progress": self.args.extra_task_spec["success_progress"],
                "table_reach_distance": self.args.extra_task_spec["table_reach_distance"],
            },
        )
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.args.max_tasks = self.reset_tasks
        self.house_inds_index = 0

