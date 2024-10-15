import os
import sys
import pdb
import random
import copy
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from allenact.utils.misc_utils import prepare_locals_for_super

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


class ObjectNavTask(Task[Controller]):
    def __init__(
        self,
        controller: Controller,
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
        self.controller = controller
        self.house = house
        self.reward_config = reward_config
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.mirror = task_info["mirrored"]

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List = (
            []
        )  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0

        self.task_info["followed_path"] = [
            self.controller.last_event.metadata["agent"]["position"]
        ]
        self.task_info["taken_actions"] = []
        self.task_info["action_successes"] = []
        self.task_info["rewards"] = []
        self.task_info["dist_to_target"] = []
        self.task_info["target_locations"] = []

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
            else (self.task_info["mode"] == "do_not_viz_eval" or random.random() < 1 / 50)
        )
        if self.visualize:
            for object_id in self.task_info["target_object_ids"]:
                obj_id_to_obj_pos = {
                    o["objectId"]: o["axisAlignedBoundingBox"]["center"]
                    for o in self.controller.last_event.metadata["objects"]
                }
                self.task_info["target_locations"].append(obj_id_to_obj_pos[object_id])
        
        # self.observations = [self.controller.last_event.frame]
        # TODO: hacky. Set to auto-id the camera sensor or get obs and de-normalize in a function
        self.observations = [self.sensor_suite.sensors['rgb_lowres'].frame_from_env(self.controller,self)]
        self._metrics = None

    def min_approx_geo_distance_to_target(self) -> float:
        agent_position = self.controller.last_event.metadata["agent"]["position"]
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
            controller=self.controller,
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
            for o in self.controller.last_event.metadata["objects"]
        }
        for object_id in self.task_info["target_object_ids"]:
            min_dist = min(
                min_dist,
                IThorEnvironment.position_dist(
                    obj_id_to_obj_pos[object_id],
                    self.controller.last_event.metadata["agent"]["position"],
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
                controller=self.controller,
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

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(cfg.mdp.actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cfg.mdp.actions

    def close(self) -> None:
        self.controller.stop()

    def _step(self, action: int) -> RLStepResult:
        action_str = self.class_action_names()[action]

        if self.mirror:
            if action_str == "RotateRight":
                action_str = "RotateLeft"
            elif action_str == "RotateLeft":
                action_str = "RotateRight"

        self.task_info["taken_actions"].append(action_str)

        ### for the Kiana Experiment (tm)

        # # self._success = self._is_goal_in_range()
        # self._success = self.dist_to_target_func() <= 1 and self.dist_to_target_func() > 0
        # self._took_end_action = self._success

        # if action_str == "End":
        #     # self._took_end_action = True
        #     # self._success = self._is_goal_in_range()
        #     self.last_action_success = self._success
        #     self.task_info["action_successes"].append(self._success)

        if action_str == "End":
            ### for regular objectnav
            # self._took_end_action = True
            # self._success = self._is_goal_in_range()
            # self.last_action_success = self._success
            # self.task_info["action_successes"].append(True)

            ### for exploration only
            self.last_action_success = False
            self.task_info["action_successes"].append(False)
        else:
            self.controller.step(action=action_str)
            self.last_action_success = bool(self.controller.last_event)

            position = self.controller.last_event.metadata["agent"]["position"]
            self.path.append(position)
            self.task_info["followed_path"].append(position)
            self.task_info["action_successes"].append(self.last_action_success)

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            # self.observations.append(self.controller.last_event.frame)
            # TODO: same as above
            self.observations.append(self.sensor_suite.sensors['rgb_lowres'].frame_from_env(self.controller,self))

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(
        self, mode: Literal["rgb", "depth"] = "rgb", *args, **kwargs
    ) -> np.ndarray:
        if mode == "rgb":
            frame = self.controller.last_event.frame.copy()
        elif mode == "depth":
            frame = self.controller.last_event.depth_frame.copy()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

        if self.mirror:
            frame = np.fliplr(frame)

        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            obj
            for obj in self.controller.last_event.metadata["objects"]
            if obj["visible"] and obj["objectType"] == self.task_info["object_type"]
        )

    def shaping(self) -> float:
        cur_distance = self.dist_to_target_func()
        self.task_info["dist_to_target"].append(cur_distance)
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        cur_distance = self.dist_to_target_func()

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

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_config.step_penalty

        if not self.last_action_success: # and "Look" not in self.task_info["taken_actions"][-1]:
            reward += self.reward_config.failed_action_penalty

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config.goal_success_reward
            else:
                reward += self.reward_config.failed_stop_reward
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
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
        metrics["dist_to_target"] = self.dist_to_target_func()
        metrics["total_reward"] = np.sum(self._rewards)
        metrics["spl"] = spl_metric(
            success=self._success,
            optimal_distance=self.optimal_distance,
            travelled_distance=self.travelled_distance,
        )
        metrics["success"] = self._success

        self._metrics = metrics
        return metrics


class ObjectNavToInstanceTask(ObjectNavTask):
    def __init__(
        self,
        controller: Controller,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_config: RewardConfig,
        distance_cache: DynamicDistanceCache,
        distance_type: str = "geo",
        visualize: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))
        self.distance_cache = None
        assert self.distance_type == "l2"

        self.plausible_target_ids = set(self.task_info["target_object_ids"])
        self.seen_plausible_target_ids = set()
        self.task_info["target_object_ids"] = [
            random.choice(self.task_info["target_object_ids"])
        ]
        self.target_object_id = self.task_info["target_object_ids"][0]

        self._num_newly_seen_targets = 0

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def _step(self, action: int) -> RLStepResult:
        action_str = self.class_action_names()[action]

        if self.mirror:
            if action_str == "RotateRight":
                action_str = "RotateLeft"
            elif action_str == "RotateLeft":
                action_str = "RotateRight"

        self.task_info["taken_actions"].append(action_str)

        self._num_newly_seen_targets = 0
        if action_str == "End":
            visible_object_ids = set(
                o["objectId"]
                for o in self.controller.last_event.metadata["objects"]
                if o["visible"]
            )
            self._took_end_action = (
                self.target_object_id in visible_object_ids
                or len(visible_object_ids & self.plausible_target_ids) == 0
            )
            self._num_newly_seen_targets = len(
                (visible_object_ids & self.plausible_target_ids)
                - self.seen_plausible_target_ids
            )
            self.seen_plausible_target_ids |= (
                visible_object_ids & self.plausible_target_ids
            )
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
            self.task_info["action_successes"].append(True)
        else:
            self.controller.step(action=action_str)
            self.last_action_success = bool(self.controller.last_event)

            position = self.controller.last_event.metadata["agent"]["position"]
            self.path.append(position)
            
            pose = copy.deepcopy(
                self.controller.last_event.metadata["agent"]["position"]
            )
            pose["rotation"] = self.controller.last_event.metadata["agent"]["rotation"][
                "y"
            ]
            pose["horizon"] = self.controller.last_event.metadata["agent"][
                "cameraHorizon"
            ]
            self.task_info["followed_path"].append(pose)

            self.task_info["action_successes"].append(self.last_action_success)

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            self.observations.append(self.controller.last_event.frame)

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def _is_goal_in_range(self) -> bool:
        return any(
            obj
            for obj in self.controller.last_event.metadata["objects"]
            if obj["visible"] and obj["objectId"] == self.target_object_id
        )

    def _unseen_target_in_range(self) -> bool:
        return (
            len(
                set(
                    obj["objectId"]
                    for obj in self.controller.last_event.metadata["objects"]
                    if obj["visible"] and obj["objectId"] in self.plausible_target_ids
                )
                - self.seen_plausible_target_ids
            )
            != 0
        )

    def shaping(self) -> float:
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        cur_distance = self.dist_to_target_func()

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

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_config.step_penalty

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config.goal_success_reward
            else:
                reward += self.reward_config.failed_stop_reward
        elif self._num_newly_seen_targets != 0:
            reward += (
                self.reward_config.goal_success_reward
                # * self._num_newly_seen_targets
            )
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        metrics["seen_plausible_targets"] = len(self.seen_plausible_target_ids)
        metrics["saw_other_target_before_success"] = 1 * (
            metrics["success"] and len(self.seen_plausible_target_ids) > 1
        )
        metrics["found_object_of_type"] = 1 * (len(self.seen_plausible_target_ids) > 0)
        return metrics


class ObjectNavToAllInstancesTask(ObjectNavTask):
    def __init__(
        self,
        controller: Controller,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_config: RewardConfig,
        distance_cache: DynamicDistanceCache,
        distance_type: str = "geo",
        visualize: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))
        self.distance_cache = None
        assert self.distance_type == "l2"

        self.plausible_target_ids = set(self.task_info["target_object_ids"])
        self.seen_plausible_target_ids = set()

        self._num_newly_seen_targets = 0

        self.object_id_to_closest_distance = self.l2_distance_to_remaining_targets()

    def min_l2_distance_to_target(self) -> float:
        # TODO: VERY HACKY, making this into a max L2 function across targets
        #   as this is what should be used for this task
        dists = list(self.l2_distance_to_remaining_targets().values())
        if len(dists) == 0:
            return 0.0
        return max(dists)

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def _step(self, action: int) -> RLStepResult:
        action_str = self.class_action_names()[action]

        if self.mirror:
            if action_str == "RotateRight":
                action_str = "RotateLeft"
            elif action_str == "RotateLeft":
                action_str = "RotateRight"

        self.task_info["taken_actions"].append(action_str)

        self._num_newly_seen_targets = 0
        if action_str == "End":
            visible_object_ids = set(
                o["objectId"]
                for o in self.controller.last_event.metadata["objects"]
                if o["visible"]
            )
            self._num_newly_seen_targets = len(
                (visible_object_ids & self.plausible_target_ids)
                - self.seen_plausible_target_ids
            )
            self.seen_plausible_target_ids |= (
                visible_object_ids & self.plausible_target_ids
            )

            self.task_info["target_object_ids"] = list(
                self.plausible_target_ids - self.seen_plausible_target_ids
            )
            self._success = len(self.task_info["target_object_ids"]) == 0
            self._took_end_action = (
                self._success
                or len(visible_object_ids & self.plausible_target_ids) == 0
            )

            self.last_action_success = self._success
            self.task_info["action_successes"].append(True)
        else:
            self.controller.step(action=action_str)
            self.last_action_success = bool(self.controller.last_event)

            position = self.controller.last_event.metadata["agent"]["position"]
            self.path.append(position)
            self.task_info["followed_path"].append(position)
            self.task_info["action_successes"].append(self.last_action_success)

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            self.observations.append(self.controller.last_event.frame)

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def _is_goal_in_range(self) -> bool:
        raise NotImplementedError

    def _unseen_target_in_range(self) -> bool:
        return (
            len(
                set(
                    obj["objectId"]
                    for obj in self.controller.last_event.metadata["objects"]
                    if obj["visible"] and obj["objectId"] in self.plausible_target_ids
                )
                - self.seen_plausible_target_ids
            )
            != 0
        )

    def l2_distance_to_remaining_targets(self) -> Dict[str, float]:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        obj_id_to_obj_pos = {
            o["objectId"]: o["axisAlignedBoundingBox"]["center"]
            for o in self.controller.last_event.metadata["objects"]
        }
        object_id_to_dist = {}
        for object_id in self.task_info["target_object_ids"]:
            object_id_to_dist[object_id] = IThorEnvironment.position_dist(
                obj_id_to_obj_pos[object_id],
                self.controller.last_event.metadata["agent"]["position"],
            )
        return object_id_to_dist

    def shaping(self) -> float:
        assert self.distance_type == "l2"

        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        cur_dists = self.l2_distance_to_remaining_targets()
        if self._num_newly_seen_targets == 0:
            assert len(cur_dists) != 0
            reward = max(
                max(self.object_id_to_closest_distance[oid] - cur_dists[oid], 0)
                for oid in self.object_id_to_closest_distance
            )
            reward = reward * self.reward_config.shaping_weight

        try:
            self.object_id_to_closest_distance = {
                oid: min(self.object_id_to_closest_distance[oid], cur_dists[oid])
                for oid in cur_dists  # Must be `for oid in cur_dists` as we want to throw away oid's that may have been seen
            }
        except:
            print(
                self.object_id_to_closest_distance,
                cur_dists,
                self.seen_plausible_target_ids,
                self.plausible_target_ids,
                self.task_info["target_object_ids"],
            )
        self.last_distance = None

        return reward

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_config.step_penalty

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config.goal_success_reward
            else:
                reward += self.reward_config.failed_stop_reward
        elif self._num_newly_seen_targets != 0:
            reward += (
                self.reward_config.goal_success_reward
                # * self._num_newly_seen_targets
            )
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        metrics["seen_plausible_targets"] = len(self.seen_plausible_target_ids)
        metrics["prop_targets_seen"] = len(self.seen_plausible_target_ids) / len(
            self.plausible_target_ids
        )
        metrics["saw_other_target_before_success"] = 1 * (
            metrics["success"] and len(self.seen_plausible_target_ids) > 1
        )
        metrics["found_object_of_type"] = 1 * (len(self.seen_plausible_target_ids) > 0)
        return metrics

class ObjectNavRealTask(ObjectNavTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        self.directory = f'output/saved_images/{time_now}'
        os.makedirs(self.directory, exist_ok=True)
        self.bonks = None
        self.visualize_every_frame = False
    
    def _is_goal_in_range(self) -> bool:
        success = False
        print('I think I found a(n)', self.task_info['object_type'], '. Was I correct? Set success and self.bonks in trace (default false).' )
        ForkedPdb().set_trace()
        return success
    
    def _step(self, action: int) -> RLStepResult:
        end_early = False

        if self.visualize_every_frame:
            last_frame = self.controller.last_event.frame
            path2img = os.path.join(self.directory, datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '.png')
            import matplotlib.pyplot as plt; plt.imsave(path2img, last_frame)

        if self.num_steps_taken() == self.max_steps - 1:
            print('Reached horizon. Did I run into things a lot? Set self.bonks in trace (default zero).' )
            ForkedPdb().set_trace()
        
        if self.num_steps_taken() % 50 == 0 and self.num_steps_taken() > 0:
            print('Have I run into things more than 10 times? Set end_early=True and self.bonks in trace (default zero).' )
            ForkedPdb().set_trace()

        sr = super()._step(action)

        if end_early:
            self._took_end_action = True
            print('Ending early for too many environment run-ins')

        self.task_info["number_of_interventions"] = self.bonks
        print(str(self.num_steps_taken()))
        # cv2.imshow('current locobot', self.env.last_event.cv2img)
        # cv2.waitKey(1)
        return sr
    
    def judge(self) -> float:
        reward = 0.0
        self._rewards.append(reward)
        cur_distance = self.dist_to_target_func()
        self.task_info["dist_to_target"].append(cur_distance)
        self.task_info["rewards"].append(float(reward))
        return reward
    
    # possibly also metrics

class ObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        args: TaskSamplerArgs,
        object_nav_task_type: Type = ObjectNavTask,
    ) -> None:
        self.args = args
        self.object_nav_task_type = object_nav_task_type
        random.shuffle(self.args.house_inds)

        self.controller: Optional[Controller] = None
        self.distance_cache = DynamicDistanceCache(rounding=1)

        # get the total number of tasks assigned to this process
        self.reset_tasks = self.args.max_tasks

        self.resample_same_scene_freq = args.resample_same_scene_freq
        """The number of times to resample the same houses before moving to the next on."""

        self.house_inds_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[ObjectNavTask] = None

        self.reachable_positions_map: Dict[int, Vector3] = dict()
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
    def last_sampled_task(self) -> Optional[ObjectNavTask]:
        # NOTE: This book-keeping should be done in TaskSampler...
        return self._last_sampled_task

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()

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
        visibility_points = self.controller.step(
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
                event = self.controller.step(
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

        event = self.controller.step(action="ResetObjectFilter", raise_for_failure=True)
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

    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.increment_scene_index()

        # self.controller.step(action="DestroyHouse", raise_for_failure=True)
        self.controller.reset()
        self.house = self.args.houses[self.house_index]

        self.controller.step(
            action="CreateHouse", house=self.house, raise_for_failure=True
        )

        # NOTE: Set reachable positions
        if self.house_index not in self.reachable_positions_map:
            pose = self.house["metadata"]["agent"].copy()
            if self.args.controller_args["agentMode"] == "locobot":
                del pose["standing"]
            event = self.controller.step(action="TeleportFull", **pose)
            if not event:
                get_logger().warning(f"Initial teleport failing in {self.house_index}.")
                return False
            rp_event = self.controller.step(action="GetReachablePositions")
            if not rp_event:
                # NOTE: Skip scenes where GetReachablePositions fails
                get_logger().warning(
                    f"GetReachablePositions failed in {self.house_index}"
                )
                return False
            reachable_positions = rp_event.metadata["actionReturn"]
            self.reachable_positions_map[self.house_index] = reachable_positions
        return True

    def increment_scene_index(self):
        self.house_inds_index = (self.house_inds_index + 1) % len(self.args.house_inds)

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

        # NOTE: Setup the Controller
        if self.controller is None:
            self.controller = Controller(**self.args.controller_args)
            get_logger().info(
                f"Using Controller commit id: {self.controller._build.commit_id}"
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

        # NOTE: Choose target object
        while True:
            try:
                # NOTE: The loop avoid a very rare edge case where the agent
                # starts out trapped in some part of the room.
                target_object_type, target_object_ids = self.sample_target_object_ids()
                break
            except ValueError:
                while not self.increment_scene():
                    pass

        if random.random() < cfg.procthor.p_randomize_materials:
            self.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.controller.step(action="ResetMaterials", raise_for_failure=True)

        door_ids = [door["id"] for door in self.house["doors"]]
        self.controller.step(
            action="SetObjectFilter",
            objectIds=target_object_ids + door_ids,
            raise_for_failure=True,
        )

        # NOTE: Set agent pose
        standing = (
            {}
            if self.args.controller_args["agentMode"] == "locobot"
            else {"standing": True}
        )
        starting_pose = AgentPose(
            position=random.choice(self.reachable_positions),
            rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
            horizon=30,
            **standing,
        )
        event = self.controller.step(action="TeleportFull", **starting_pose)
        if not event:
            get_logger().warning(
                f"Teleport failing in {self.house_index} at {starting_pose}"
            )

        self.episode_index += 1
        self.args.max_tasks -= 1

        task_id = (f'object_nav_{target_object_type}'
            f'__proc_{str(self.args.process_ind)}'
            f'__epidx_{str(self.episode_index)}')

        self._last_sampled_task = self.object_nav_task_type(
            controller=self.controller,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type="l2",
            house=self.house,
            distance_cache=self.distance_cache,
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
            },
        )
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.args.max_tasks = self.reset_tasks
        self.house_inds_index = 0


class FullObjectNavTestTaskSampler(ObjectNavTaskSampler):
    """Works with PRIOR's object-nav-eval tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=min(self.reset_tasks // 10, 4), dtype=np.uint8
            ).tolist()
        )
        self.args.controller_args = self.args.controller_args.copy()
        self.houses = prior.load_dataset("procthor-10k")[
            "val" if cfg.evaluation.test_on_validation or not cfg.eval else "test"
        ]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_scene is None or self.last_scene != ep["scene"]:
                self.last_scene = ep["scene"]
                self.controller.reset(
                    scene=(
                        self.houses[ep["scene"]]
                        if ep["sceneDataset"] == "procthor-10k"
                        else ep["scene"]
                    )
                )

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                task_info={
                    "mode": self.args.mode,
                    "house_name": str(ep["scene"]),
                    "sceneDataset": ep["sceneDataset"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task


class ObjectNavTestTaskSampler(ObjectNavTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=min(self.reset_tasks // 10, 4), dtype=np.uint8
            ).tolist()
        )
        self.args.controller_args = self.args.controller_args.copy()

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_scene is None or self.last_scene != ep["scene"]:
                self.last_scene = ep["scene"]
                self.controller.reset(ep["scene"])

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                task_info={
                    "mode": self.args.mode,
                    "house_name": ep["scene"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task

class RealObjectNavTaskSampler(ObjectNavTaskSampler):
    def __init__(self, args: TaskSamplerArgs, object_nav_task_type: Type = ObjectNavTask) -> None:
        super().__init__(args, object_nav_task_type)
        assert cfg.eval

        self.num_starting_positions = 3

        self.real_object_index = 0
        self.starting_position_index = 0
        self.host = cfg.real.host 
        get_logger().debug('Initializing on host '+ self.host+ '. Verify that is the correct platform before continuing.')
        ForkedPdb().set_trace()
        self.args.controller_args = {"host":self.host, #+".corp.ai2"
                                    "port":9000, "width":1280, "height":720}
        self.object_set = ['Sofa', 'Bed', 'Television', 'Vase', 'Apple'] # base set - robothor and # self.args.target_object_types
        # self.object_set = ['Television', 'Vase', 'Apple', 'Chair', 'GarbageCan'] # for the office
        # self.object_set = ['Sofa', 'Bed', 'Television'] # for habitat
        # self.object_set = ['Sofa', 'Chair', 'Television', 'Vase', 'Apple']
        self.args.max_tasks = len(self.object_set) * self.num_starting_positions
        
    def next_task(self) -> Optional[ObjectNavRealTask]:
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
        if self.controller is None:
            self.controller = ai2thor.robot_controller.Controller(**self.args.controller_args)
            self.controller.step({"action": "Pass"})
        
        # normalize the camera horizon at 0
        horiz = self.controller.last_event.metadata["agent"]["cameraHorizon"]
        while abs(horiz) > 1: # TODO make this better, it might get weird edge cases
            if horiz > 0:
                self.controller.step({"action": "LookUp"})
            else:
                self.controller.step({"action": "LookDown"})
            horiz = self.controller.last_event.metadata["agent"]["cameraHorizon"] 

        # # start at 30 below for not fine-tune. Phone2Proc fine tune starts at 0
        if cfg.real.initial_horizon == 30:
            self.controller.step({"action": "LookDown"})
        
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
            self.controller.step({"action": "Pass"})
            
            task_start = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            task_id = (f'realRobothor__{target_object["object_type"]}'
                        f'__starting_position_{str(self.starting_position_index+1)}'
                        f'__model_{eval_checkpoint_name}'
                        f'__task_start_{task_start}')

            self.args.max_tasks -= 1
            self.episode_index += 1
            
            self._last_sampled_task = self.object_nav_task_type(
                visualize=True,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type="realWorld",
                distance_cache=self.distance_cache,
                task_info={
                    "mode": self.args.mode,
                    "house_name": 'Real Robothor',
                    "target_object_ids": [target_object['object_id']],
                    "object_type": target_object['object_type'],
                    "starting_pose": {},
                    "mirrored": False,
                    "id": task_id,
                },
            )
            self.real_object_index += 1
            if self.real_object_index == len(self.object_set):
                self.real_object_index = 0
                self.starting_position_index +=1 

            return self._last_sampled_task
        
class SimOfCurrentRealObjectNavTaskSampler(ObjectNavTaskSampler):
    def __init__(self, args: TaskSamplerArgs, object_nav_task_type: Type = ObjectNavTask) -> None:
        super().__init__(args, object_nav_task_type)

        self.args.controller_args["branch"] = "test-dev-variants" #test-dev-variants nanna

        # dark one
        # del self.args.controller_args["branch"] # branch supercedes commit id argument
        # self.args.controller_args["commit_id"] = '47b45bbfd14e9cef767b3ff7d9056ddd52f69ab8'

        # infinite wiggle one
        # del self.args.controller_args["branch"] # branch supercedes commit id argument
        # self.args.controller_args["commit_id"] = 'd0fd55829ad0c30361b33c3019e1cfab75eefde2'

        del self.args.controller_args["scene"]
        self.scene_index = 0 #'FloorPlan_test-dev2_2' #'FloorPlan_RoboTHOR_Real' 
        base_scene = 'FloorPlan_test-dev2_2'
        self.scene_list = [base_scene]
        self.scene_list.extend([base_scene+'_'+str(x+1) for x in range(30) if (x not in []) ])
        self.scene_list = self.scene_list * 7 # do each twenty times
        
        if cfg.phone2proc.stochastic_controller:
            self.controller_type = StochasticController
        else:
            self.controller_type = Controller

        self.possible_initial_horizon = [0]
        self.potential_camera_fovs = [self.args.controller_args["fieldOfView"]]

        if cfg.phone2proc.stochastic_camera_params:
            self.potential_camera_fovs = [i for i in np.arange(48, 65, 0.2)]
            self.possible_initial_horizon = [i for i in np.arange(-5, 5, 0.2)]
        
        self.sampler_start = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        
        ## specific poses and objects machinery
        # facing_bed_pose = {'position': {'x': 4.5, 'y': 0.9009996652603149, 'z': -1.25}, 'rotation': {'x': 0, 'y': 180, 'z': 0}, 'horizon': 30}
        if self.args.mode == "eval":
            couch_pose = {'position': {'x': 2.614, 'y': 0.9009996652603149, 'z': -3.35}, 'rotation': {'x': 0, 'y': 0, 'z': 0}, 
                            'horizon': 30+random.choice(self.possible_initial_horizon)}
            living_room_pose = {'position': {'x': 9.2404, 'y': 0.9009996652603149, 'z': -2.5}, 'rotation': {'x': 0, 'y': 0, 'z': 0}, 
                            'horizon': 30+random.choice(self.possible_initial_horizon)}        
            bed_pose = {'position': {'x': 3.68, 'y': 0.9009996652603149, 'z': -1.506}, 'rotation': {'x': 0, 'y': 270, 'z': 0}, 
                            'horizon': 30+random.choice(self.possible_initial_horizon)}

            self.target_object_types = ['Sofa', 'Bed', 'Television', 'Vase', 'Apple'] 
            # self.target_object_types = ['Bed', 'Television','Sofa'] # Habitat ones
            self.num_starting_positions = 3
            self.real_object_index = 0
            self.starting_position_index = 0
            self.starting_positions = [couch_pose, living_room_pose, bed_pose]
            self.args.max_tasks = len(self.target_object_types) * self.num_starting_positions * len(self.scene_list) #15

    
    # def is_object_visible(self,object_id):
    #     return True # visibility points incompatible with commit id? didn't work. irrelevant anyway
    
    def reset(self):
        super().reset()
        if self.args.mode == "eval":
            self.real_object_index = 0
            self.starting_position_index = 0
            self.scene_index=0
            self.args.max_tasks = 15#len(self.target_object_types) * self.num_starting_positions * len(self.scene_list) #15
    
    def increment_scene(self) -> bool:
        self.starting_position_index = 0
        self.scene_index += 1
        self.controller.reset(scene=self.scene_list[self.scene_index])
        # self.objects_in_scene_map = [] # reset this easier
        return True
    
    @property
    def target_objects_in_scene(self) -> Dict[str, List[str]]:
        """Return a map from the object type to the objectIds in the scene."""
        if self.scene_index in self.objects_in_scene_map:
            return self.objects_in_scene_map[self.scene_index]

        event = self.controller.step(action="ResetObjectFilter", raise_for_failure=True)
        objects = event.metadata["objects"]
        out = {}
        for obj in objects:
            if obj["objectType"] in self.target_object_types_set:
                if obj["objectType"] not in out:
                    out[obj["objectType"]] = []
                out[obj["objectType"]].append(obj["objectId"])
        self.objects_in_scene_map[self.scene_index] = out
        return out

        
    def next_task(self, force_advance_scene: bool = False):
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None
        
        if self.starting_position_index == len(self.starting_positions):
            while not self.increment_scene():
                pass

        # NOTE: Setup the Controller
        if self.controller is None:
            self.controller = self.controller_type(**self.args.controller_args)
            self.controller.reset(scene=self.scene_list[self.scene_index])
            rp_event = self.controller.step(action="GetReachablePositions")
            if not rp_event:
                # NOTE: Skip scenes where GetReachablePositions fails
                get_logger().warning(
                    f"GetReachablePositions failed in sim of current real, something is wrong"
                )
                return False
            reachable_positions = rp_event.metadata["actionReturn"]
            self.reachable_positions_map[self.house_index] = reachable_positions
        
        self.controller.reset(scene=self.scene_list[self.scene_index])
        target_object_type, target_object_ids = self.sample_target_object_ids()

        event = None
        attempts = 0
        while not event:
            attempts+=1
            starting_pose = AgentPose(
                position=random.choice(self.reachable_positions),
                rotation=Vector3(x=0, y=random.choice([i for i in range(0,360,30)]), z=0),
                horizon=random.choice(self.possible_initial_horizon),
            )
            event = self.controller.step(action="TeleportFull", **starting_pose)
            self.controller.nominal_neck = 0
            if attempts > 10:
                get_logger().error(f"Teleport failed {attempts-1} times in {self.scene_list[self.scene_index]} - something may be wrong")
        
        self.controller.step(action="LookDown") # start looking down

        if self.args.mode == "eval":
            target_object_type = self.target_object_types[self.real_object_index]
            target_object_ids = self.target_objects_in_scene.get(target_object_type, [])
            self.controller.reset(scene=self.scene_list[self.scene_index])
            
            event = self.controller.step(action="TeleportFull", **self.starting_positions[self.starting_position_index])
            self.controller.nominal_neck = -1
            # self.controller.step(action="LookUp") # HABITAT ONLY
            if not event:
                print(f'Specified start failed in {self.scene_list[self.scene_index]}')

        self.episode_index += 1
        self.args.max_tasks -= 1

        # identify which model is being evaluated
        eval_checkpoint_name = "evaluation_model_not_specified"
        if cfg.checkpoint is None and cfg.pretrained_model.name is not None:
            eval_checkpoint_name = cfg.pretrained_model.name
        elif cfg.checkpoint is not None:
            eval_checkpoint_name = cfg.checkpoint.split('/')[-1]
        eval_checkpoint_name = eval_checkpoint_name.split('.')[0]
    
        if self.args.mode == "eval":
            task_id = (f'sim_of_current_real_{target_object_type}'
                f'__model_{eval_checkpoint_name}'
                f'__starting_position_{str(self.starting_position_index+1)}'
                f'__sampler_start_{self.sampler_start}'
                f'__epidx_{str(self.episode_index)}')
        else:
            task_id = (f'sim_of_current_real_{target_object_type}'
                f'__model_{eval_checkpoint_name}'
                f'__sampler_start_{self.sampler_start}'
                f'__epidx_{str(self.episode_index)}')

        self._last_sampled_task = self.object_nav_task_type(
            controller=self.controller,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type="l2",
            house={},#self.house, # NOTE does not work with geo distance 
            distance_cache=self.distance_cache,
            task_info={
                "mode": self.args.mode,
                "process_ind": self.args.process_ind,
                "house_name": self.scene_list[self.scene_index],
                "rooms": 1,#len(self.house["rooms"]),
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
                "id": task_id,
            },
        )
        if self.args.mode == "eval":
            self.real_object_index += 1
            if self.real_object_index == len(self.target_object_types):
                self.real_object_index = 0
                self.starting_position_index +=1 

        return self._last_sampled_task
    
class ObjectNavTaskSamplerProcthorFineTune(ObjectNavTaskSampler):
    def __init__(self, args: TaskSamplerArgs, object_nav_task_type: Type = ObjectNavTask) -> None:
        super().__init__(args, object_nav_task_type)
        if cfg.phone2proc.stochastic_controller:
            self.controller_type = StochasticController
        else:
            self.controller_type = Controller

        self.possible_initial_horizon = [0]
        self.potential_camera_fovs = [self.args.controller_args["fieldOfView"]]

        if cfg.phone2proc.stochastic_camera_params:
            self.potential_camera_fovs = [i for i in np.arange(48, 65, 0.2)]
            self.possible_initial_horizon = [i for i in np.arange(-5, 5, 0.2)]

    def increment_scene(self) -> bool: 
        # redefined just for field of view
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.increment_scene_index()

        # self.controller.step(action="DestroyHouse", raise_for_failure=True)
        self.controller.reset(fieldOfView=random.choice(self.potential_camera_fovs))
        self.house = self.args.houses[self.house_index]

        self.controller.step(
            action="CreateHouse", house=self.house, raise_for_failure=True
        )

        # NOTE: Set reachable positions
        if self.house_index not in self.reachable_positions_map:
            pose = self.house["metadata"]["agent"].copy()
            if self.args.controller_args["agentMode"] == "locobot":
                del pose["standing"]
            event = self.controller.step(action="TeleportFull", **pose)
            if not event:
                get_logger().warning(f"Initial teleport failing in {self.house_index}.")
                return False
            rp_event = self.controller.step(action="GetReachablePositions")
            if not rp_event:
                # NOTE: Skip scenes where GetReachablePositions fails
                get_logger().warning(
                    f"GetReachablePositions failed in {self.house_index}"
                )
                return False
            reachable_positions = rp_event.metadata["actionReturn"]
            self.reachable_positions_map[self.house_index] = reachable_positions
        return True
    
    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        # redefined to set neck action limiting/neck noise
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

        # NOTE: Setup the Controller
        if self.controller is None:
            self.controller = self.controller_type(**self.args.controller_args)
            get_logger().info(
                f"Using Controller commit id: {self.controller._build.commit_id}"
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

        # NOTE: Choose target object
        while True:
            try:
                target_object_type, target_object_ids = self.sample_target_object_ids()
                break
            except ValueError:
                while not self.increment_scene():
                    pass

        if random.random() < cfg.procthor.p_randomize_materials:
            self.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.controller.step(action="ResetMaterials", raise_for_failure=True)

        door_ids = [door["id"] for door in self.house["doors"]]
        self.controller.step(
            action="SetObjectFilter",
            objectIds=target_object_ids + door_ids,
            raise_for_failure=True,
        )

        # Teleport to a random reachable position and location
        event = None
        attempts = 0
        while not event:
            attempts+=1
            starting_pose = AgentPose(
                position=random.choice(self.reachable_positions),
                rotation=Vector3(x=0, y=random.choice([i for i in range(0,360,30)]), z=0),
                horizon=30+random.choice(self.possible_initial_horizon),
            )
            event = self.controller.step(action="TeleportFull", **starting_pose)
            self.controller.nominal_neck = -1
            if attempts > 10:
                get_logger().error(f"Teleport failed {attempts-1} times in house {self.house_index} - something may be wrong")
                continue
            

        self.episode_index += 1
        self.args.max_tasks -= 1

        self._last_sampled_task = self.object_nav_task_type(
            controller=self.controller,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type="l2",
            house=self.house,
            distance_cache=self.distance_cache,
            task_info={
                "mode": self.args.mode,
                "process_ind": self.args.process_ind,
                "house_name": str(self.house_index),
                "rooms": len(self.house["rooms"]),
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "id": f"house{self.house_inds_index}__proc{self.args.process_ind}__global{str(self.episode_index-1)}__{target_object_type}",
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
            },
        )
        return self._last_sampled_task
    


class Phone2ProcObjectNavTaskSampler(ObjectNavTaskSampler):
    def __init__(self, args: TaskSamplerArgs, object_nav_task_type: Type = ObjectNavTask) -> None:
        super().__init__(args, object_nav_task_type)
        # self.args.controller_args["branch"] = "main" # rate limit? wtf
        self.bad_house_indices = [] # 461 and 467
        if cfg.phone2proc.stochastic_controller:
            self.controller_type = StochasticController
        else:
            self.controller_type = Controller

        self.possible_initial_horizon = [0]
        self.potential_camera_fovs = [self.args.controller_args["fieldOfView"]]

        if cfg.phone2proc.stochastic_camera_params:
            self.potential_camera_fovs = [i for i in np.arange(48, 65, 0.2)]
            self.possible_initial_horizon = [i for i in np.arange(-5, 5, 0.2)]

        if len(self.args.houses[0]['rooms']) == 7: # heuristic, not arbitrarily scalable. would be better to have in identifier in general metadata
            self.usable_starting_positions = BACK_APARTMENT_USABLE_POSITIONS
        elif len(self.args.houses[0]['rooms']) == 3:
            self.usable_starting_positions = KIANAS_USABLE_POSITIONS
        else:
            self.usable_starting_positions = ROBOTHOR_USABLE_POSITIONS

    def randomize_lighting_and_materials(self):

        randomize_wall_and_floor_materials(self.house)
        randomize_lighting(self.house)
        self.controller.reset(scene=self.house)
        return True

    # def is_object_visible(self, object_id: str) -> bool:
    #     return True

    def sample_target_object_ids(self) -> Tuple[str, List[str]]:
        """Sample target objects.

        Objects returned will all be of the same objectType. Only considers visible
        objects in the house.
        """
        if cfg.phone2proc.chosen_objects is not None:
            candidates = dict()
            for obj_type in cfg.phone2proc.chosen_objects:
                instances_of_type = self.target_objects_in_scene.get(obj_type, [])
                if not instances_of_type:
                    continue
                visible_ids = []
                for object_id in instances_of_type:
                    if self.is_object_visible(object_id=object_id):
                        visible_ids.append(object_id)
                if visible_ids:
                    candidates[obj_type] = visible_ids
                        
            if candidates:
                return random.choice(list(candidates.items()))
            else:
                objs = " ".join(map(str,cfg.phone2proc.chosen_objects))
                # get_logger().error(f"No visible objects of types {objs} available in house {self.house_index}")
                raise Exception            
        
        rare_objects = []
        # excluded_objects = ["Bed","Chair","Sofa","Television","Toilet"]
        # excluded_objects = [obj for obj in self.target_object_types_set if obj not in ["Apple","Vase"]]
        if random.random() < cfg.training.object_selection.p_greedy_target_object:
            for obj_type, count in reversed(self.obj_type_counter.most_common()):
                instances_of_type = self.target_objects_in_scene.get(obj_type, [])

                # NOTE: object type doesn't appear in the scene.
                if not instances_of_type or (obj_type in rare_objects and random.random()<2):
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
                if obj_type in rare_objects and random.random()<2:
                    continue
                visible_ids = []
                for object_id in object_ids:
                    if self.is_object_visible(object_id=object_id):
                        visible_ids.append(object_id)

                if visible_ids:
                    candidates[obj_type] = visible_ids

            if candidates:
                return random.choice(list(candidates.items()))


    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.increment_scene_index()

        self.house = self.args.houses[self.house_index]
        self.controller.reset(scene=self.house, fieldOfView=random.choice(self.potential_camera_fovs))

        # NOTE: Teleport into house and set reachable positions
        if self.house_index not in self.reachable_positions_map:
            event = None
            attempts = 0
            while not event:
                attempts+=1
                pose = random.choice(self.usable_starting_positions)
                event = self.controller.step(action="TeleportFull", **pose)    
                if attempts > 10:
                    get_logger().error(f"Initial teleport failed {attempts-1} times in house {self.house_index}")
                    return False

            rp_event = self.controller.step(action="GetReachablePositions")
            if not rp_event:
                # NOTE: Skip scenes where GetReachablePositions fails
                get_logger().warning(
                    f"GetReachablePositions failed in {self.house_index}"
                )
                return False
            reachable_positions = rp_event.metadata["actionReturn"]
            self.reachable_positions_map[self.house_index] = reachable_positions
        return True
        
    def next_task(self, force_advance_scene: bool = False):
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

        # NOTE: Setup the Controller
        if self.controller is None:
            self.controller = self.controller_type(**self.args.controller_args)
            get_logger().info(
                f"Using Controller commit id: {self.controller._build.commit_id}"
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
        
        # NOTE: Choose target object
        while True:
            try:
                # NOTE: The loop avoid a very rare edge case where the agent
                # starts out trapped in some part of the room.
                target_object_type, target_object_ids = self.sample_target_object_ids()
                break
            except:
                while not self.increment_scene():
                    pass
            
        if random.random() < cfg.procthor.p_randomize_materials:
            self.randomize_lighting_and_materials()
            self.controller.step(action="RandomizeMaterials")
        else:
            self.controller.step(action="ResetMaterials")
        
        if cfg.mdp.reward.train.new_object_reward == 0:
            door_ids = [door["id"] for door in self.house["doors"]]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids + door_ids,
                raise_for_failure=True,
            )

        # Teleport to a random reachable position and location
        event = None
        attempts = 0
        while not event:
            attempts+=1
            starting_pose = AgentPose(
                position=random.choice(self.reachable_positions),
                rotation=Vector3(x=0, y=random.choice([i for i in range(0,360,30)]), z=0),
                horizon=30+random.choice(self.possible_initial_horizon),
            )
            event = self.controller.step(action="TeleportFull", **starting_pose)
            self.controller.nominal_neck = -1
            if attempts > 10:
                get_logger().error(f"Teleport failed {attempts-1} times in house {self.house_index} - something may be wrong")
            
        self.episode_index += 1
        self.args.max_tasks -= 1

        # identify which model is being evaluated
        eval_checkpoint_name = "evaluation_model_not_specified"
        if cfg.checkpoint is None and cfg.pretrained_model.name is not None:
            eval_checkpoint_name = cfg.pretrained_model.name
        elif cfg.checkpoint is not None:
            eval_checkpoint_name = cfg.checkpoint.split('/')[-1]
        eval_checkpoint_name = eval_checkpoint_name.split('.')[0]
        
        task_id = (f'phone_to_proc_{target_object_type}'
            f'__model_{eval_checkpoint_name}'
            f'__proc_{str(self.args.process_ind)}'
            f'__epidx_{str(self.episode_index)}')

        self._last_sampled_task = self.object_nav_task_type(
            controller=self.controller,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type="l2",
            house=self.house,#self.house, # NOTE does not work with geo distance 
            distance_cache=self.distance_cache,
            task_info={
                "mode": self.args.mode,
                "process_ind": self.args.process_ind,
                "house_name": "phone_to_proc",
                "rooms": 1,#len(self.house["rooms"]),
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
                "id": task_id,
            },
        )
        return self._last_sampled_task

class Phone2ProcValidTaskSampler(Phone2ProcObjectNavTaskSampler):
    def __init__(self, args: TaskSamplerArgs, object_nav_task_type: Type = ObjectNavTask) -> None:
        super().__init__(args, object_nav_task_type)

        self.target_object_types = ['AlarmClock', 'Bed', 'Television', 'Vase', 'Apple']
        self.usable_starting_positions = self.usable_starting_positions # enforce only use the first 3
        self.real_object_index = 0
        self.starting_position_index = 0

        assert self.args.mode == "eval"
    
    def increment_scene_index(self):
        self.house_inds_index = random.choice(range(len(self.args.house_inds))) #(self.house_inds_index + 1) % len(self.args.house_inds)
    
    def increment_scene(self) -> bool:
        internal_scene_success = super().increment_scene()
        if not internal_scene_success:
            return False
        
        # skip house if all target objects are not present
        all_objects = list(findkeys(self.house['objects'],'assetId'))
        keyed_obj_types = [('Clock' if ('AlarmClock' in s) else s) for s in self.target_object_types]
        if len([e for e in keyed_obj_types if e not in '\n'.join(all_objects)]) > 0:
            return False

        pos2 = self.controller.step(action="TeleportFull", **self.usable_starting_positions[2])
        pos1 = self.controller.step(action="TeleportFull", **self.usable_starting_positions[1])
        pos0 = self.controller.step(action="TeleportFull", **self.usable_starting_positions[0])
        if not (pos0 and pos1 and pos2):
            return False

        return True
    
    def reset(self):
        super().reset()
        self.starting_position_index = 0
        self.real_object_index = 0

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

        # NOTE: Setup the Controller
        if self.controller is None:
            self.controller = self.controller_type(**self.args.controller_args)
            get_logger().info(
                f"Using Controller commit id: {self.controller._build.commit_id}"
            )
            while not self.increment_scene():
                pass

        # NOTE: determine if the house should be changed.
        if (
            self.episode_index == 0
            or
            self.episode_index % (len(self.target_object_types) * len(self.usable_starting_positions)) == 0
        ):
            self.starting_position_index = 0
            self.real_object_index = 0
            while not self.increment_scene():
                pass
                
        target_object_type = self.target_object_types[self.real_object_index]
        target_object_ids = self.target_objects_in_scene.get(target_object_type, [])
        
        event = self.controller.step(action="TeleportFull", **self.usable_starting_positions[self.starting_position_index])
        self.controller.nominal_neck = -1
        if not event:
            get_logger().error(f"Teleport failed in house {self.house_index} for position {self.starting_position_index}")
            self.real_object_index = 0
            self.starting_position_index += 1
            return None

        self.episode_index += 1
        self.args.max_tasks -= 1

        task_id = (f'phone_to_proc_{target_object_type}'
            f'__house_{self.house_index}'
            f'__starting_position_{self.starting_position_index}'
            f'__proc_{str(self.args.process_ind)}'
            f'__epidx_{str(self.episode_index)}')

        self._last_sampled_task = self.object_nav_task_type(
            controller=self.controller,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type="l2",
            house=self.house,
            distance_cache=self.distance_cache,
            task_info={
                "mode": self.args.mode,
                "process_ind": self.args.process_ind,
                "house_name": str(self.house_index),
                "rooms": len(self.house["rooms"]),
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": self.starting_position_index,
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
                "id": task_id,
            },
        )
        self.real_object_index += 1
        if self.real_object_index == len(self.target_object_types):
            self.real_object_index = 0
            self.starting_position_index +=1 
        
        return self._last_sampled_task




class ObjectNavInDomainTestTaskSampler(ObjectNavTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=min(self.reset_tasks // 10, 4), dtype=np.uint8
            ).tolist()
        )
        self.args.controller_args = self.args.controller_args.copy()

        houses = prior.load_dataset("procthor-10k")
        if cfg.eval and not cfg.evaluation.test_on_validation:
            self.houses = houses["test"]
        else:
            self.houses = houses["validation"]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_scene is None or self.last_scene != ep["scene"]:
                self.last_scene = ep["scene"]
                self.controller.reset(scene="Procedural")
                house = self.houses[ep["scene"]]
                self.controller.step(action="CreateHouse", house=house)

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                task_info={
                    "mode": self.args.mode,
                    "house_name": ep["scene"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task


class ProcTHORObjectNavTestTaskSampler(ObjectNavTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_house_idx = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=self.reset_tasks // 10, dtype=np.uint8
            ).tolist()
        )

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_house_idx is None or self.last_house_idx != ep["house"]:
                # NOTE: reusing unused variable name. Had trouble adding one real quick.
                houses_dataset = self.args.resample_same_scene_freq

                house = houses_dataset[ep["house"]]
                self.controller.reset()
                self.controller.step(
                    action="CreateHouse", house=house, raise_for_failure=True
                )
                self.last_house_idx = ep["house"]

            self.controller.step(
                action="SetObjectFilter",
                objectIds=ep["targetObjectIds"],
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                get_logger().error(
                    f"Teleport failing in {ep['house']} at {ep['agentPose']}"
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                task_info={
                    "mode": self.args.mode,
                    "house_name": ep["house"],
                    "target_object_ids": ep["targetObjectIds"],
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['house']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task


class TrainiTHORTaskSampler(ObjectNavTaskSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        random.shuffle(self.args.houses)

    @property
    def reachable_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.reachable_positions_map[self.house_inds_index]

    @property
    def house_index(self) -> int:
        return self.house_inds_index

    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.house_inds_index = (self.house_inds_index + 1) % len(self.args.houses)
        self.controller.reset(scene=self.args.houses[self.house_inds_index])
        event = self.controller.step(action="GetReachablePositions")
        if not event:
            return False
        self.reachable_positions_map[self.house_inds_index] = event.metadata[
            "actionReturn"
        ]
        return True

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

        # NOTE: Setup the Controller
        if self.controller is None:
            self.controller = Controller(**self.args.controller_args)
            get_logger().info(
                f"Using Controller commit id: {self.controller._build.commit_id}"
            )

        # NOTE: determine if the house should be changed.
        if force_advance_scene or (
            self.resample_same_scene_freq > 0
            and self.episode_index % self.resample_same_scene_freq == 0
        ):
            while not self.increment_scene():
                pass

        shuffled_objects = False
        if cfg.ithor.p_shuffle_objects > 0:
            # NOTE: Don't cache visible objects because they might change.
            self.visible_objects_cache = dict()

            # NOTE: Must update because objectIds update after initial random spawn.
            self.objects_in_scene_map = dict()

            if random.random() < cfg.ithor.p_shuffle_objects:
                event = self.controller.step(
                    action="InitialRandomSpawn",
                    randomSeed=random.randint(0, 2 ** 30),
                    forceVisible=True,
                )
                if event:
                    shuffled_objects = True
                else:
                    get_logger().warning("InitialRandomSpawn failing!")

        # NOTE: Choose target object
        while True:
            try:
                # NOTE: The loop avoid a very rare edge case where the agent
                # starts out trapped in some part of the room.
                target_object_type, target_object_ids = self.sample_target_object_ids()
                break
            except ValueError:
                while not self.increment_scene():
                    pass

        if random.random() < cfg.procthor.p_randomize_materials:
            self.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.controller.step(action="ResetMaterials", raise_for_failure=True)

        self.controller.step(
            action="SetObjectFilter",
            objectIds=target_object_ids,
            raise_for_failure=True,
        )

        # NOTE: Set agent pose
        standing = (
            {}
            if self.args.controller_args["agentMode"] == "locobot"
            else {"standing": True}
        )
        starting_pose = AgentPose(
            position=random.choice(self.reachable_positions),
            rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
            horizon=30,
            **standing,
        )
        event = self.controller.step(action="TeleportFull", **starting_pose)
        if not event:
            get_logger().warning(
                f"Teleport failing in {self.house_index} at {starting_pose}"
            )

        self.episode_index += 1
        self.args.max_tasks -= 1

        self._last_sampled_task = ObjectNavTask(
            controller=self.controller,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type=self.args.distance_type,
            distance_cache=self.distance_cache,
            task_info={
                "mode": self.args.mode,
                "process_ind": self.args.process_ind,
                "rooms": 1,
                "house_name": self.args.houses[self.house_inds_index],
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
                "shuffled_objects": shuffled_objects,
            },
        )
        return self._last_sampled_task
