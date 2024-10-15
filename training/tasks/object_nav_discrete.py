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
from training.robot.stretch_controller import StretchController
from training.robot.stretch_initialization_utils import ALL_STRETCH_ACTIONS
from training.robot.type_utils import THORActions

from matplotlib import pyplot as plt

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
        self.stretch_controller = controller
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
            self.stretch_controller.get_current_agent_position()
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
                    for o in self.stretch_controller.controller.last_event.metadata["objects"]
                }
                self.task_info["target_locations"].append(obj_id_to_obj_pos[object_id])
        
        # self.observations = [self.stretch_controller.last_event.frame]
        # TODO: hacky. Set to auto-id the camera sensor or get obs and de-normalize in a function
        print(self.sensor_suite)
        print(self.sensor_suite.sensors)
        self.observations = [
            self.sensor_suite.sensors['rgb_lowres_navigation'].frame_from_env(self.stretch_controller,self),
            self.sensor_suite.sensors['rgb_lowres_manipulation'].frame_from_env(self.stretch_controller,self)
        ]
        self._metrics = None

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

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(ALL_STRETCH_ACTIONS))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return ALL_STRETCH_ACTIONS

    def close(self) -> None:
        self.stretch_controller.stop()

    def _step(self, action: int) -> RLStepResult:
        # action_str = self.class_action_names()[action]
        stretch_action = ALL_STRETCH_ACTIONS[action]
        # self.stretch_controller.agent_step(
        #     stretch_action
        # )

        # if self.mirror:
        #     if action_str == "RotateRight":
        #         action_str = "RotateLeft"
        #     elif action_str == "RotateLeft":
        #         action_str = "RotateRight"
        # print(self.stretch_controller.las)
        self.task_info["taken_actions"].append(stretch_action)

        ### for the Kiana Experiment (tm)

        # self._success = self._is_goal_in_range()
        self._success = self.dist_to_target_func() <= 1 and self.dist_to_target_func() > 0
        self._took_end_action = self._success

        # if action_str == "End":
        #     # self._took_end_action = True
        #     # self._success = self._is_goal_in_range()
        #     self.last_action_success = self._success
        #     self.task_info["action_successes"].append(self._success)

        # if action_str == "End":
        #     ### for regular objectnav
        #     # self._took_end_action = True
        #     # self._success = self._is_goal_in_range()
        #     # self.last_action_success = self._success
        #     # self.task_info["action_successes"].append(True)

        #     ### for exploration only
        #     self.last_action_success = False
        #     self.task_info["action_successes"].append(False)
        # else:
        #     self.stretch_controller.step(action=action_str)
        #     self.last_action_success = bool(self.stretch_controller.controller.last_event)

        #     position = self.stretch_controller.get_current_agent_position()
        #     self.path.append(position)
        #     self.task_info["followed_path"].append(position)
        #     self.task_info["action_successes"].append(self.last_action_success)
        if stretch_action == THORActions.done:
            self.last_action_success = False
            self.task_info["action_successes"].append(False)
        else:
            self.stretch_controller.agent_step(
                stretch_action
            )
            self.last_action_success = bool(self.stretch_controller.controller.last_event)

            position = self.stretch_controller.get_current_agent_position()
            self.path.append(position)
            self.task_info["followed_path"].append(position)
            self.task_info["action_successes"].append(self.last_action_success)

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            # self.observations.append(self.stretch_controller.last_event.frame)
            # TODO: same as above
            self.observations.append(self.sensor_suite.sensors['rgb_lowres_navigation'].frame_from_env(self.stretch_controller,self))
            self.observations.append(self.sensor_suite.sensors['rgb_lowres_manipulation'].frame_from_env(self.stretch_controller,self))
        # print(self.get_observations())
        # plt.imshow(self.get_observations()["rgb_lowres_navigation"])
        # plt.imshow(self.get_observations()["rgb_lowres_manipulation"])
        # plt.show()
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

        if not self.last_action_success and "Look" not in self.task_info["taken_actions"][-1]:
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
            last_frame = self.stretch_controller.last_event.frame
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

        self.stretch_controller: Optional[StretchController] = None
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

    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.increment_scene_index()

        # self.stretch_controller.step(action="DestroyHouse", raise_for_failure=True)
        self.house = self.args.houses[self.house_index]

        # self.stretch_controller.controller.step(
        #     action="CreateHouse", house=self.house, raise_for_failure=True
        # )

        self.stretch_controller.reset(self.house)

        # NOTE: Set reachable positions
        if self.house_index not in self.reachable_positions_map:
            pose = self.house["metadata"]["agent"].copy()
            if self.args.controller_args["agentMode"] == "locobot":
                del pose["standing"]
            event = self.stretch_controller.step(action="TeleportFull", **pose)
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
            self.stretch_controller.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.stretch_controller.controller.step(action="ResetMaterials", raise_for_failure=True)

        door_ids = [door["id"] for door in self.house["doors"]]
        self.stretch_controller.step(
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
            horizon=0,
            **standing,
        )
        event = self.stretch_controller.step(action="TeleportFull", **starting_pose)
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
            controller=self.stretch_controller,
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
        self.stretch_controller.reset(scene=self.args.houses[self.house_inds_index])
        event = self.stretch_controller.step(action="GetReachablePositions")
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
        if self.stretch_controller is None:
            self.stretch_controller = Controller(**self.args.controller_args)
            get_logger().info(
                f"Using Controller commit id: {self.stretch_controller._build.commit_id}"
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
                event = self.stretch_controller.step(
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
            self.stretch_controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.stretch_controller.step(action="ResetMaterials", raise_for_failure=True)

        self.stretch_controller.step(
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
            horizon=0,
            **standing,
        )
        event = self.stretch_controller.step(action="TeleportFull", **starting_pose)
        if not event:
            get_logger().warning(
                f"Teleport failing in {self.house_index} at {starting_pose}"
            )

        self.episode_index += 1
        self.args.max_tasks -= 1

        self._last_sampled_task = ObjectNavTask(
            controller=self.stretch_controller,
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