import copy
import pdb
import random
import math
import torch
import ai2thor
import numpy as np
from ai2thor.controller import Controller
# import ai2thor.robot_client #import Controller as RealController
import cv2
from PIL import Image
# from lang_sam import LangSAM
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime

from typing import Dict, Optional, Set, Sequence, List, Tuple

from allenact.utils.system import get_logger
from .stretch_initialization_utils import (
    INTEL_VERTICAL_FOV,
    STRETCH_BUILD_ID,
    AGENT_RADIUS_LIST,
    AGENT_MOVEMENT_CONSTANT,
    ADDITIONAL_ARM_ARGS,
    ADDITIONAL_MOVING_ARGS,
    AGENT_ROTATION_DEG,
    WRIST_ROTATION,
    ARM_MOVE_CONSTANT,
    HORIZON,
    STRETCH_WRIST_INVALID_MIN,
    STRETCH_WRIST_INVALID_MAX,
    INTEL_CAMERA_WIDTH,
    INTEL_CAMERA_HEIGHT
)
from .utils_functions import (
    get_rooms_polymap_and_type,
    get_room_id_from_location,
    sum_dist_path, calc_arm_movement,
)

from .type_utils import THORActions
from collections import deque

class StretchController:
    def __init__(self, reverse_at_boundary=False, smaller_action=False, his_len=5, arm_y_range=(0.4, 0.8), **kwargs):
        self.reverse_at_boundary = reverse_at_boundary
        self.smaller_action = smaller_action
        self.arm_y_range = arm_y_range
        get_logger().warning(self.reverse_at_boundary)
        get_logger().warning(f"smaller actino: {self.smaller_action}")
        self.controller = Controller(**kwargs)
        get_logger().info(f"Using Controller commit id: {self.controller._build.commit_id}")
        assert STRETCH_BUILD_ID in self.controller._build.commit_id
        self.mode = None
        self.his_len = his_len
        self._nav_cam_his = deque(maxlen=self.his_len)
        self._manip_cam_his = deque(maxlen=self.his_len)
        self._proprio_his = deque(maxlen=self.his_len)

    def get_objects_in_hand_sphere(self):
        return self.controller.last_event.metadata["arm"]["pickupableObjects"]

    def get_held_objects(self):
        return self.controller.last_event.metadata["arm"]["heldObjects"]

    def get_arm_sphere_center(self):
        return self.controller.last_event.metadata["arm"]["handSphereCenter"]

    def get_object_properties(self, object_id):
        return self.controller.last_event.get_object(object_id)

    def agent_l2_distance_to_object(self, object_id, ignore_y=False):
        agent_location = self.controller.last_event.metadata["agent"]["position"]

        # TODO suboptimal proxy
        object_location = self.controller.last_event.get_object(object_id)["position"]

        return math.sqrt(
            (agent_location["x"] - object_location["x"]) ** 2
            + (0 if ignore_y else (agent_location["z"] - object_location["z"]) ** 2)
            + (agent_location["z"] - object_location["z"]) ** 2
        )

    @property
    def navigation_camera(self):
        frame = self.controller.last_event.frame
        return frame[:, 6:390, :]

    @property
    def manipulation_camera(self):
        frame = self.controller.last_event.third_party_camera_frames[0]
        return frame[:, 6:390, :3]

    @property
    def nav_cam_his(self):
        return np.concatenate(
            self._nav_cam_his,
            axis=0
        )

    @property
    def manip_cam_his(self):
        return np.concatenate(
            self._manip_cam_his,
            axis=0
        )

    @property
    def proprio_his(self):
        return np.concatenate(
            self._proprio_his,
            axis=0
        )

    @property
    def manipulation_depth_frame(self):
        frame = self.controller.last_event.third_party_depth_frames[0]
        return frame[:, 6:390]

    @property
    def navigation_depth_frame(self):
        frame = self.controller.last_event.depth_frame
        return frame[:, 6:390]

    def get_relative_stretch_current_arm_state(self):
        arm = self.controller.last_event.metadata["arm"]["joints"]
        z = arm[-1]["rootRelativePosition"]["z"]
        x = arm[-1]["rootRelativePosition"]["x"]
        assert abs(x - 0) < 1e-3
        y = arm[0]["rootRelativePosition"]["y"] - 0.16297650337219238
        return dict(x=x, y=y, z=z)

    def step(self, **kwargs):
        return self.controller.step(**kwargs)

    def calibrate_cameras(self):
        self.step(action="Teleport", horizon=0, standing=True)
        self.step(
            action="RotateCameraMount",
            degrees=27.0 + random.choice(np.arange(-2, 2, 0.2)),
            secondary=False,
        )
        self.step(
            action="ChangeFOV",
            fieldOfView=59 + random.choice(np.arange(-1, 1, 0.1)),
            camera="FirstPersonCharacter",
        )
        self.step(
            action="RotateCameraMount",
            degrees=32.0 + random.choice(np.arange(-2, 2, 0.2)),
            secondary=True,
        )
        self.step(
            action="ChangeFOV",
            fieldOfView=59 + random.choice(np.arange(-1, 1, 0.1)),
            camera="SecondaryCamera",
        )

    def reset_obs_his(self):
        for _ in range(self.his_len):
            self._nav_cam_his.append(self.navigation_camera[np.newaxis, ...])
        for _ in range(self.his_len):
            self._manip_cam_his.append(self.manipulation_camera[np.newaxis, ...])
        for _ in range(self.his_len):
            self._proprio_his.append(self.get_processed_arm_proprioception())

    def update_obs_his(self):
        self._nav_cam_his.append(self.navigation_camera[np.newaxis, ...])
        self._manip_cam_his.append(self.manipulation_camera[np.newaxis, ...])
        self._proprio_his.append(self.get_processed_arm_proprioception())

    def reset(self, scene):
        self.current_scene_json = scene
        self.agent_ids = [i for (i, r) in AGENT_RADIUS_LIST]

        # add metadata here for navmesh?
        base_agent_navmesh = {
            "agentHeight": 1.8,
            "agentSlope": 10,
            "agentClimb": 0.5,
            "voxelSize": 0.1666667,
        }
        scene["metadata"]["navMeshes"] = [
            {**base_agent_navmesh, **{"id": i, "agentRadius": r}} for (i, r) in AGENT_RADIUS_LIST
        ]

        scene["metadata"]["agent"]["horizon"] = HORIZON

        reset_event = self.controller.reset(scene=scene)

        self.calibrate_cameras()

        teleport_event = self.controller.step(
            action="TeleportFull",
            **scene["metadata"]["agent"],  # forceAction=True
        )

        if not teleport_event.metadata["lastActionSuccess"]:
            get_logger().info("FAILED TO TELEPORT AGENT AFTER INITIALIZATION", scene)
            return teleport_event

        # Do not display the unrealistic blue sphere on the agent's gripper
        self.controller.step("ToggleMagnetVisibility", visible=False, raise_for_failure=True)
        self.reset_obs_his()
        # self.set_object_filter([])
        # get_logger().warning(self.controller.last_event.metadata["thirdPartyCameras"][0]["fieldOfView"])
        return reset_event

    def get_all_objects_of_type(self, object_type):
        return self.controller.last_event.objects_by_type(object_type)

    def objects_is_visible_in_camera(self, object_ids, which_camera="nav", maximum_distance=2):
        # choices: 'nav','manip','both'
        # some duplication with get seen but backwards compatible/utility function
        obj_in_nav = self.controller.step(
            "GetVisibleObjects", maxDistance=maximum_distance
        ).metadata["actionReturn"]
        obj_in_manip = self.controller.step( 
            "GetVisibleObjects", maxDistance=maximum_distance, thirdPartyCameraIndex=0
        ).metadata["actionReturn"]
        camera_presence = {
            "nav": obj_in_nav,
            "manip": obj_in_manip,
            "both": list(set((obj_in_nav or []) + (obj_in_manip or []))),
        }
        for object_id in object_ids:
            if object_id in camera_presence.get(which_camera, False):
                return True
        return False

    def object_is_visible_in_camera(self, object_id, which_camera="nav", maximum_distance=2):
        # choices: 'nav','manip','both'
        # some duplication with get seen but backwards compatible/utility function
        obj_in_nav = self.controller.step(
            "GetVisibleObjects", maxDistance=maximum_distance
        ).metadata["actionReturn"]
        obj_in_manip = self.controller.step( 
            "GetVisibleObjects", maxDistance=maximum_distance, thirdPartyCameraIndex=0
        ).metadata["actionReturn"]
        camera_presence = {
            "nav": obj_in_nav,
            "manip": obj_in_manip,
            "both": list(set((obj_in_nav or []) + (obj_in_manip or []))),
        }
        if object_id in camera_presence.get(which_camera, False):
            return True
        return False

    def get_seen_objects(self, uninteresting_object_types, which_camera="nav", maximum_distance=4):
        obj_in_nav = self.controller.step(
            "GetVisibleObjects", maxDistance=maximum_distance
        ).metadata["actionReturn"]
        obj_in_manip = self.controller.step(
            "GetVisibleObjects", maxDistance=maximum_distance, thirdPartyCameraIndex=0
        ).metadata["actionReturn"]
        camera_presence = {
            "nav": obj_in_nav,
            "manip": obj_in_manip,
            "both": list(set((obj_in_nav or []) + (obj_in_manip or []))),
        }
        seen_ids = [
            id
            for id in camera_presence.get(which_camera, False)
            if self.controller.last_event.get_object(id)["objectType"]
            not in uninteresting_object_types
        ]
        return seen_ids

    def get_unseen_objects(self, seen_ids, uninteresting_object_types):
        return set(
            o["objectId"]
            for o in self.controller.last_event.metadata["objects"]
            if o["objectId"] not in seen_ids and o["objectType"] not in uninteresting_object_types
        )

    def get_object_type_and_pos_dict(self, uninteresting_object_types):
        all_obj = {}
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["objectType"] in uninteresting_object_types:
                continue
            all_obj[obj["objectId"]] = {
                "type": obj["objectType"],
                "pos": obj["position"],
            }
        return all_obj

    def set_object_filters_by_object_type(self, object_types):
        if len(object_types) == 0:  # no-op for no object types (exploration)
            return True
        target_object_ids = []
        for obj_type in object_types:
            target_object_ids += self.get_all_objects_of_type(obj_type)
        target_object_ids = [obj["objectId"] for obj in target_object_ids]
        door_ids = [door["id"] for door in self.current_scene_json["doors"]]
        return self.controller.step(
            action="SetObjectFilter",
            objectIds=target_object_ids + door_ids,
            raise_for_failure=True,
        )

    def reset_object_filter(self):
        return self.controller.step(action="ResetObjectFilter")

    def get_object_receptacle_types(self, object_id):
        source_receptacle_ids = self.get_object_properties(object_id)["parentReceptacles"]

        if source_receptacle_ids is None:
            source_receptacle_ids = []

        source_receptacle_types = [
            self.get_object_properties(obj_id)["objectType"] for obj_id in source_receptacle_ids
        ]
        return source_receptacle_types

    def get_locations_on_receptacle(self, receptacle_id):
        result = self.step(
            action="GetSpawnCoordinatesAboveReceptacle", objectId=receptacle_id, anywhere=True
        )
        return result.metadata["actionReturn"]

    def get_current_agent_position(self):
        return self.controller.last_event.metadata["agent"]["position"]

    def get_current_agent_full_pose(self):
        return self.controller.last_event.metadata["agent"]

    def get_available_objects_from_list(self, target_object_types, pickupable=False):
        available_types = []
        for obj_type in target_object_types:
            all_valid_objects = self.get_all_objects_of_type(obj_type)
            if pickupable:
                all_valid_objects = [o for o in all_valid_objects if o["pickupable"]]
            if len(all_valid_objects) > 0:
                available_types.append(obj_type)
        return available_types

    def get_obj_pos_from_obj_id(self, object_id):
        return self.controller.last_event.get_object(object_id)["axisAlignedBoundingBox"]["center"]

    def get_object_position(self, object_id):
        try:
            return self.controller.last_event.get_object(object_id)["position"]
        except:
            event = self.controller.last_event.get_object(object_id)
            print(event)
            print(object_id)

    def get_reachable_positions(self):
        rp_event = self.controller.step(action="GetReachablePositions")
        if not rp_event:
            # NOTE: Skip scenes where GetReachablePositions fails
            get_logger().warning(f"GetReachablePositions failed in {self.current_scene_json}")
            return False
        reachable_positions = rp_event.metadata["actionReturn"]
        return reachable_positions

    def stop(self):
        self.controller.stop()

    def agent_step(self, action):
        agents_full_pose_before_action = copy.deepcopy(
            dict(
                agent_pose=self.get_current_agent_full_pose(),
                arm_pose=self.get_relative_stretch_current_arm_state(),
            )
        )
        if action == THORActions.move_ahead:
            action_dict = dict(action="MoveAgent", ahead=AGENT_MOVEMENT_CONSTANT)
        elif action == THORActions.move_back:
            action_dict = dict(action="MoveAgent", ahead=-AGENT_MOVEMENT_CONSTANT)
        elif action in [
            THORActions.rotate_left,
            THORActions.rotate_right,
            THORActions.rotate_left_small,
            THORActions.rotate_right_small,
        ]:  #  add for smaller rotations
            if action == THORActions.rotate_right:
                degree = AGENT_ROTATION_DEG
            elif action == THORActions.rotate_left:
                degree = -AGENT_ROTATION_DEG
            elif action == THORActions.rotate_right_small:
                degree = AGENT_ROTATION_DEG / 5
            elif action == THORActions.rotate_left_small:
                degree = -AGENT_ROTATION_DEG / 5

            action_dict = dict(action="RotateAgent", degrees=degree)
        elif action in [
            THORActions.move_arm_down,
            THORActions.move_arm_in,
            THORActions.move_arm_out,
            THORActions.move_arm_up,
            THORActions.move_arm_down_small,
            THORActions.move_arm_in_small,
            THORActions.move_arm_out_small,
            THORActions.move_arm_up_small,
        ]:
            base_position = self.get_relative_stretch_current_arm_state()
            change_value = ARM_MOVE_CONSTANT
            small_change_value = ARM_MOVE_CONSTANT / 5
            if action == THORActions.move_arm_up:
                base_position["y"] += change_value
            elif action == THORActions.move_arm_down:
                base_position["y"] -= change_value
            elif action == THORActions.move_arm_out:
                base_position["z"] += change_value
            elif action == THORActions.move_arm_in:
                base_position["z"] -= change_value
            if action == THORActions.move_arm_up_small:
                base_position["y"] += small_change_value
            elif action == THORActions.move_arm_down_small:
                base_position["y"] -= small_change_value
            elif action == THORActions.move_arm_out_small:
                base_position["z"] += small_change_value
            elif action == THORActions.move_arm_in_small:
                base_position["z"] -= small_change_value
            action_dict = dict(
                action="MoveArm",
                position=dict(x=base_position["x"], y=base_position["y"], z=base_position["z"]),
            )
        elif action in [
            THORActions.wrist_open,
            THORActions.wrist_close,
        ]:  #  add for smaller actions
            rotation_value = WRIST_ROTATION
            if action == THORActions.wrist_open:
                rotation_value = -rotation_value
            elif action == THORActions.wrist_close:
                rotation_value = rotation_value
            action_dict = dict(action="RotateWristRelative", yaw=rotation_value)
        elif action == THORActions.pickup:
            action_dict = dict(action="PickupObject")
        elif action == THORActions.dropoff:
            action_dict = dict(action="ReleaseObject")
        elif action in [THORActions.done, THORActions.sub_done]:
            action_dict = dict(action="Pass")
        else:
            print("Action not defined")
            pdb.set_trace()
        if action_dict["action"] in ["MoveAgent", "RotateWristRelative", "MoveArm"]:
            action_dict = {**action_dict, **ADDITIONAL_ARM_ARGS}
        event = self.step(**action_dict)

        if action == THORActions.dropoff:
            self.step(action="AdvancePhysicsStep", simSeconds=2)

        agents_full_pose_after_action = copy.deepcopy(
            dict(
                agent_pose=self.get_current_agent_full_pose(),
                arm_pose=self.get_relative_stretch_current_arm_state(),
            )
        )

        # test for checking move arm is failing or not
        #  return false if arm move  is called but pose is not changed
        if action in THORActions.ARM_ACTIONS:
            if (
                calc_arm_movement(
                    agents_full_pose_before_action["arm_pose"],
                    agents_full_pose_after_action["arm_pose"],
                )
                < 1e-3
            ):
                event.metadata["lastActionSuccess"] = False

        return event
    
    @property
    def continuous_action_dim(self):
        return 5

    def continuous_agent_step(self, action, return_to_start=True):
        # agents_full_pose_before_action = copy.deepcopy(
        #     dict(
        #         agent_pose=self.get_current_agent_full_pose(),
        #         arm_pose=self.get_relative_stretch_current_arm_state(),
        #     )
        # )
        # action = moving forard, rotation,
        if self.smaller_action:
            action[:2] = action[:2] / 2
        action_dict = dict(
            action="RotateAgent", degrees=AGENT_ROTATION_DEG*action[1], **ADDITIONAL_MOVING_ARGS,
            renderImage=False,
        )
        event = self.step(**action_dict)
        # print("Rotate Success:", event.metadata["lastActionSuccess"])
        rotate_success = (abs(action[1]) < 1e-3) or event.metadata["lastActionSuccess"]

        action_dict = dict(
            action="MoveAgent", ahead=AGENT_MOVEMENT_CONSTANT*action[0], **ADDITIONAL_MOVING_ARGS,
            renderImage=False,
        )
        event = self.step(**action_dict)
        move_success = (abs(action[0]) < 1e-3) or event.metadata["lastActionSuccess"]
        # print("Rotate Event Success:", event_rotate.metadata["lastActionSuccess"])

        base_position = self.get_relative_stretch_current_arm_state()
        base_position["y"] += ARM_MOVE_CONSTANT * action[2]
        base_position["z"] += ARM_MOVE_CONSTANT * action[3]

        base_position["y"] = np.clip(base_position["y"], *self.arm_y_range)
        base_position["z"] = np.clip(base_position["z"], 0, 0.8064)
        action_dict = dict(
            action="MoveArm",
            position=dict(x=base_position["x"], y=base_position["y"], z=base_position["z"]),
            **ADDITIONAL_ARM_ARGS,
            returnToStart = return_to_start,
            renderImage=False,
        )
        event = self.step(**action_dict)
        arm_success = (abs(action[2]) < 1e-3 and abs(action[3]) < 1e-3) or event.metadata["lastActionSuccess"]

        # Clip Rotation Angle
        # if WRIST_ROTATION * action[4]
        curr_wrist = self.get_arm_wrist_raw_rotation() # -180 to 180
        
        rotation_value = WRIST_ROTATION * action[4]
        test_log = False
        if (STRETCH_WRIST_INVALID_MIN <= (curr_wrist + rotation_value) <= STRETCH_WRIST_INVALID_MAX):
            if self.reverse_at_boundary:
                # gap = np.abs()
                # curr_wrist should be in 
                # dis_to_other_boundary = max(
                #     np.abs(STRETCH_WRIST_INVALID_MAX - curr_wrist),
                #     np.abs(STRETCH_WRIST_INVALID_MIN - curr_wrist)
                # ) 
                dis_to_mid = np.abs(90 - curr_wrist)
                #     np.abs(STRETCH_WRIST_INVALID_MIN - curr_wrist)
                # # )
                # rotation_value = -np.sign(action[4]) * (360 - dis_to_other_boundary - 2)
                rotation_value = -np.sign(action[4]) * (180 - dis_to_mid)
                # test_log = True
            else:
                rotation_value = 0

        action_dict = dict(
            action="RotateWristRelative", yaw=rotation_value, **ADDITIONAL_ARM_ARGS,
            renderImage=True,
            returnToStart = return_to_start
        )
        # if test_log:
        #     get_logger().warning(
        #         f"Before Step: {action[4]}, {rotation_value}, {self.controller.last_event.metadata['arm']['joints'][-1]['rootRelativeRotation']['w']}"
        #     )
        event = self.step(**action_dict)
        wrist_success = (abs(action[4]) < 1e-3) or event.metadata["lastActionSuccess"]

        self.update_obs_his()

        return event, (arm_success and wrist_success), (move_success and rotate_success)



    def continuous_agent_step_nw(self, action, return_to_start=True):
        if self.smaller_action:
            action[:2] = action[:2] / 2
        action_dict = dict(
            action="RotateAgent", degrees=AGENT_ROTATION_DEG*action[1], **ADDITIONAL_MOVING_ARGS,
            renderImage=False,
        )
        event = self.step(**action_dict)
        rotate_success = (abs(action[1]) < 1e-3) or event.metadata["lastActionSuccess"]

        action_dict = dict(
            action="MoveAgent", ahead=AGENT_MOVEMENT_CONSTANT*action[0], **ADDITIONAL_MOVING_ARGS,
            renderImage=False,
        )
        event = self.step(**action_dict)
        move_success = (abs(action[0]) < 1e-3) or event.metadata["lastActionSuccess"]
        # print("Rotate Event Success:", event_rotate.metadata["lastActionSuccess"])

        base_position = self.get_relative_stretch_current_arm_state()
        base_position["y"] += ARM_MOVE_CONSTANT * action[2]
        base_position["z"] += ARM_MOVE_CONSTANT * action[3]

        base_position["y"] = np.clip(base_position["y"], *self.arm_y_range)
        base_position["z"] = np.clip(base_position["z"], 0, 0.8064)
        action_dict = dict(
            action="MoveArm",
            position=dict(x=base_position["x"], y=base_position["y"], z=base_position["z"]),
            **ADDITIONAL_ARM_ARGS,
            returnToStart = return_to_start,
            renderImage=True,
        )
        event = self.step(**action_dict)
        arm_success = (abs(action[2]) < 1e-3 and abs(action[3]) < 1e-3) or event.metadata["lastActionSuccess"]

        self.update_obs_his()

        return event, arm_success, (move_success and rotate_success)

    def base_subgoal_to_action(self, base_subgoal):
        x, z, theta = base_subgoal

        forward_distance = np.sqrt(x ** 2 + z ** 2)

        rotation_angle_first = np.rad2deg(np.arctan(
            y / (x + 1e-5)
        ))

        if x < 0:
            forward_distance = - forward_distance

        rotation_anlge_second = theta - rotation_angle_first

        return [rotation_angle_first, forward_distance, rotation_anlge_second]

    def arm_subgoal_to_action(self, arm_subgoal):
        arm_subgoal[0] = np.sum(self.arm_y_range) / 2 + arm_subgoal[0] * (self.arm_y_range[1] - self.arm_y_range[0]) * 0.5
        # get_logger().warning()
        # Maximum Range is 0.53
        arm_subgoal[1] = arm_subgoal[1] * 0.53
        # 
        arm_subgoal[2] = arm_subgoal[2] * 75

        return arm_subgoal
    #     y, z, x = arm_subgoal

    #     target_

    #     return 


    def continuous_agent_step_with_subgoal(self, action, return_to_start=True):
        # action = moving forard, rotation,

        arm_or_base = action[-1]
        # if action[-1] = 1

        # Assume action all with in [-1, 1]

        base_subgoal = action[:3]
        base_subgoal[0] *= AGENT_MOVEMENT_CONSTANT
        base_subgoal[1] *= AGENT_MOVEMENT_CONSTANT
        base_subgoal[2] *= 180

        arm_subgoal = action[3:6].copy()
        arm_subgoal = self.arm_subgoal_to_action(arm_subgoal)
        # action_subgoal[0] = np.sum(self.arm_y_range) / 2 + action_subgoal[0] * (self.arm_y_range[1] - self.arm_y_range[0]) * 0.5
        # # get_logger().warning()
        # # Maximum Range is 0.53
        # action_subgoal[1] = action_subgoal[1] * 0.53
        # # 
        # action_subgoal[2] = action_subgoal[2] * 75

        rotate_success_0 = True
        move_success = True  
        rotate_success_1 = True
        arm_success = True
        wrist_success = True
        event = None
        if arm_or_base == 0:
            base_action_seq = self.base_subgoal_to_action(base_subgoal)

            # Base Subgoal
            action_dict = dict(
                action="RotateAgent", degrees=base_action_seq[0], **ADDITIONAL_MOVING_ARGS,
                renderImage=False,
            )
            event = self.step(**action_dict)
            rotate_success_0 = (abs(action[1]) < 1e-3) or event.metadata["lastActionSuccess"]

            action_dict = dict(
                action="MoveAgent", ahead=base_action_seq[1], **ADDITIONAL_MOVING_ARGS,
                renderImage=False,
            )
            event = self.step(**action_dict)
            move_success = (abs(action[0]) < 1e-3) or event.metadata["lastActionSuccess"]

            action_dict = dict(
                action="RotateAgent", degrees=base_action_seq[2], **ADDITIONAL_MOVING_ARGS,
                renderImage=True,
            )
            event = self.step(**action_dict)
            rotate_success_1 = (abs(action[1]) < 1e-3) or event.metadata["lastActionSuccess"]

        elif arm_or_base == 1:
            # Arm Subgoal
            base_position = self.get_relative_stretch_current_arm_state()
            base_position["y"] = arm_subgoal[0]
            base_position["z"] = arm_subgoal[1]
            # base_position["y"] = np.clip(base_position["y"], *self.arm_y_range)
            # base_position["z"] = np.clip(base_position["z"], 0, 0.8064)

            action_dict = dict(
                action="MoveArm",
                position=dict(x=base_position["x"], y=base_position["y"], z=base_position["z"]),
                **ADDITIONAL_ARM_ARGS,
                returnToStart = return_to_start,
                renderImage=False,
            )
            event = self.step(**action_dict)
            arm_success = (abs(action[2]) < 1e-3 and abs(action[3]) < 1e-3) or event.metadata["lastActionSuccess"]

            # Clip Rotation Angle
            curr_wrist = self.get_arm_wrist_raw_rotation() # -180 to 180
            
            rotation_value = arm_subgoal[2] - curr_wrist
            action_dict = dict(
                action="RotateWristRelative", yaw=rotation_value, **ADDITIONAL_ARM_ARGS,
                renderImage=True,
                returnToStart = return_to_start
            )
            event = self.step(**action_dict)
            wrist_success = (abs(action[4]) < 1e-3) or event.metadata["lastActionSuccess"]
        # else:
            

        self.update_obs_his()

        return event, (arm_success and wrist_success), (move_success and rotate_success_0 and rotate_success_1)


    def get_arm_wrist_position(self):
        joint = self.controller.last_event.metadata["arm"]["joints"][-1]
        assert joint["name"] == "stretch_robot_wrist_2_jnt"
        return [joint["rootRelativePosition"][k] for k in ["x", "y", "z"]]

    def get_arm_wrist_absolute_position(self):
        joint = self.controller.last_event.metadata["arm"]["joints"][-1]
        assert joint["name"] == "stretch_robot_wrist_2_jnt"
        return [joint["position"][k] for k in ["x", "y", "z"]]

    def get_arm_wrist_rotation(self):
        joint = self.controller.last_event.metadata["arm"]["joints"][-1]
        assert joint["name"] == "stretch_robot_wrist_2_jnt"
        return math.fmod(
            joint["rootRelativeRotation"]["w"] * joint["rootRelativeRotation"]["y"], 360
        )

    def get_arm_proprioception(self):
        arm_position = self.get_arm_wrist_position()
        arm_rotation = self.get_arm_wrist_raw_rotation()
        full_pose = arm_position + [arm_rotation]
        return full_pose
    
    def get_processed_arm_proprioception(self):
        arm_info = self.get_arm_proprioception()
        y= arm_info[1]
        z = arm_info[2]
        w_rotation = arm_info[3]
        pos_encoding = 2 * (((w_rotation + 270) % 360) / 360) - 1
        w_sin = np.sin(w_rotation / 360 * 2 * np.pi)
        w_cos = np.cos(w_rotation / 360 * 2 * np.pi)
        obs = np.array([
            y, z, w_sin, w_cos, pos_encoding
            # y, z, pos_encoding
        ], dtype=np.float32)
        return obs

    def get_arm_wrist_raw_rotation(self):
        current_wrist = copy.deepcopy(self.controller.last_event.metadata["arm"]["joints"][-1])
        assert current_wrist["name"] == "stretch_robot_wrist_2_jnt"
        return current_wrist["rootRelativeRotation"]["w"]

    # calculate the shrotest path to that location
    def get_shortest_path_to_object(
        self, object_id, initial_position=None, initial_rotation=None, specific_agent_meshes=None
    ):
        """
        Computes the shortest path to an object from an initial position using a controller
        :param controller: agent controller
        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids
        if initial_position is None:
            initial_position = self.get_current_agent_position()
        for nav_mesh_id in specific_agent_meshes:
            args = dict(
                action="GetShortestPath",
                objectId=object_id,
                position=initial_position,
                navMeshId=nav_mesh_id,  # update to incorporate navmesh
            )
            if initial_rotation is not None:
                args["rotation"] = initial_rotation
            event = self.step(**args)
            if event.metadata["lastActionSuccess"]:
                corners = event.metadata["actionReturn"]["corners"]
                if len(corners) == 0:
                    continue
                self.last_successful_path = corners
                return corners  # This will slow down data generation

        return None

    # calculate the shrotest path to that location
    def get_shortest_path_to_point(
        self,
        target_position,
        initial_position=None,
        initial_rotation=None,
        specific_agent_meshes=None,
    ):
        """
        Computes the shortest path to an object from an initial position using a controller
        :param controller: agent controller
        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids
        if initial_position is None:
            initial_position = self.get_current_agent_position()

        for nav_mesh_id in specific_agent_meshes:
            args = dict(
                action="GetShortestPathToPoint",
                position=initial_position,
                target=target_position,
                navMeshId=nav_mesh_id,  # update to incorporate navmesh
            )
            if initial_rotation is not None:
                args["rotation"] = initial_rotation
            event = self.step(**args)
            if event.metadata["lastActionSuccess"]:
                corners = event.metadata["actionReturn"]["corners"]
                if len(corners) == 0:
                    continue
                self.last_successful_path = corners
                return corners  # This will slow down data generation

        return None

    def get_closest_object_from_ids(self, object_ids, return_id_and_dist: bool = False):
        all_paths = [
            (
                obj_id,
                self.get_shortest_path_to_object(
                    obj_id, specific_agent_meshes=[self.agent_ids[-1]]
                ),
            )
            for obj_id in object_ids
        ]

        min_dist = float("inf")
        closest_obj_id = None
        for obj_id, path in all_paths:
            if path is None:
                continue
            dist = sum_dist_path(path)
            if dist < min_dist:
                min_dist = dist
                closest_obj_id = obj_id
        return closest_obj_id if not return_id_and_dist else (closest_obj_id, min_dist)

    def get_closest_object_of_type(
        self,
        object_type,
        return_id_and_dist: bool = False,
    ):
        object_ids = [o["objectId"] for o in self.get_all_objects_of_type(object_type)]

        return self.get_closest_object_from_ids(object_ids, return_id_and_dist)

    def get_shortest_path_to_room(self, room_id, specific_agent_meshes=None):
        room_polymap, room_type = get_rooms_polymap_and_type(
            self.current_scene_json
        )  # TODO this can be cached
        x = room_polymap[room_id].centroid.x
        z = room_polymap[room_id].centroid.y
        current_agent_position = self.controller.last_event.metadata["agent"]["position"]
        y = current_agent_position["y"]
        # navigate to point
        path = self.get_shortest_path_to_point(
            dict(x=x, y=y, z=z),
            initial_position=current_agent_position,
            specific_agent_meshes=specific_agent_meshes,
        )
        return path

    def get_objects_room_id_and_type(self, object_id):
        object_position = self.get_object_position(object_id)
        room_polymap, room_type = get_rooms_polymap_and_type(
            self.current_scene_json
        )
        room_id = get_room_id_from_location(room_polymap, object_position)
        return room_id, room_type[room_id]

    def find_closest_room_of_list(self, room_ids, return_id_and_dist: bool = False):
        all_paths = []
        for room_id in room_ids:
            path = self.get_shortest_path_to_room(
                room_id, specific_agent_meshes=[self.agent_ids[-1]]
            )
            all_paths.append((room_id, path))

        min_dist = float("inf")
        closest_room_id = None
        for room_id, path in all_paths:
            if path is None:
                continue
            dist = sum_dist_path(path)
            if dist < min_dist:
                min_dist = dist
                closest_room_id = room_id

        return closest_room_id if not return_id_and_dist else (closest_room_id, min_dist)


class StretchStochasticController(StretchController):
    def step(self, **kwargs):
        # TODO Add stochastic motion for arm
        if "action" in kwargs and kwargs["action"] == "MoveAgent":
            kwargs["ahead"] += np.random.normal(0, 0.01, 1)[0]
        if "action" in kwargs and kwargs["action"] == "RotateAgent":
            kwargs["degrees"] += np.random.normal(0, 0.5, 1)[0]
        return super(StretchStochasticController, self).step(**kwargs)


from timeout_decorator import timeout, TimeoutError

TIMEOUT_SECONDS = 15

@timeout(TIMEOUT_SECONDS)
def real_robot_call_with_timeout(controller, action_list):
    event = controller.controller.step({
        "action": action_list
    })
    return event
