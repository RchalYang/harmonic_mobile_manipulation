import os
from typing import (
    Dict,
    Any,
    Optional,
    List
)
import json
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import gym.spaces as gyms
from ai2thor.controller import Controller

from allenact.base_abstractions.misc import EnvType
from allenact.base_abstractions.sensor import Sensor
from training.tasks.object_nav import ObjectNavTask, ForkedPdb
from utils.utils import compute_movement_efficiency, reconstruct_resized_map_estimate
from utils.map_utils import centered_pixel_from_point

SpaceDict = gyms.Dict


class ProcTHORObjectNavCallbackSensor(Sensor[Controller, ObjectNavTask]):
    def __init__(self, uuid: str):
        super().__init__(uuid=uuid, observation_space=gyms.Discrete(1))

    def get_observation(
        self, env: EnvType, task: Optional[ObjectNavTask], *args: Any, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        if not task.visualize:
            return None

        # NOTE: Create top-down trajectory path visualization
        agent_path = [
            dict(x=p["x"], y=0.25, z=p["z"])
            for p in task._metrics["task_info"]["followed_path"]
        ]
        if len(task.stretch_controller.controller.last_event.third_party_camera_frames) <= 2:
            # assumes this is the only third party camera
            event = task.stretch_controller.controller.step(action="GetMapViewCameraProperties")
            cam = event.metadata["actionReturn"].copy()
            cam["orthographicSize"] += 1
            task.stretch_controller.controller.step(
                action="AddThirdPartyCamera", skyboxColor="white", **cam
            )
        event = task.stretch_controller.controller.step(action="VisualizePath", positions=agent_path)
        task.stretch_controller.controller.step(action="HideVisualizedPath")

        return {
            "observations": task.observations,
            # "map_observations": getattr(task, 'map_observations', None),
            "path": event.third_party_camera_frames[2],
            "house": task.house,
            **task._metrics,
        }

class LocalLoggingSensor(Sensor[Controller,ObjectNavTask]):

    def get_observation(
        self, env:Controller, task:ObjectNavTask, *args: Any, **kwargs: Any
    ) -> Any:
        if not task.visualize:
            return None
    
        # NOTE: Create top-down trajectory path visualization
        agent_path = [
            dict(x=p["x"], y=0.25, z=p["z"])
            for p in task._metrics["task_info"]["followed_path"]
        ]
        if task.distance_type != "realWorld":
            # THIS DOES NOT WORK FOR STRETCH (adjust count of third-party cameras)
            if len(env.last_event.third_party_camera_frames) < 1:
                event = env.step({"action": "GetMapViewCameraProperties"})
                cam = event.metadata["actionReturn"].copy()
                cam["orthographicSize"] += 1
                env.step(
                    {"action": "AddThirdPartyCamera", "skyboxColor":"white", **cam}
                )
            event = env.step({"action": "VisualizePath", "positions":agent_path})
            try:
                env.step({"action":"HideVisualizedPath"})
            except:
                pass # commit must be too early to use this commend. Reset scene in task sampler instead.
            path = event.third_party_camera_frames[0]
        else:
            # TODO: implement nominal action evolution to replace path. do it in step?
            pass
            # fig, ax = plt.subplots()
            # ax = plt.axes()
            # xs = [p["x"] for p in agent_path]
            # zs = [p["z"] for p in agent_path]
            # ax.plot(xs, zs, marker='o', color='g')
            # ax.set_title("Nominal agent path from origin/start")

        df = pd.read_csv(
            f"output/ac-data/{task.task_info['id']}.txt",
            names=list(task.class_action_names())+["EstimatedValue"],
        )
        try:
            ep_length = task._metrics["ep_length"]
        except:
            ForkedPdb().set_trace()

        # get returns from each step
        returns = []
        for r in reversed(task.task_info["rewards"]):
            if len(returns) == 0:
                returns.append(r)
            else:
                returns.append(r + returns[-1] * 0.99) # gamma value
        returns = returns[::-1]

        video_frames = []
        for step in range(task._metrics["ep_length"] + 1):
            is_first_frame = step == 0
            is_last_frame = step == task._metrics["ep_length"]

            agent_frame = np.array(
                Image.fromarray(task.observations[step]).resize((224, 224))
            )
            frame_number = step
            dist_to_target = task.task_info["dist_to_target"][step]

            if is_first_frame:
                last_action_success = None
                last_reward = None
                return_value = None
            else:
                last_action_success = task.task_info["action_successes"][step - 1]
                last_reward = task.task_info["rewards"][step - 1]
                return_value = returns[step - 1]

            if is_last_frame:
                action_dist = None
                critic_value = None
                taken_action = None
            else:
                policy_critic_value = df.iloc[step].values.tolist()
                action_dist = policy_critic_value[:len(task.class_action_names())] 
                critic_value = policy_critic_value[-1]

                taken_action = task.task_info["taken_actions"][step]

            video_frame = self.get_video_frame(
                agent_frame=agent_frame,
                frame_number=frame_number,
                action_names=task.class_action_names(),
                last_reward=(
                    round(last_reward, 2) if last_reward is not None else None
                ),
                critic_value=(
                    round(critic_value, 2) if critic_value is not None else None
                ),
                return_value=(
                    round(return_value, 2) if return_value is not None else None
                ),
                dist_to_target=round(dist_to_target, 2),
                action_dist=action_dist,
                ep_length=ep_length,
                last_action_success=last_action_success,
                taken_action=taken_action,
            )
            video_frames.append(video_frame)

        for _ in range(9):
            video_frames.append(video_frames[-1])

        os.makedirs(f"output/trajectories/{task.task_info['id']}", exist_ok=True)

        imsn = ImageSequenceClip([frame for frame in video_frames], fps=10)
        imsn.write_videofile(f"output/trajectories/{task.task_info['id']}/frames.mp4")

        # save the top-down path
        if task.distance_type != "realWorld":
            Image.fromarray(path).save(f"output/trajectories/{task.task_info['id']}/path.png")
        else:
            pass
            # fig.savefig(f"output/trajectories/{task.task_info['id']}/path.png")
            # path=np.array(Image.open(f"output/trajectories/{task.task_info['id']}/path.png")) # this is really dumb

        # save the value function over time
        fig, ax = plt.subplots()
        estimated_values = df.EstimatedValue.to_numpy()
        ax.plot(estimated_values, label="Critic Estimated Value")
        ax.plot(returns, label="Return")
        ax.set_ylabel("Value")
        ax.set_xlabel("Time Step")
        ax.set_title("Value Function over Time")
        ax.legend()
        fig.savefig(
            f"output/trajectories/{task.task_info['id']}/value_fn.svg",
            bbox_inches="tight",
        )
        plt.clf()

        task_out = {
                    "id": task.task_info["id"],
                    "spl": task._metrics["spl"],
                    "success": task._metrics["success"],
                    "reach_target": task._metrics["reach_target"],
                    "finalDistance": task.task_info["dist_to_target"][-1],
                    "initialDistance": task.task_info["dist_to_target"][0],
                    "minDistance": min(task.task_info["dist_to_target"]),
                    "episodeLength": task._metrics["ep_length"],
                    "confidence": (
                        None
                        if task.task_info["taken_actions"][-1] != "End"
                        else df.End.to_list()[-1]
                    ),
                    "failedActions": len(
                        [s for s in task.task_info["action_successes"] if not s]
                    ),
                    "targetObjectType": task.task_info["object_type"],
                    "numTargetObjects": len(task.task_info["target_object_ids"]),
                    "mirrored": task.task_info["mirrored"],
                    "scene": {
                        "name": task.task_info["house_name"],
                        "split": "train",
                        "rooms": 1,
                    },
                }
        task_out['movement_efficiency'] = compute_movement_efficiency(task.task_info)
        task_out['nominal_actions'] = task.task_info["taken_actions"]
        task_out['nominal_action_success'] = task.task_info["action_successes"]
        
        if 'number_of_interventions' in task.task_info:
            task_out['number_of_interventions'] = task.task_info['number_of_interventions']

        
        # if this is a mapping episode
        if hasattr(task, 'aggregate_map'):
            # extra metrics - work for blocky gt map
            task_out['map_coverage'] = task._metrics['map_coverage']
            task_out['map_exploration_efficiency'] = task._metrics['map_exploration_efficiency']
            task_out['percentage_rooms_visited'] = task._metrics['percentage_rooms_visited']
            task_out['room_visitation_efficiency'] = task._metrics['room_visitation_efficiency']
            task_out['seen_objects'] = task._metrics['seen_objects']
            task_out['new_object_rate'] = task._metrics['new_object_rate']

            # save the aggregate ground truth map
            combo = np.sum(task.aggregate_map[:,:,[1,2]],axis=2)
            obj_id_to_obj_pos = {
                o["objectId"]: o["axisAlignedBoundingBox"]["center"]
                for o in task.controller.last_event.metadata["objects"]
            }
            for object_id in task.task_info["target_object_ids"]:
                obj_pos = obj_id_to_obj_pos[object_id]
                row,col = centered_pixel_from_point([obj_pos['x'],obj_pos['z']],**task.map_scaling_params)
                combo[row-1:row+1,col-1:col+1]=3
            
            plt.clf()
            plt.imshow(combo)
            plt.savefig(f"output/trajectories/{task.task_info['id']}/agg_map.png")

            fname_map_estimate = f"output/mapping-data/{task.task_info['id']}.txt"
            if os.path.isfile(fname_map_estimate):

                fix_map_size = nn.AdaptiveAvgPool2d((40,40))
                sig_map_est = nn.Sigmoid()
                
                resized_walls = fix_map_size(torch.from_numpy(task.aggregate_map[:,:,1]).type(torch.FloatTensor).unsqueeze(0))# resize the walls once
                resized_walls = (resized_walls > 0.2)[0,:,:].float().numpy()
                gt_positions = fix_map_size(torch.from_numpy(task.aggregate_map[:,:,2]).type(torch.FloatTensor).unsqueeze(0))[0,:,:].numpy()

                # for every line in the mapping output save, turn it into an image
                map_df = pd.read_csv(fname_map_estimate,header=None)
                map_frames = []
                for step in range(task._metrics["ep_length"]):
                    current_loc = (agent_path[step]["x"], agent_path[step]["z"])
                    row,col = centered_pixel_from_point(current_loc,task.map_scaling_params['xyminmax'] , (40,40)) 
                    recon_map = reconstruct_resized_map_estimate(map_df.iloc[step].values.reshape((40,40)),
                                                                    resized_walls,gt_positions,(row,col),sig_map_est)
                    map_frames.append(self.get_mapping_video_frame(1-recon_map,step))
                            
                imsn = ImageSequenceClip([(frame) for frame in map_frames], fps=10)
                imsn.write_videofile(f"output/trajectories/{task.task_info['id']}/map.mp4")

        with open(f"output/trajectories/{task.task_info['id']}/data.json", "w") as f:
            json.dump(
                task_out,
                f,
            )
        
        return {
            "observations": task.observations,
            "path": [],#path,
            **task._metrics,
        }

    @staticmethod
    def get_mapping_video_frame(
        aggregate_map: np.ndarray,
        frame_number: int,
    ) -> np.array:
        agent_height, agent_width, ch = aggregate_map.shape
        font_to_use = "Arial.ttf" # possibly need a full path here
        full_font_load = ImageFont.truetype(font_to_use, 8)

        IMAGE_BORDER = 3
        image_dims = (agent_height + 2*IMAGE_BORDER,
                        agent_width + 2*IMAGE_BORDER,
                        ch)
        image = np.full(image_dims, 255, dtype=np.uint8)

        image[
            IMAGE_BORDER : IMAGE_BORDER + agent_height, IMAGE_BORDER : IMAGE_BORDER + agent_width, :
        ] = aggregate_map*255

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)

        img_draw.text(
            (IMAGE_BORDER * 1.1, IMAGE_BORDER * 1),
            str(frame_number),
            font=full_font_load,#ImageFont.truetype(font_to_use, 25),
            fill="black",
        )

        return np.array(text_image)
    
    @staticmethod
    def get_video_frame(
        agent_frame: np.ndarray,
        frame_number: int,
        action_names: List[str],
        last_reward: Optional[float],
        critic_value: Optional[float],
        return_value: Optional[float],
        dist_to_target: float,
        action_dist: Optional[List[float]],
        ep_length: int,
        last_action_success: Optional[bool],
        taken_action: Optional[str],
    ) -> np.array:
        
        agent_height, agent_width, ch = agent_frame.shape

        font_to_use = "Arial.ttf" # possibly need a full path here
        full_font_load = ImageFont.truetype(font_to_use, 14)

        IMAGE_BORDER = 25
        TEXT_OFFSET_H = 60
        TEXT_OFFSET_V = 30

        image_dims = (agent_height + 2*IMAGE_BORDER +  30,
                        agent_width + 2*IMAGE_BORDER + 200,
                        ch)
        image = np.full(image_dims, 255, dtype=np.uint8)

        
        image[
            IMAGE_BORDER : IMAGE_BORDER + agent_height, IMAGE_BORDER : IMAGE_BORDER + agent_width, :
        ] = agent_frame

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)
        # font size 25, aligned center and middle
        if action_dist is not None:
            for i, (prob, action) in enumerate(
                zip(
                    action_dist,action_names
                )
            ):
                img_draw.text(
                    (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, (TEXT_OFFSET_V+5) + i * 20),
                    action,
                    font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                    fill="gray" if action != taken_action else "black",
                    anchor="rm",
                )
                img_draw.rectangle(
                    (
                        IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H+5),
                        TEXT_OFFSET_V + i * 20,
                        IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H+5) + int(100 * prob),
                        (TEXT_OFFSET_V+10) + i * 20,
                    ),
                    outline="blue",
                    fill="blue",
                )

        img_draw.text(
            (IMAGE_BORDER * 1.1, IMAGE_BORDER * 1),
            str(frame_number),
            font=full_font_load,#ImageFont.truetype(font_to_use, 25),
            fill="white",
        )

        oset = -10
        if last_reward is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 175 + oset),
                "Last Reward:",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 175 + oset),
                " " + ("+" if last_reward > 0 else "") + str(last_reward),
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="lm",
            )

        oset = 10
        if critic_value is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 175 + oset),
                "Critic Value:",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 175 + oset),
                " " + ("+" if critic_value > 0 else "") + str(critic_value),
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="lm",
            )

        if return_value is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 195 + oset),
                "Return:",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 195 + oset),
                " " + ("+" if return_value > 0 else "") + str(return_value),
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="lm",
            )

        if last_action_success is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 235),
                "Last Action:",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 235),
                " Success" if last_action_success else " Failure",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="green" if last_action_success else "red",
                anchor="lm",
            )

        if taken_action == "manual override":
                    img_draw.text(
            (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H + 50, TEXT_OFFSET_V + 5 * 20),
            "Manual Override",
            font=full_font_load,#ImageFont.truetype(font_to_use, 14),
            fill="red",
            anchor="rm",
        )

        img_draw.text(
            (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 145),
            "Target Dist:",
            font=full_font_load,#ImageFont.truetype(font_to_use, 14),
            fill="gray",
            anchor="rm",
        )
        img_draw.text(
            (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 145),
            f" {dist_to_target}m",
            font=full_font_load,#ImageFont.truetype(font_to_use, 14),
            fill="gray",
            anchor="lm",
        )

        lower_offset = 10
        progress_bar_height = 20

        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + agent_width,
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline="lightgray",
            fill="lightgray",
        )
        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + int(frame_number * agent_width / ep_length),
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline="blue",
            fill="blue",
        )

        return np.array(text_image)
