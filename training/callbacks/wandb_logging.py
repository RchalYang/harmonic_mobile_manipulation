from typing import Any, Dict, List, Sequence, Optional

from allenact.base_abstractions.sensor import Sensor
from training.callbacks.callback_sensors import ProcTHORObjectNavCallbackSensor

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import os
from collections import defaultdict

import numpy as np
import wandb
from allenact.base_abstractions.callbacks import Callback
from training import cfg
from utils.utils import ForkedPdb
from utils.map_utils import build_centered_map, centered_pixel_from_point

from allenact.utils.system import get_logger

WANDB_SENSOR_UUID = "WANDB_CALLBACK_SENSOR"

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

class WandbLogging(Callback):
    def __init__(self):
        # NOTE: Makes it more statistically meaningful
        self.aggregate_by_means_across_n_runs: int = 10
        self.by_means_iter: int = 0
        self.by_metrics = dict()

        self.val_aggregate_by_means_across_n_runs: int = 1
        self.val_by_means_iter: int = 0
        self.val_by_metrics = dict()

    def setup(self, name: str, **kwargs) -> None:
        name_with_seed = f"{name if cfg.wandb.name is None else cfg.wandb.name}_{cfg.seed}"
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=name_with_seed,
            group=name if cfg.wandb.name is None else cfg.wandb.name,
            config=kwargs,
        )

    @staticmethod
    def get_columns(tasks: List[Dict[str, Any]]) -> List[str]:
        """Get the columns of the quantitative table."""
        types = int, float, str, bool, wandb.Image, wandb.Video
        task = tasks[np.argmax([len(t.keys()) for t in tasks])] # nb set of accumulation doesn't preserve order

        columns = []
        for key in task.keys():
            if isinstance(task[key], types):
                columns.append(key)

        for key in task["task_info"]:
            if isinstance(task["task_info"][key], types):
                columns.append(f"task_info/{key}")
        return columns

    @staticmethod
    def get_quantitative_table(tasks_data: List[Any], step: int) -> wandb.Table:
        """Get the quantitative table."""
        tasks_data = [td[WANDB_SENSOR_UUID] for td in tasks_data if td[WANDB_SENSOR_UUID] is not None]

        if len(tasks_data) == 0:
            return wandb.Table()

        data = []
        columns = WandbLogging.get_columns(tasks_data)
        columns.insert(0, "step")
        columns.insert(0, "target_map")
        columns.insert(0, "path")
        columns.insert(0, "observations")
        if "map_observations" in tasks_data[0]:
            columns.insert(0,"map_video")
        
        font_to_use = "Arial.ttf"  # possibly need a full path here
        full_font_load = ImageFont.truetype(font_to_use, 14)
        
        for task in tasks_data:
            frames = task["observations"][:-1]
            actions = task["task_info"]["taken_actions"]
            if "openness" in task["task_info"]:
                opennesses = task["task_info"]["openness"]
            rewards = task["task_info"]["rewards"]
            frames_with_progress = []
            map_frames = []

            # mapping
            base_map, xyminmax, pixel_sizes = build_centered_map(task["house"])
            viz_map = np.sum(base_map[:,:,[1,2]],axis=2)/2 # leave this step in case of trajectory as well later

            # NOTE: add progress bars
            for i, frame in enumerate(frames):
                # NOTE: flip the images if the task is mirrored
                if "mirrored" in task["task_info"] and task["task_info"]["mirrored"]:
                    frame = np.fliplr(frame)
                BORDER_SIZE = 15
                ACTION_REGION_SIZE = 350

                TEXT_MARGIN_H = 10
                TEXT_MARGIN_W = 10

                frame_with_progress = np.full(
                    (
                        frame.shape[0] + 50 + BORDER_SIZE * 2,
                        frame.shape[1] + BORDER_SIZE * 2 + ACTION_REGION_SIZE + TEXT_MARGIN_W,
                        frame.shape[2],
                    ),
                    fill_value=255,
                    dtype=np.uint8,
                )

                # NOTE: add the agent image
                frame_with_progress[
                    BORDER_SIZE : BORDER_SIZE + frame.shape[0],
                    BORDER_SIZE : BORDER_SIZE + frame.shape[1],
                ] = frame

                text_image = Image.fromarray(frame_with_progress)
                img_draw = ImageDraw.Draw(text_image)
                        
                TEXT_OFFSET_H = 30
                TEXT_OFFSET_W = 120

                BAR_OFFSET_W = 120

                taken_action = actions[i]
                
                action_names = ["move", "rotate", "arm_y", "arm_z", "wrist"]
                action_names = action_names[:len(taken_action)]
                for j in range(len(action_names)):
                    img_draw.text(
                        (
                            BORDER_SIZE * 2 + frame.shape[1] + TEXT_OFFSET_W,
                            (TEXT_OFFSET_H + 5) + j * 30,
                        ),
                        action_names[j],
                        font=ImageFont.truetype(font_to_use, 30),
                        fill="black",
                        anchor="rm",
                    )
                    if taken_action[j] > 0:
                        img_draw.rectangle(
                            (
                                BORDER_SIZE * 2 + BAR_OFFSET_W + frame.shape[1] + (TEXT_OFFSET_W + 5),
                                TEXT_OFFSET_H + j * 30,
                                BORDER_SIZE * 2 + BAR_OFFSET_W + frame.shape[1] + (TEXT_OFFSET_W + 5) + int(100 * taken_action[j]),
                                TEXT_OFFSET_H + 20 + j * 30,
                            ),
                            outline="blue",
                            fill="blue",
                        )
                    elif taken_action[j] < 0:
                        img_draw.rectangle(
                            (
                                BORDER_SIZE * 2 + BAR_OFFSET_W + frame.shape[1] + (TEXT_OFFSET_W + 5) + int(100 * taken_action[j]),
                                TEXT_OFFSET_H + j * 30,
                                BORDER_SIZE * 2 + BAR_OFFSET_W + frame.shape[1] + (TEXT_OFFSET_W + 5),
                                TEXT_OFFSET_H + 20 + j * 30,
                            ),
                            outline="green",
                            fill="green",
                        )

                if "openness" in task["task_info"]:
                    img_draw.text(
                        (
                            BORDER_SIZE * 2 + frame.shape[1] + TEXT_OFFSET_W,
                            (TEXT_OFFSET_H + 5) + len(action_names) * 30,
                        ),
                        "Openness",
                        font=ImageFont.truetype(font_to_use, 30),
                        fill="black",
                        anchor="rm",
                    )
                
                    openness = opennesses[i]
                    img_draw.text(
                        (
                            BORDER_SIZE * 2 + frame.shape[1] + TEXT_OFFSET_W + 100,
                            (TEXT_OFFSET_H + 5) + len(action_names) * 30,
                        ),
                        str(round(openness, 4)),
                        font=ImageFont.truetype(font_to_use, 30),
                        fill="black",
                        anchor="rm",
                    )

                img_draw.text(
                    (
                        BORDER_SIZE * 2 + frame.shape[1] + TEXT_OFFSET_W,
                        (TEXT_OFFSET_H + 5) + len(action_names) * 30 + 30,
                    ),
                    "Reward",
                    font=ImageFont.truetype(font_to_use, 30),
                    fill="black",
                    anchor="rm",
                )
            
                reward = rewards[i]
                img_draw.text(
                    (
                        BORDER_SIZE * 2 + frame.shape[1] + TEXT_OFFSET_W + 100,
                        (TEXT_OFFSET_H + 5) + len(action_names) * 30 + 30,
                    ),
                    str(round(reward, 4)),
                    font=ImageFont.truetype(font_to_use, 30),
                    fill="black",
                    anchor="rm",
                )

                frame_with_progress = np.array(text_image)
                # NOTE: add the progress bar
                progress_bar = frame_with_progress[-35:-15, BORDER_SIZE:-BORDER_SIZE]
                progress_bar[:] = (225, 225, 225)
                if len(frames) > 1:
                    num_progress_pixels = int(
                        progress_bar.shape[1] * i / (len(frames) - 1)
                    )
                    progress_bar[:, :num_progress_pixels] = (38, 94, 212)

                frames_with_progress.append(frame_with_progress)

                # mapping
                # ForkedPdb().set_trace()
                if "map_observations" in task:
                    curr_agg_map = np.clip(np.sum(task["map_observations"][i],axis=2)/3,0,1)*255.0
                    for obj_pos in task["task_info"]["target_locations"]:
                        row,col = centered_pixel_from_point([obj_pos['x'],obj_pos['z']],xyminmax, pixel_sizes)
                        curr_agg_map[row-1:row+1,col-1:col+1]=255
                    map_frames.append(curr_agg_map)


            frames = np.stack(frames_with_progress, axis=0)
            frames = np.moveaxis(frames, [1, 2, 3], [2, 3, 1])
            trajectory = wandb.Video(frames, fps=5, format="mp4")

            if "map_observations" in task:
                stacked_map_frames = np.stack(map_frames, axis=0)
                stacked_map_frames = np.repeat(np.expand_dims(stacked_map_frames,axis=1),3,axis=1).astype(np.uint8)
                # stacked_map_frames = np.moveaxis(stacked_map_frames, [1, 2, 3], [2, 3, 1])
                map_video = wandb.Video(stacked_map_frames, fps=20, format="gif")


            
            for obj_pos in task["task_info"]["target_locations"]:
                row,col = centered_pixel_from_point([obj_pos['x'],obj_pos['z']],xyminmax, pixel_sizes)
                viz_map[row-1:row+1,col-1:col+1]=3

            entry = []
            for column in columns:
                if column == "observations":
                    entry.append(trajectory)
                elif column == "map_video":
                    entry.append(map_video)
                elif column == "step":
                    entry.append(step)
                elif column == "path":
                    entry.append(wandb.Image(task["path"]))
                elif column == "target_map":
                    entry.append(wandb.Image(np.clip(viz_map,0,1)))
                elif column.startswith("task_info/"):
                    entry.append(task["task_info"][column[len("task_info/") :]])
                else:
                    try:
                        entry.append(task[column])
                    except:
                        entry.append(None)

            data.append(entry)

        # clean up column names
        columns = [
            c[len("task_info/") :] if c.startswith("task_info/") else c for c in columns
        ]

        return wandb.Table(data=data, columns=columns)

    def on_train_log(
        self,
        metrics: List[Dict[str, Any]],
        metric_means: Dict[str, float],
        step: int,
        tasks_data: List[Any],
        **kwargs,
    ) -> None:
        """Log the train metrics to wandb."""
        table = self.get_quantitative_table(tasks_data=tasks_data, step=step)
        quantitative_table = (
            {f"train-quantitative-examples/{step:012}": table} if table.data else {}
        )

        for episode in metrics:
            by_rooms_key = (
                f"train-metrics-by-rooms/{episode['task_info']['rooms']}-rooms"
            )
            by_obj_type_key = (
                f"train-metrics-by-obj-type/{episode['task_info']['object_type']}"
            )

            by_dist_key = (
                f"train-metrics-by-dist-cat/{episode['task_info']['init_dist_category']}"
            )

            for k in (by_rooms_key, by_obj_type_key, by_dist_key):
                if k not in self.by_metrics:
                    self.by_metrics[k] = {
                        "means": {
                            "reward": 0,
                            "ep_length": 0,
                            "success": 0,
                            "reach_target": 0,
                            "spl": 0,
                            "dist_to_target": 0,
                        },
                        "count": 0,
                    }
                if "opening_type" in episode['task_info']:
                    self.by_metrics[k]["means"]["max_openness"] = 0
                self.by_metrics[k]["count"] += 1
                for metric in self.by_metrics[k]["means"]:
                    old_mean = self.by_metrics[k]["means"][metric]
                    self.by_metrics[k]["means"][metric] = (
                        old_mean
                        + (episode[metric] - old_mean) / self.by_metrics[k]["count"]
                    )

            if "opening_type" in episode['task_info']:
                by_opening_type_key = (
                    f"train-metrics-by-opening-type/{episode['task_info']['opening_type']}"
                )

                # for k in (by_rooms_key, by_obj_type_key):
                k = by_opening_type_key
                if k not in self.by_metrics:
                    self.by_metrics[k] = {
                        "means": {
                            "reward": 0,
                            "ep_length": 0,
                            "success": 0,
                            "reach_target": 0,
                            "spl": 0,
                            "dist_to_target": 0,
                            "max_openness": 0,
                        },
                        "count": 0,
                    }
                self.by_metrics[k]["count"] += 1
                for metric in self.by_metrics[k]["means"]:
                    old_mean = self.by_metrics[k]["means"][metric]
                    self.by_metrics[k]["means"][metric] = (
                        old_mean
                        + (episode[metric] - old_mean) / self.by_metrics[k]["count"]
                    )            

        by_means_dict = {}
        self.by_means_iter += 1
        if self.by_means_iter % self.aggregate_by_means_across_n_runs == 0:
            # NOTE: log by means
            for metric, info in self.by_metrics.items():
                for mean_key, mean in info["means"].items():
                    key = f"/{mean_key}-".join(metric.split("/"))
                    by_means_dict[key] = mean
            # NOTE: reset the means
            self.by_metrics = dict()

        wandb.log(
            {**metric_means, **by_means_dict, **quantitative_table, "step": step,}
        )

    @staticmethod
    def get_metrics_table(tasks: List[Any]) -> wandb.Table:
        """Get the metrics table."""
        columns = WandbLogging.get_columns(tasks)
        data = []
        for task in tasks:
            entry = []
            for column in columns:
                if column.startswith("task_info/"):
                    entry.append(task["task_info"][column[len("task_info/") :]])
                else:
                    try:
                        entry.append(task[column])
                    except:
                        entry.append(None) # fail case metrics absent
            data.append(entry)

        columns = [
            c[len("task_info/") :] if c.startswith("task_info/") else c for c in columns
        ]
        return wandb.Table(data=data, columns=columns)

    # @staticmethod
    def get_metric_plots(self,
        metrics: Dict[str, Any], split: Literal["valid", "test"], step: int
    ) -> Dict[str, Any]:
        """Get the metric plots."""
        plots = {}
        table = WandbLogging.get_metrics_table(metrics["tasks"])

        # NOTE: Log difficulty SPL and success rate
        if "difficulty" in metrics["tasks"][0]["task_info"]:
            plots[f"{split}-success-by-difficulty-{step:012}"] = wandb.plot.bar(
                table,
                "difficulty",
                "success",
                title=f"{split} Success by Difficulty ({step:,} steps)",
            )
            plots[f"{split}-Reach-Target-by-difficulty-{step:012}"] = wandb.plot.bar(
                table,
                "difficulty",
                "reach_target",
                title=f"{split} Reach Target by Difficulty ({step:,} steps)",
            )
            plots[f"{split}-spl-by-difficulty-{step:012}"] = wandb.plot.bar(
                table,
                "difficulty",
                "spl",
                title=f"{split} SPL by Difficulty ({step:,} steps)",
            )

        # NOTE: Log object type SPL and success rate
        if "object_type" in metrics["tasks"][0]["task_info"]:
            plots[f"{split}-success-by-object-type-{step:012}"] = wandb.plot.bar(
                table,
                "object_type",
                "success",
                title=f"{split} Success by Object Type ({step:,} steps)",
            )
            plots[f"{split}-Reach-Target-by-object-type-{step:012}"] = wandb.plot.bar(
                table,
                "object_type",
                "reach_target",
                title=f"{split} Reach Target by Difficulty ({step:,} steps)",
            )
            plots[f"{split}-spl-by-object-type-{step:012}"] = wandb.plot.bar(
                table,
                "object_type",
                "spl",
                title=f"{split} SPL by Object Type ({step:,} steps)",
            )
            
            for task in metrics["tasks"]:
                by_obj_type_key = (
                    f"val-metrics-by-obj-type/{task['task_info']['object_type']}"
                )

                for k in [by_obj_type_key]:
                    if k not in self.val_by_metrics:
                        self.val_by_metrics[k] = {
                            "means": {
                                "reward": 0,
                                "ep_length": 0,
                                "success": 0,
                                "reach_target": 0,
                                "spl": 0,
                                "dist_to_target": 0,
                                # "map_coverage": 0,
                            },
                            "count": 0,
                        }
                    self.val_by_metrics[k]["count"] += 1
                    for metric in self.val_by_metrics[k]["means"]:
                        old_mean = self.val_by_metrics[k]["means"][metric]
                        self.val_by_metrics[k]["means"][metric] = (
                            old_mean
                            + (task[metric] - old_mean) / self.val_by_metrics[k]["count"]
                        )

            by_means_dict = {}
            self.val_by_means_iter += 1
            if self.val_by_means_iter % self.val_aggregate_by_means_across_n_runs == 0:
                # NOTE: log by means
                for metric, info in self.val_by_metrics.items():
                    for mean_key, mean in info["means"].items():
                        key = f"/{mean_key}-".join(metric.split("/"))
                        by_means_dict[key] = mean
                # NOTE: reset the means
                self.val_by_metrics = dict()


        # NOTE: Log object type SPL and success rate
        if "init_dist_category" in metrics["tasks"][0]["task_info"]:
            plots[f"{split}-success-by-init-dist-to-target-{step:012}"] = wandb.plot.bar(
                table,
                "init_dist_category",
                "success",
                title=f"{split} Success by Init Dist To Target ({step:,} steps)",
            )
            plots[f"{split}-Reach-Target-by-init-dist-to-target-{step:012}"] = wandb.plot.bar(
                table,
                "init_dist_category",
                "reach_target",
                title=f"{split} Reach Target by Init Dist To Target ({step:,} steps)",
            )
            plots[f"{split}-spl-by-object-type-{step:012}"] = wandb.plot.bar(
                table,
                "init_dist_category",
                "spl",
                title=f"{split} SPL by Init Dist To Target ({step:,} steps)",
            )
            
            for task in metrics["tasks"]:
                # by_obj_t_key = (
                #     f"val-metrics-by-init-dist-to-target/{task['task_info']['object_type']}"
                # )
                by_dist_key = (
                    f"val-metrics-by-dist-cat/{task['task_info']['init_dist_category']}"
                )

                for k in [by_dist_key]:
                    if k not in self.val_by_metrics:
                        self.val_by_metrics[k] = {
                            "means": {
                                "reward": 0,
                                "ep_length": 0,
                                "success": 0,
                                "reach_target": 0,
                                "spl": 0,
                                "dist_to_target": 0,
                                # "map_coverage": 0,
                            },
                            "count": 0,
                        }
                    self.val_by_metrics[k]["count"] += 1
                    for metric in self.val_by_metrics[k]["means"]:
                        old_mean = self.val_by_metrics[k]["means"][metric]
                        self.val_by_metrics[k]["means"][metric] = (
                            old_mean
                            + (task[metric] - old_mean) / self.val_by_metrics[k]["count"]
                        )

            by_means_dict = {}
            self.val_by_means_iter += 1
            if self.val_by_means_iter % self.val_aggregate_by_means_across_n_runs == 0:
                # NOTE: log by means
                for metric, info in self.val_by_metrics.items():
                    for mean_key, mean in info["means"].items():
                        key = f"/{mean_key}-".join(metric.split("/"))
                        by_means_dict[key] = mean
                # NOTE: reset the means
                self.val_by_metrics = dict()

        return plots, by_means_dict

    @staticmethod
    def get_by_scene_dataset_log(
        metrics: Dict[str, Any], split: Literal["train", "val", "test"]
    ) -> Dict[str, float]:
        by_scene_data = defaultdict(
            lambda: {
                "count": 0,
                "means": {
                    "reward": 0,
                    "ep_length": 0,
                    "success": 0,
                    "reach_target": 0,
                    "spl": 0,
                    "dist_to_target": 0,
                },
            }
        )
        if (
            len(metrics["tasks"]) > 0
            and "sceneDataset" in metrics["tasks"][0]["task_info"]
        ):
            for task in metrics["tasks"]:
                scene_dataset = task["task_info"]["sceneDataset"]
                by_scene_data[scene_dataset]["count"] += 1
                for key in by_scene_data[scene_dataset]["means"]:
                    old_mean = by_scene_data[scene_dataset]["means"][key]
                    value = float(task[key])
                    by_scene_data[scene_dataset]["means"][key] = (
                        old_mean
                        + (value - old_mean) / by_scene_data[scene_dataset]["count"]
                    )
        by_scene_data_log = {}
        for scene_dataset in by_scene_data:
            for mean, value in by_scene_data[scene_dataset]["means"].items():
                by_scene_data_log[f"{split}-{scene_dataset}/{mean}"] = value

        return by_scene_data_log

    def on_valid_log(
        self,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        checkpoint_file_name: str,
        tasks_data: List[Any],
        step: int,
        **kwargs,
    ) -> None:
        """Log the validation metrics to wandb."""

        by_scene_dataset_log = WandbLogging.get_by_scene_dataset_log(
            metrics, split="val"
        )
        plots, by_means_dict = (
            self.get_metric_plots(metrics=metrics, split="valid", step=step)
            if metrics
            else {}
        )
        table = self.get_quantitative_table(tasks_data=tasks_data, step=step)
        val_table = (
            {f"valid-quantitative-examples/{step:012}": table} if table.data else {}
        )
        wandb.save(checkpoint_file_name)
        wandb.log(
            {**metric_means, **by_means_dict, **plots, **by_scene_dataset_log, "step": step, **val_table}
        )

    def on_test_log(
        self,
        checkpoint_file_name: str,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        step: int,
        **kwargs,
    ) -> None:
        """Log the test metrics to wandb."""

        by_scene_dataset_log = WandbLogging.get_by_scene_dataset_log(
            metrics, split="test"
        )
        plots, by_mean_dict = self.get_metric_plots(metrics=metrics, split="test", step=step) if metrics else ({}, {})
        table = self.get_quantitative_table(tasks_data=tasks_data, step=step)
        get_logger().warning(metric_means)
        get_logger().warning(plots)
        get_logger().warning(by_scene_dataset_log)
        wandb.log(
            {
                **metric_means,
                **plots,
                **by_scene_dataset_log,
                "step": step,
                f"test-quantitative-examples/{step:012}": table,
            }
        )

    def after_save_project_state(self, base_dir: str) -> None:
        wandb.save(os.path.join(base_dir, "*"))
        wandb.save(os.path.expanduser("~/.hydra/config.yaml"))

    def callback_sensors(self) -> Optional[Sequence[Sensor]]:
        return [ProcTHORObjectNavCallbackSensor(uuid=WANDB_SENSOR_UUID)]
