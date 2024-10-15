import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import Literal

import numpy as np
import wandb
from allenact.base_abstractions.callbacks import Callback
from allenact.base_abstractions.sensor import Sensor
import gym

from training.callbacks.callback_sensors import LocalLoggingSensor
from training.tasks.object_nav import ObjectNavTask, ForkedPdb



class LocalLogging(Callback):
    def __init__(self):
        # NOTE: Makes it more statistically meaningful
        self.aggregate_by_means_across_n_runs: int = 10
        self.by_means_iter: int = 0
        self.by_metrics = dict()
    
    def callback_sensors(self) -> Optional[Sequence[Sensor]]:
        """Determines the data returned to the `tasks_data` parameter in the
        above *_log functions."""
        return [LocalLoggingSensor(uuid="local_logging_callback_sensor",
                                   observation_space=gym.spaces.Discrete(1)),]

    @staticmethod
    def get_columns(task: Dict[str, Any]) -> List[str]:
        """Get the columns of the quantitative table."""
        types = int, float, str, bool, wandb.Image, wandb.Video

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
        if len(tasks_data) == 0:
            return wandb.Table()

        data = []
        columns = LocalLogging.get_columns(tasks_data[0])
        columns.insert(0, "step")
        columns.insert(0, "path")
        columns.insert(0, "observations")

        for task in tasks_data:
            frames = task["observations"]
            frames_with_progress = []

            # NOTE: add progress bars
            for i, frame in enumerate(frames):
                # NOTE: flip the images if the task is mirrored
                if "mirrored" in task["task_info"] and task["task_info"]["mirrored"]:
                    frame = np.fliplr(frame)
                BORDER_SIZE = 15

                frame_with_progress = np.full(
                    (
                        frame.shape[0] + 50 + BORDER_SIZE * 2,
                        frame.shape[1] + BORDER_SIZE * 2,
                        frame.shape[2],
                    ),
                    fill_value=255,
                    dtype=np.uint8,
                )

                # NOTE: add border for action failures
                if i > 1 and not task["task_info"]["action_successes"][i - 1]:
                    frame_with_progress[0 : BORDER_SIZE * 2 + frame.shape[0]] = (
                        255,
                        0,
                        0,
                    )

                # NOTE: add the agent image
                frame_with_progress[
                    BORDER_SIZE : BORDER_SIZE + frame.shape[0],
                    BORDER_SIZE : BORDER_SIZE + frame.shape[1],
                ] = frame

                # NOTE: add the progress bar
                progress_bar = frame_with_progress[-35:-15, BORDER_SIZE:-BORDER_SIZE]
                progress_bar[:] = (225, 225, 225)
                if len(frames) > 1:
                    num_progress_pixels = int(
                        progress_bar.shape[1] * i / (len(frames) - 1)
                    )
                    progress_bar[:, :num_progress_pixels] = (38, 94, 212)

                frames_with_progress.append(frame_with_progress)

            frames = np.stack(frames_with_progress, axis=0)
            frames = np.moveaxis(frames, [1, 2, 3], [2, 3, 1])
            trajectory = wandb.Video(frames, fps=5, format="mp4")

            entry = []
            for column in columns:
                if column == "observations":
                    entry.append(trajectory)
                elif column == "step":
                    entry.append(step)
                elif column == "path":
                    entry.append(wandb.Image(task["path"]))
                elif column.startswith("task_info/"):
                    entry.append(task["task_info"][column[len("task_info/") :]])
                else:
                    entry.append(task[column])

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

            for k in (by_rooms_key, by_obj_type_key):
                if k not in self.by_metrics:
                    self.by_metrics[k] = {
                        "means": {
                            "reward": 0,
                            "ep_length": 0,
                            "success": 0,
                            "spl": 0,
                            "dist_to_target": 0,
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

    @staticmethod
    def get_metrics_table(tasks: List[Any]) -> wandb.Table:
        """Get the metrics table."""
        columns = LocalLogging.get_columns(tasks[0])
        data = []
        for task in tasks:
            entry = []
            for column in columns:
                if column.startswith("task_info/"):
                    entry.append(task["task_info"][column[len("task_info/") :]])
                else:
                    entry.append(task[column])
            data.append(entry)

        columns = [
            c[len("task_info/") :] if c.startswith("task_info/") else c for c in columns
        ]
        return wandb.Table(data=data, columns=columns)

    @staticmethod
    def get_metric_plots(
        metrics: Dict[str, Any], split: Literal["valid", "test"], step: int
    ) -> Dict[str, Any]:
        """Get the metric plots."""
        plots = {}
        table = LocalLogging.get_metrics_table(metrics["tasks"])

        # NOTE: Log difficulty SPL and success rate
        if "difficulty" in metrics["tasks"][0]["task_info"]:
            plots[f"{split}-success-by-difficulty-{step:012}"] = wandb.plot.bar(
                table,
                "difficulty",
                "success",
                title=f"{split} Success by Difficulty ({step:,} steps)",
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
            plots[f"{split}-spl-by-object-type-{step:012}"] = wandb.plot.bar(
                table,
                "object_type",
                "spl",
                title=f"{split} SPL by Object Type ({step:,} steps)",
            )

        return plots

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
        plots = (
            self.get_metric_plots(metrics=metrics, split="valid", step=step)
            if metrics
            else {}
        )
        table = self.get_quantitative_table(tasks_data=tasks_data, step=step)
        val_table = (
            {f"valid-quantitative-examples/{step:012}": table} if table.data else {}
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
        trajectories = []
        for filename in glob.glob("output/trajectories/*/data.json"):
            with open(filename, "r") as f:
                trajectories.append(json.load(f))
            # os.remove(filename) # don't delete

        with open("output/trajectories/data.json", "w") as f:
            json.dump(trajectories, f, indent=2, sort_keys=True)

        sum_trajectories = [{'total_success':sum(i['success'] is True for i in trajectories)/len(trajectories)}]
        for scene in set([i["scene"]["name"] for i in trajectories]):
            scene_traj = [i for i in trajectories if i["scene"]["name"]==scene]
            sum_trajectories.append({scene : sum(i['success'] is True for i in scene_traj)/len(scene_traj)})
        for obj in set([i["targetObjectType"] for i in trajectories]):
            obj_traj = [i for i in trajectories if i["targetObjectType"]==obj]
            # for p in set([d['id'].split('_starting_')[1].split('_sampler_')[0] for d in trajectories]):
            #     p_traj = [i for i in obj_traj if i['id'].split('_starting_')[1].split('_sampler_')[0]==scene]
            #     print(obj)
            #     print(p)
            #     print(str(sum(i['success'] is True for i in p_traj)/len(p_traj)))
                
            sum_trajectories.append({obj : sum(i['success'] is True for i in obj_traj)/len(obj_traj)})
        
        for d in trajectories:
            x={}
            x['target']=d['targetObjectType']
            x['start'] = d['id'].split('_starting_')[1].split('_sampler_')[0]
            x['length'] = d['episodeLength']
            x['success'] = d['success']
            x['bumps'] = d['failedActions'] # this is just for sim trajectories
            x['movement_efficiency'] = d['movement_efficiency']
            x['scene'] = d["scene"]["name"]
            if 'map_coverage' in d:
                x['map_coverage'] = d['map_coverage']
                x['map_exploration_efficiency'] = d['map_exploration_efficiency']
                x['percentage_rooms_visited'] = d['percentage_rooms_visited']

            sum_trajectories.append(x)

        
        with open("output/trajectories/summary_data.json", "w") as f:
            json.dump(sum_trajectories, f, indent=2,sort_keys=True)

