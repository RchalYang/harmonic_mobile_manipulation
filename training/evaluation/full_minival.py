"""
Setup for evaluation on the full set of evaluation tasks.
"""

import prior

from training import cfg
from training.experiments.rgb_clipresnet50gru_ddppo import (
    ObjectNavRGBClipResNet50PPOExperimentConfig,
)
from training.tasks.object_nav import FullObjectNavTestTaskSampler, ObjectNavTaskSampler


class ObjectNavEvalConfig(ObjectNavRGBClipResNet50PPOExperimentConfig):
    EVAL_TASKS = prior.load_dataset("object-nav-eval", minival=True)

    @classmethod
    def tag(cls):
        return super().tag() + "-All-THOR-Target"

    def make_sampler_fn(self, task_sampler_args, **kwargs):
        if task_sampler_args.mode == "eval":
            return FullObjectNavTestTaskSampler(args=task_sampler_args)
        else:
            return ObjectNavTaskSampler(args=task_sampler_args)

    def valid_task_sampler_args(self, **kwargs):
        out = self._get_sampler_args_for_scene_split(
            houses=self.EVAL_TASKS["val"],
            allow_oversample=False,
            max_tasks=cfg.evaluation.max_val_tasks,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,  # ignored
            mode="eval",
            **kwargs,
        )
        return {"task_sampler_args": out}

    def test_task_sampler_args(self, **kwargs):
        if self.TEST_ON_VALIDATION:
            return self.valid_task_sampler_args(**kwargs)

        out = self._get_sampler_args_for_scene_split(
            houses=self.EVAL_TASKS["test"],
            allow_oversample=False,
            max_tasks=cfg.evaluation.max_test_tasks,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,  # ignored
            mode="eval",
            **kwargs,
        )
        return {"task_sampler_args": out}
