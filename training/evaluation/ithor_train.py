"""
Setup for evaluation on the hand designed generated houses.
"""

from training.tasks.object_nav import ObjectNavTestTaskSampler, TrainiTHORTaskSampler

from .full_minival import ObjectNavEvalConfig


class ObjectNaviTHORTrainConfig(ObjectNavEvalConfig):
    @classmethod
    def tag(cls):
        return super().tag() + "-iTHOR-Train"

    def make_sampler_fn(self, task_sampler_args, **kwargs):
        if task_sampler_args.mode == "eval":
            return ObjectNavTestTaskSampler(args=task_sampler_args)
        else:
            return TrainiTHORTaskSampler(args=task_sampler_args)

    def train_task_sampler_args(self, **kwargs):
        kitchens = [f"FloorPlan{i}" for i in range(1, 21)]
        living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 21)]
        bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 21)]
        bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 21)]

        scenes = kitchens + living_rooms + bedrooms + bathrooms
        out = self._get_sampler_args_for_scene_split(
            houses=scenes,
            allow_oversample=False,
            max_tasks=float("inf"),
            allow_flipping=True,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,  # ignored
            mode="train",
            **kwargs,
        )
        return {"task_sampler_args": out}
