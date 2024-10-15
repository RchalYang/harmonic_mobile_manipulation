
from training.experiments.rgb_clipresnet50gru_ddppo import (
    ObjectNavRGBClipResNet50PPOExperimentConfig,
)
from training.tasks.object_nav import ObjectNavRealTask, RealObjectNavTaskSampler

class ObjectNavCurrentRealOnly(ObjectNavRGBClipResNet50PPOExperimentConfig):

    OBJECT_NAV_TASK_TYPE = ObjectNavRealTask
    @classmethod
    def tag(cls):
        return super().tag() + "-real_evaluation"

    def make_sampler_fn(self, task_sampler_args, **kwargs):
        return RealObjectNavTaskSampler(args=task_sampler_args)