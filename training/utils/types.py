from typing import Any, Dict, List, Optional

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

import datasets
from allenact.base_abstractions.sensor import Sensor
from attrs import define


class Vector3(TypedDict):
    x: float
    y: float
    z: float


@define
class TaskSamplerArgs:
    process_ind: int
    """The process index number."""

    mode: Literal["train", "eval"]
    """Whether we are in training or evaluation mode."""

    house_inds: List[int]
    """Which houses to use for each process."""

    houses: datasets.Dataset
    """The hugging face Dataset of all the houses in the split."""

    sensors: List[Sensor]
    """The sensors to use for each task."""

    controller_args: Dict[str, Any]
    """The arguments to pass to the AI2-THOR controller."""

    action_scale: List[float]
    """Action scale for different actions"""

    reward_config: Dict[str, Any]
    """The reward configuration to use."""

    target_object_types: List[str]
    """The object types to use as targets."""

    max_steps: int
    """The maximum number of steps to run each task."""

    max_tasks: int
    """The maximum number of tasks to run."""

    distance_type: str
    """The type of distance computation to use ("l2" or "geo")."""

    resample_same_scene_freq: int
    """
    Number of times to sample a scene/house before moving to the next one.
    
    If <1 then will never 
        sample a new scene (unless `force_advance_scene=True` is passed to `next_task(...)`.
    ."""

    visualize: bool

    his_len: int
    # Can we remove?
    deterministic_cudnn: bool = False
    loop_dataset: bool = True
    seed: Optional[int] = None
    allow_flipping: bool = False

    extra_task_spec: Dict[str, Any] = None

@define
class RewardConfig:
    step_penalty: float
    energy_penalty: float
    goal_success_reward: float
    knob_success_reward: float
    grasp_success_reward: float
    table_success_reward: float
    open_section_reward: float
    open_initial_reward: float
    complete_task_reward: float
    failed_stop_reward: float
    shaping_weight: float
    reached_horizon_reward: float
    positive_only_reward: bool
    failed_action_penalty: float
    new_room_reward: float
    new_object_reward: float
    end_effector_position_reward: float
    manipulation_shaping_scale: float
    manipulation_shaping_moving_scale: float
    manipulation_shaping_weight: float

    too_close_penalty: float

    navigation_energy_penalty_scale: float
    # For Cleaning Table
    cleaning_reward: float
    per_dirt_reward: float

class AgentPose(TypedDict):
    position: Vector3
    rotation: Vector3
    horizon: int
    standing: bool
