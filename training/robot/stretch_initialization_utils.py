import ai2thor.fifo_server
from .type_utils import THORActions

# AGENT_ROTATION_DEG = 15
# AGENT_MOVEMENT_CONSTANT = 0.05
# HORIZON = 0  # RH: Do not change from 0! this is now set elsewhere with RotateCameraMount actions
# ARM_MOVE_CONSTANT = 0.025
AGENT_ROTATION_DEG = 30
AGENT_MOVEMENT_CONSTANT = 0.2
HORIZON = 0  # RH: Do not change from 0! this is now set elsewhere with RotateCameraMount actions
ARM_MOVE_CONSTANT = 0.1
WRIST_ROTATION = 10

ORIGINAL_INTEL_W, ORIGINAL_INTEL_H = 1280, 720
INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT = 396, 224
INTEL_VERTICAL_FOV = 59
do_not_use_branch = "cam_adjust"
AGENT_RADIUS_LIST = [(0, 0.5), (1, 0.4), (2, 0.3), (3, 0.2)]

STRETCH_BUILD_ID = "679fe9c8581e79670ba28d2f1b26710d01514b9a"  # better obj vis fns and collision

MAXIMUM_SERVER_TIMEOUT = 1000  # default : 100 Need to increase this for cloudrendering

STRETCH_ENV_ARGS = dict(
    gridSize=AGENT_MOVEMENT_CONSTANT,
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
    visibilityDistance=1.0,
    visibilityScheme="Distance",
    fieldOfView=INTEL_VERTICAL_FOV,
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold=True,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
    renderInstanceSegmentation=False,
    agentMode="stretch",
    renderDepthImage=False,
    cameraNearPlane=0.01,  # VERY VERY IMPORTANT
    branch=None,  # IMPORTANT do not use branch
    commit_id=STRETCH_BUILD_ID,
    server_timeout=MAXIMUM_SERVER_TIMEOUT,
    snapToGrid=False,
    fastActionEmit=True,
    action_scale=[0.2, 0.2, 0.2, 0.2, 0.2]
)

assert (
    STRETCH_ENV_ARGS.get("branch") is None and STRETCH_ENV_ARGS["commit_id"] is not None
), "Should always specify the commit id and not the branch."


ADDITIONAL_MOVING_ARGS = {
    # "disableRendering": True,
    "returnToStart": True,
    "speed": 1,
}

ADDITIONAL_ARM_ARGS = {
    # "disableRendering": True,
    # "returnToStart": True,
    "speed": 1,
}

# STRETCH_WRIST_MIN = 155
# STRETCH_WRIST_MAX = 180

STRETCH_WRIST_INVALID_MIN = 75
STRETCH_WRIST_INVALID_MAX = 105

ALL_STRETCH_ACTIONS = [
    THORActions.move_ahead,
    THORActions.rotate_right,
    THORActions.rotate_left,
    THORActions.move_back,
    THORActions.done,
    THORActions.sub_done,
    THORActions.rotate_left_small,
    THORActions.rotate_right_small,
    THORActions.pickup,
    THORActions.move_arm_in,
    THORActions.move_arm_out,
    THORActions.move_arm_up,
    THORActions.move_arm_down,
    THORActions.wrist_open,
    THORActions.wrist_close,
    THORActions.move_arm_down_small,
    THORActions.move_arm_in_small,
    THORActions.move_arm_out_small,
    THORActions.move_arm_up_small,
    THORActions.dropoff,
]
