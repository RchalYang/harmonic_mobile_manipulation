# from stretch_initialization_utils import STRETCH_ENV_ARGS
import imageio
import random
import ai2thor.controller
import prior
import numpy as np
import pdb
import ai2thor
from ai2thor.platform import CloudRendering
# from training.robot.stretch_initialization_utils import STRETCH_ENV_ARGS

import ai2thor.fifo_server
# from .type_utils import THORActions

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

gpu_index = 0
args = {"gpu_device": gpu_index, "platform": CloudRendering}

dataset = prior.load_dataset(
    "procthor-10k"
)["train"]

# c = ai2thor.controller.Controller(
#     scene=dataset[0],
#     **STRETCH_ENV_ARGS,
#     ** args
# )

# controller.step(
#     action="AddThirdPartyCamera",
#     position=dict(x=1.5, y=1.5, z=1.5),
#     rotation=dict(x=45, y=30, z=0),
#     fieldOfView=90
# )


dataset = prior.load_dataset("procthor-10k")['train']
scene_description = dataset[0]

for object_list in scene_description["objects"]:
  print(object_list)
  print(type(object_list))

print(type(scene_description["objects"]))
scene_description["objects"].append(
    {
        'assetId': 'GarbageBag_21_1', 'id': 'object_to_move',
        'kinematic': False,
        'position': {'x': 2.5, 'y': 0, 'z': 2.5}, 'rotation': {'x': 0, 'y': 0, 'z': 0}, 'material': None
    }
)

# exit()
controller = ai2thor.controller.Controller(
    **args,
    ** STRETCH_ENV_ARGS, scene=scene_description)

controller.step(
    action="AddThirdPartyCamera",
    position=dict(x=1.5, y=1.5, z=1.5),
    rotation=dict(x=45, y=30, z=0),
    fieldOfView=90
)


random_position = {'x': 3.8, 'y': 0.9, 'z': 2.0}

teleport_dict = dict(action='TeleportFull', standing=True,
                     position=random_position, rotation=dict(x=0, y=270, z=0), horizon=0)
controller.step(**teleport_dict)


def get_both_frames(controller):
  first_frame = controller.last_event.frame[..., :3]
  second_frame = controller.last_event.third_party_camera_frames[0][..., :3]
  return np.concatenate([first_frame, second_frame], axis=1)


def get_third_party_frame(controller):
  return controller.last_event.third_party_camera_frames[1]


all_frames = [get_both_frames(controller)]
third_party_farmes = [get_third_party_frame(controller)]

steps = 720
controller.step(
    dict(action='MoveArm', position=dict(x=0, y=0.1, z=0.5)),
)

r = 1.1
for i in range(120):
  controller.step(
      dict(action='MoveAgent', ahead=0.025),
  )
  controller.step(
      dict(action='MoveArm', position=dict(
          x=0, y=0.1, z=0.1 + 0.9 * (i / 120))),
  )
  controller.step(
      dict(action='RotateAgent', degrees=0.3),
  )
  print(i, controller.last_event.metadata["agent"])

  all_frames.append(get_both_frames(controller))
  third_party_farmes.append(get_third_party_frame(controller))


def save_video(file_path, frames):
  kwargs = {
      "fps": 20,
      "quality": 5,
  }
  imageio.mimwrite(file_path, np.array(frames), macro_block_size=1, **kwargs)


save_video('push_object.mp4', all_frames)
save_video('thirdparty_push_object.mp4', third_party_farmes)

# pdb.set_trace()
