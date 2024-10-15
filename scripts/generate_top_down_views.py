import datetime
import json
import cv2
import random

from ai2thor.controller import Controller
import prior



def get_top_down_view(controller, offset):
    event = controller.step(action="GetMapViewCameraProperties")
    cam = event.metadata["actionReturn"].copy()
    # cam['position']['x'] += offset
    # cam['position']['y'] += offset
    # cam['position']['z'] += offset
    cam['rotation']['x'] = 60
    cam['rotation']['z'] = 0
    cam['rotation']['y'] = 90
    cam["orthographicSize"] = None
    cam["orthographic"] = False
    event = controller.step(
        action="AddThirdPartyCamera", skyboxColor="white", **cam
    )
    return event.third_party_camera_frames[0]


scenes = 'local_scenes/all_kianas.json'
with open(scenes) as f:
    all_houses=json.load(f)

controller_params = {'branch': 'main', 'width': 1080, 'height': 720, 'rotateStepDegrees': 30, 'visibilityDistance': 1, 'gridSize': 0.25, 'agentMode': 'locobot', 'fieldOfView': 90, 'snapToGrid': False, 'renderDepthImage': False, 'x_display': ':0.0', 'scene': 'Procedural'}
house = all_houses[0]

controller = Controller(**controller_params)
idx=0
for house in all_houses[idx:idx+15]:
    controller.reset(scene=house)
    top_frame = get_top_down_view(controller,idx-2)
    # cv2.imshow('window',top_frame)
    # cv2.waitKey(0)
    cv2.imwrite(f'output/top_view_houses/kianas/{idx}60.png',top_frame)
    idx += 1

