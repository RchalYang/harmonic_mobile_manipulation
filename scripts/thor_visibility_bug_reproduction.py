import json
from ai2thor.controller import Controller

scenes = 'local_scenes/all_back_apartment.json'
with open(scenes) as f:
    all_houses=json.load(f)

controller_params = {'branch': 'main', 'width': 400, 'height': 300, 'rotateStepDegrees': 30, 'visibilityDistance': 1, 'gridSize': 0.25, 
                     'agentMode': 'locobot', 'fieldOfView': 63.453048374758716, 'snapToGrid': False, 'renderDepthImage': False, 
                     'x_display': ':0.0', 'scene': 'Procedural'}
h = all_houses[0]
controller = Controller(**controller_params)
controller.reset(scene=h)
controller.step(action="Teleport",position={'x': -11.0, 'y': 0.9009997844696045, 'z': 1.25},rotation={'x': 0, 'y': 270, 'z': 0})
controller.step("RotateLeft")
obj=[obj for obj in controller.last_event.metadata["objects"] if obj["objectType"]=="Bowl"]
obj[0]["visible"]


# # houseplant
# self.controller.step(action="Teleport",position={'x': 1.75, 'y': 0.9009997844696045, 'z': 1.0},rotation={'x': 0, 'y': 90, 'z': 0})
# controller.step("RotateLeft")

# visibility_points = self.controller.step(action="GetVisibilityPoints", objectId=object_id, raise_for_failure=True).metadata["actionReturn"]
# self.get_nearest_positions(world_position=visibility_points[0])