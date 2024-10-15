import os
from math import floor
import random
import pdb
import sys
from turtle import pos
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import copy
import numpy as np
import torch
import torch.nn as nn

import ai2thor
from ai2thor.controller import Controller
from allenact.base_abstractions.misc import ActorCriticOutput

from procthor.databases import material_database, solid_wall_colors
from training import cfg

from allenact.utils.system import get_logger

FLOOR_MATERIALS = material_database["Wood"]
WALL_MATERIALS = material_database["Wall"]

P_SAMPLE_SOLID_WALL_COLOR = 0.5
"""Probability of sampling a solid wall color instead of a material."""


ROBOTHOR_USABLE_POSITIONS = [{'horizon': 30, 'position': {'x': 0.0021370719589164366, 'y': 0.95, 'z': 0.48081013947823736},'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -0.7495494126915947, 'y': 0.95, 'z': 0.713313342267424}, 'rotation': {'x': 0, 'y': 90, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': 0.5004505873084071, 'y': 0.95, 'z': 0.21445044272873526}, 'rotation': {'x': 0, 'y': 180, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': 0.2520650495061627, 'y': 0.95, 'z': 0.22801799940049428}, 'rotation': {'x': 0, 'y': 0, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -1.4989227145848343, 'y': 0.95, 'z': -2.7865978308536685}, 'rotation': {'x': 0, 'y': 180, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -0.7487419646875413, 'y': 0.95, 'z': -2.7854200269302396}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -0.9995494126915947, 'y': 0.95, 'z': 0.4636670807612613}, 'rotation': {'x': 0, 'y': 180, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -6.998741944818365, 'y': 0.95, 'z': -0.020057500470285206}, 'rotation': {'x': 0, 'y': 90, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': 0.7521370675435435, 'y': 0.95, 'z': -0.03544331171977877}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -0.9995494325607694, 'y': 0.95, 'z': 0.2142599938512686}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': 0.00045058730840707995, 'y': 0.95, 'z': -0.03647036301709594}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -1.247765064620637, 'y': 0.95, 'z': 0.7127750944028093}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': 1.0027686909983533, 'y': 0.95, 'z': 0.21398085728231164}, 'rotation': {'x': 0, 'y': 0, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -0.49766940447108077, 'y': 0.95, 'z': 0.21342217923554863}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -0.24954943256077122, 'y': 0.95, 'z': 0.7158457409101113}, 'rotation': {'x': 0, 'y': 90, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -0.9995494325607694, 'y': 0.95, 'z': 0.2142599938512686}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': 0.2504505873084071, 'y': 0.95, 'z': 0.4627743351987159}, 'rotation': {'x': 0, 'y': 180, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -0.9995494126915929, 'y': 0.95, 'z': 0.7143848803939243}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': -0.247741402639285, 'y': 0.95, 'z': 0.46436312919665346}, 'rotation': {'x': 0, 'y': 0, 'z': 0}}, 
                                {'horizon': 30, 'position': {'x': 0.7504510553378942, 'y': 0.95, 'z': -0.036612531575594076}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}]

BACK_APARTMENT_USABLE_POSITIONS = [
                                {'horizon': 30, 'position': {'x': -9.65, 'y': 0.95, 'z': -2},'rotation': {'x': 0, 'y': 0, 'z': 0}}, #entryway
                                {'horizon': 30, 'position': {'x': 0, 'y': 0.95, 'z': 0}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, #kitchen
                                {'horizon': 30, 'position': {'x': -4, 'y': 0.95, 'z': -2}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, #office hallway
                                # {'horizon': 30, 'position': {'x': -1, 'y': 0.95, 'z': -1.5}, 'rotation': {'x': 0, 'y': 180, 'z': 0}}, # kitchen doorway
                                # {'horizon': 30, 'position': {'x': -12.19, 'y': 0.95, 'z': -1.29}, 'rotation': {'x': 0, 'y': 180, 'z': 0}}, 
                                # {'horizon': 30, 'position': {'x': -3, 'y': 0.95, 'z': -1.75}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}, 
                                # {'horizon': 30, 'position': {'x': -2, 'y': 0.95, 'z': -1.75}, 'rotation': {'x': 0, 'y': 270, 'z': 0}}
                                ]

KIANAS_USABLE_POSITIONS = [{'horizon': 30, 'position': {'x': 4.7, 'y': 0.95, 'z': 5.0},'rotation': {'x': 0, 'y': 0, 'z': 0}}, #bedroom
                                {'horizon': 30, 'position': {'x': 5.6, 'y': 0.95, 'z': 0.5}, 'rotation': {'x': 0, 'y': 180, 'z': 0}}, #kitchen
                                {'horizon': 30, 'position': {'x': 1, 'y': 0.95, 'z': -1}, 'rotation': {'x': 0, 'y': 90, 'z': 0}}, #living room
]

class ForkedPdb(pdb.Pdb): # used for real episode logging
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin 


def sample_wall_params():
    if random.random() < P_SAMPLE_SOLID_WALL_COLOR:
        return {"name": "PureWhite", "color": random.choice(solid_wall_colors)}
    return {"name": random.choice(WALL_MATERIALS)}


def randomize_wall_materials(partial_house) -> None:
    """Randomize the materials on each wall."""
    room_ids = set()
    for wall in partial_house["walls"]:
        room_ids.add(wall["roomId"])
    room_ids.add("ceiling")

    wall_params_per_room = dict()
    for room_id in room_ids:
        wall_params_per_room[room_id] = sample_wall_params()

    for wall in partial_house["walls"]:
        wall["material"] = wall_params_per_room[wall["roomId"]]

    # NOTE: randomize ceiling material
    partial_house["proceduralParameters"]["ceilingMaterial"] = wall_params_per_room[
        "ceiling"
    ]

def randomize_floor_materials(partial_house) -> None:
    """Randomize the materials on each floor."""
    for room in partial_house["rooms"]:
        room["floorMaterial"] = {"name": random.choice(FLOOR_MATERIALS)}


def randomize_wall_and_floor_materials(partial_house) -> None:
    """Randomize the materials on each wall and floor."""
    randomize_wall_materials(partial_house)
    randomize_floor_materials(partial_house)

def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)

    return points

from training.utils.map_utils import (
    get_room_id_from_location,
    get_rooms_polymap_and_type
)

def randomize_lighting(partial_house):
    # get base light params
    original_light_params = copy.deepcopy(partial_house['proceduralParameters']['lights'])
    updated_light_params=[]
    
    # define floor polygon - merge the rooms
    all_room_points = [x['floorPolygon'] for x in partial_house['rooms']]
    all_room_polys = [Polygon([(p['x'], p['z']) for p in rp]) for rp in all_room_points]
    floor_poly = unary_union(all_room_polys)

    # for each room
    room_poly_map, room_type_dict = get_rooms_polymap_and_type(partial_house)
    light_in_house, light_outside_house = [], []
    for light in partial_house['proceduralParameters']['lights']:
        light["roomId"] = get_room_id_from_location(room_poly_map, light["position"])
        if light["roomId"] is None:
            light_outside_house.append(light)
        else:
            light_in_house.append(light)
    # room_ids = set(get_room_id_from_location(room_poly_map, x["position"]) for x in partial_house['proceduralParameters']['lights'])
    room_ids = set(x['roomId'] for x in light_in_house)
    # get_logger().warning(partial_house['proceduralParameters']['lights'])
    for r in sorted(room_ids):
        # pick a light in that room, copy params and randomize non-location ones
        light_to_keep = [x for x in partial_house['proceduralParameters']['lights'] if x['roomId'] == r][0]
        updated_light_params.append(randomize_lighting_params(light_to_keep))

        # sample from # of lights between 0 and 5 (per room, but not bounded to the room)
        num_lights = random.randint(0,5)

        # sample in shapely polygons of ['rooms']['floorPolygon'] x z points for light positions
        light_positions = random_points_within(floor_poly,num_lights)

        for p in light_positions:
            base_light = copy.deepcopy(original_light_params[0])
            base_light['id'] = 'light_0_' + str(len(updated_light_params))
            updated_light_params.append(randomize_lighting_params(base_light,p))

    partial_house['proceduralParameters']['lights'] = updated_light_params + light_outside_house

def randomize_lighting_params(base_light,light_pos=None):
    light=copy.deepcopy(base_light)
    light['intensity'] = np.clip(random.gauss(.5,.25),0,1)
    if light_pos is not None:
        light['position']['x'] = light_pos.x
        light['position']['z'] = light_pos.y 

    # bounded randomization - unbounded uniform was too rave
    light['rgb']['r'] = random.uniform(0.5,1.0)
    light['rgb']['g'] = random.uniform(0.5,1.0)
    light['rgb']['b'] = random.uniform(0.5,1.0)

    # shadow randomization
    light['shadow']['bias'] = random.uniform(0.05,0.2)
    light['shadow']['strength'] = np.clip(random.gauss(.75,.25),0,1)

    return light

def randomize_chairs(partial_house):
    chairs = [o for o in partial_house['objects'] if "Chair" in o['id']]
    obs = [o for o in partial_house['objects'] if "obstacle" in o['id']]
    # switch one with an obstacle
    idx_chair, idx_obs = random.randint(1,len(chairs)) -1 ,random.randint(1,len(obs)) - 1
    obs_pos = copy.deepcopy(obs[idx_obs]['position'])
    chair_pos = copy.deepcopy(chairs[idx_chair]['position'])
    chairs[idx_chair]['position']['x'], chairs[idx_chair]['position']['z'] = obs_pos['x'], obs_pos['z']
    obs[idx_obs]['position']['x'], obs[idx_obs]['position']['z'] = chair_pos['x'], chair_pos['z']

    # randomize the rotations of others
    for idx in random.sample(range(len(chairs)),np.floor(len(chairs)/2).astype(int)):
        chairs[idx]['rotation']['y'] = random.choice([i for i in range(0,360,1)])

def remove_all_chairs_and_modify_openness(partial_house):
    partial_house['objects'] = [o for o in partial_house['objects'] if "Chair" not in o['id']]
    for door in partial_house['doors']:
        if door.get('openness', 0) > 0:
            door['openness']=0.8
    

class StochasticController(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nominal_neck = None # this always initializes around 0 (CHECK/add an assert in the task) set during teleport


    def step(self, action=None, **action_args) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""


        if action in ['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft']:
            super_controller = super(StochasticController, self)

            #add the continuous movements
            # action_args['returnToStart'] = False
            res = super_controller.step(action, **action_args)

            #if action is failed add small random movements
            if res.metadata['lastActionSuccess'] is False:
                random_move = random.uniform(-0.03, 0.03)
                random_rotation = random.uniform(-2,2)
                super_controller.step(dict(action='MoveAhead', moveMagnitude=random_move))#, returnToStart=False))
                rotate_action = random.choice(['RotateRight', 'RotateLeft'])
                super_controller.step(dict(action=rotate_action, degrees=random_rotation))#, returnToStart=False))
                #fix the action failure
                self.last_event.metadata['lastActionSuccess'] = False
                res.metadata['lastActionSuccess'] = False
        elif action in ['LookUp', 'LookDown']:
            neck_change = -1 if action == 'LookDown' else 1
            if (self.last_event.metadata['agent']['cameraHorizon'] < 6) & (neck_change > 0):
                if self.last_event.metadata['agent']['cameraHorizon'] == -30:
                    res = super(StochasticController, self).step(action, **action_args) # this will always fail (this is fine)
                else:
                    res = super(StochasticController, self).step(dict(action='Teleport',horizon=-30)) # just peg it out. not ideal
                    res.metadata['lastAction'] = action
                    self.nominal_neck += neck_change
            elif abs(self.nominal_neck + neck_change) > 1:
                res = super(StochasticController, self).step(action='Pass', **action_args)
                self.last_event.metadata['lastActionSuccess'] = False
                res.metadata['lastActionSuccess'] = False
            else:
                res = super(StochasticController, self).step(action, **action_args)
                self.nominal_neck += neck_change
            
            if self.nominal_neck == 0: # just checking
                try:
                    assert abs(self.last_event.metadata["agent"]["cameraHorizon"]) < 6
                except:
                    print('neck problem')
                    if self.last_event.metadata["agent"]["cameraHorizon"] > 0: # positive means below the horizon
                        self.nominal_neck = -1
                    else:
                        self.nominal_neck = 1
        else:
            res = super(StochasticController, self).step(action, **action_args)
        
        #return the action metadata
        return res 

def findkeys(node, kv):
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
               yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x

def compute_movement_efficiency(task_info):
    # if "followed_path" in task_info: # this is not exactly 1:1 since it cuts off the first point
    #     return _movement_efficiency_metric(np.array([[p['x'],p['z']] for p in task_info['followed_path']]))

    taken_actions = task_info["taken_actions"]
    action_successes = task_info["action_successes"]
    path = [[0,0,0]] # x, z, heading
    for a,s in zip(taken_actions,action_successes):
        pose = copy.deepcopy(path[-1])
        if s:
            if a == 'RotateRight':
                pose[2] = (pose[2] + 30) % 360
            elif a == 'RotateLeft':
                pose[2] = (pose[2] - 30) % 360
            elif a == 'MoveAhead':
                pose[0] = pose[0] + 0.25*np.cos(pose[2]*np.pi/180) # positive x in the 0 heading direction
                pose[1] = pose[1] + 0.25*np.sin(pose[2]*np.pi/180)
        path.append(pose)
    positions = np.array(path)
    return _movement_efficiency_metric(positions[:,0:2])

def _movement_efficiency_metric(
        positions: np.ndarray, clamp_max: float = 2.0
    ):
        psq = (positions * positions).sum(-1, keepdims=True)

        squared_dists = psq + psq.T - 2 * np.matmul(positions, positions.T)
        dists = np.sqrt(np.clip(squared_dists, a_min=0, a_max=None))

        n = positions.shape[0]
        uniform_steps = np.array([0.25 * i for i in range(n)]).reshape((-1, 1))
        max_possible_dist_traveled = np.clip(
            np.abs(uniform_steps - uniform_steps.T), a_min=1e-4, a_max=clamp_max
        )

        dist_vs_max = np.minimum(dists, clamp_max) / max_possible_dist_traveled

        return float(dist_vs_max.sum() / (n * (n - 1))) if n != 0 else 1.0

def log_ac_return(ac: ActorCriticOutput, task_id_obs):
    os.makedirs("output/ac-data/", exist_ok=True)
    assert len(task_id_obs.shape) == 3

    for i in range(len(task_id_obs[0])):
        task_id = "".join(
            [
                chr(int(k))
                for k in task_id_obs[0, i]
                if chr(int(k)) != " "
            ]
        )

        with open(f"output/ac-data/{task_id}.txt", "a") as f:
            estimated_value = ac.values[0, i].item()
            if hasattr(ac.distributions, "logits"):
                policy = nn.functional.softmax(
                    ac.distributions.logits[0, i]
                ).tolist()
            else:
                # get_logger().w
                policy = (
                    ac.distributions.mean[0, i]
                ).tolist()
            f.write(",".join(map(str, policy + [estimated_value])) + "\n")

def log_map_output(ac: ActorCriticOutput, task_id_obs):
    os.makedirs("output/mapping-data/", exist_ok=True)
    assert len(task_id_obs.shape) == 3

    for i in range(len(task_id_obs[0])):
        task_id = "".join(
            [
                chr(int(k))
                for k in task_id_obs[0, i]
                if chr(int(k)) != " "
            ]
        )
        with open(f"output/mapping-data/{task_id}.txt", "a") as f:
            map_est = ac.extras['map_estimate'].flatten().tolist()
            f.write(",".join(map("{:.3f}".format,map_est))+ "\n")


def reconstruct_resized_map_estimate(map_est,resized_walls,gt_all,pos,sig):
    agg_image = np.zeros((40,40,3))
    agg_image[:,:,0] = resized_walls
    agg_image[:,:,2] = gt_all

    agg_image[:,:,1] = np.clip((sig( torch.from_numpy(map_est).type(torch.FloatTensor))*15).numpy(),0,1)
    agg_image[pos[0],pos[1],:] = 1 # highlight the current step
    return agg_image

def add_time_to_dict(timer_dict, key, time_distance):
    timer_dict.setdefault(key, [])
    timer_dict[key].append(time_distance.total_seconds())

def average_timer(timer_dict):
    print ('--------Timer-------')
    for k,v in timer_dict.items():
        print(k, ':', sum(v) / len(v), 'len is ', len(v))
