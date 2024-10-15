from training.tasks.object_nav import (
    ObjectNavTask, 
    ObjectNavTaskSampler, 
    Phone2ProcObjectNavTaskSampler, 
    Phone2ProcValidTaskSampler, 
    ObjectNavRealTask, 
    RealObjectNavTaskSampler
)
from utils.map_utils import build_centered_map, update_aggregate_map_blocky
from utils.utils import ForkedPdb
from typing import Any, Dict, List, Optional, Tuple, Union, Type
import numpy as np
import copy
import json

from shapely.geometry import Point, Polygon
from shapely.vectorized import contains

unrewarded_objects = ["Floor","Wall","Doorway","Doorframe","Window","ShelvingUnit","CounterTop","Shelf","Drawer"]

def rooms_visited(path, room_polymap:dict, previously_visited_rooms = {}):
    elimination_polymap = {k: v for k, v in room_polymap.items() if k not in previously_visited_rooms}
    visited_rooms = copy.deepcopy(previously_visited_rooms)
    for agent_pose in path:
        for room_id, poly in visited_rooms.items():
            if poly.contains(Point(agent_pose['x'],agent_pose['z'])):
                break

        for room_id, poly in elimination_polymap.items():
            if poly.contains(Point(agent_pose['x'],agent_pose['z'])):
                del elimination_polymap[room_id]
                visited_rooms[room_id] = poly
                break
    return visited_rooms, previously_visited_rooms

def get_rooms_polymap_and_type(house):
    room_poly_map = {}
    room_type_dict = {}
    for i, room in enumerate(house["rooms"]):
        room_poly_map[room["id"]] = Polygon(
            [(p["x"], p["z"]) for p in room["floorPolygon"]]
        )
        room_type_dict[room["id"]] = room['roomType']
    return room_poly_map, room_type_dict

class ObjectNavMappingTask(ObjectNavTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregate_map = None
        self.instant_map = None
        self.room_polys = None
        self.seen_object_ids = set()
        self.visited_rooms = None
        self.object_counter = None
        self.exp_reward = {
            'new_room':False,
            'new_objects':False,
            'increased_coverage':False
        }
    
    def set_map(self,base_map, room_polys):
        self.aggregate_map = copy.deepcopy(base_map["map"])
        self.map_scaling_params = {key: base_map[key] for key in ["xyminmax","pixel_sizes"]}
        m,n,_ = self.aggregate_map.shape
        # self.instant_map = np.concatenate((copy.deepcopy(base_map["map"]),np.zeros([m,n,11])),axis=2)
        self.room_polys = room_polys
        self.map_observations = [self.sensor_suite.sensors['aggregate_map_sensor'].get_observation(self.controller,self)]
    
    def set_counter(self,object_counter):
        self.object_counter = object_counter
    
    def _step(self, action: int):
        sr = super()._step(action)                
        current_pose = self.controller.last_event.metadata["agent"]
        heading_idx = int(np.round(current_pose["rotation"]["y"] / 30)) % 11
        possible_headings = [0.0]*12 
        possible_headings[heading_idx] = 1
        map_update = [current_pose['position']['x'],current_pose['position']['z'],1]
        self.aggregate_map = update_aggregate_map_blocky(self.aggregate_map,[map_update],**self.map_scaling_params)
        # self.instant_map = map_utils.populate_instantaneous_map(self.instant_map,[map_update+possible_headings])

        ## New objects
        visible_object_ids = set(
            o["objectId"]
            for o in self.controller.last_event.metadata["objects"]
            if o["visible"]
            and o["objectType"] not in unrewarded_objects
        )

        if len(visible_object_ids.difference(self.seen_object_ids)) > 0:
            # ForkedPdb().set_trace()
            self.exp_reward['new_objects'] = True
        
        self.seen_object_ids.update(visible_object_ids)
        
        ## New rooms
        if self.visited_rooms is None:
            self.visited_rooms,_ = rooms_visited(self.path,self.room_polys)
        else:
            self.visited_rooms,previously_visited_rooms = rooms_visited(self.path[-3:],self.room_polys,self.visited_rooms)
            if len(self.visited_rooms) > len(previously_visited_rooms):
                self.exp_reward['new_room'] = True

        ## Map video
        if self.visualize:
            self.map_observations.append(self.sensor_suite.sensors['aggregate_map_sensor'].get_observation(self.controller,self))

        return sr

    def judge(self) -> float:
        base_reward = super().judge()
        if self.exp_reward['new_room']:
            base_reward += self.reward_config.new_room_reward
        if self.exp_reward['new_objects']:
            base_reward += self.reward_config.new_object_reward
        if self.exp_reward['increased_coverage']:
            raise NotImplementedError
        
        self.exp_reward = self.exp_reward.fromkeys(self.exp_reward,False) # reset
        
        self._rewards[-1] = (float(base_reward))
        self.task_info["rewards"][-1] = (float(base_reward))
        return float(base_reward)

    
    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        number_rooms_visited = len(self.visited_rooms)
        metrics['percentage_rooms_visited'] = number_rooms_visited/len(self.room_polys)
        metrics['room_visitation_efficiency'] = number_rooms_visited/metrics["ep_length"]
        metrics['map_coverage'] = np.sum(self.aggregate_map[:,:,[2]].flatten())/(np.sum(self.aggregate_map[:,:,[0]].flatten()))
        metrics['map_exploration_efficiency'] = np.sum(self.aggregate_map[:,:,[2]].flatten())/metrics["ep_length"]
        metrics['seen_objects'] = len(self.seen_object_ids)
        metrics['new_object_rate'] = len(self.seen_object_ids)/metrics["ep_length"]
        all_object_types = ['Sofa','Television','Bed','Chair','Toilet','AlarmClock',
                            'Apple','BaseballBat','BasketBall','Bowl','GarbageCan',
                            'HousePlant','Laptop','Mug','SprayBottle','Vase']
        for obj in all_object_types:
            metrics['z_object_fraction_'+obj] = 0
            metrics['z_counter_'+obj] = self.object_counter[obj]
            if sum(self.object_counter.values())>0:
                metrics['z_counter_fraction_'+obj] = self.object_counter[obj]/sum(self.object_counter.values())
            else:
                metrics['z_counter_fraction_'+obj]=0
        metrics['z_object_fraction_'+self.task_info["object_type"]] = 1

        if not self._success:
            metrics['map_coverage_failure'] = metrics['map_coverage']
            metrics['map_exp_eff_failure'] = metrics['map_exploration_efficiency']
            metrics['fail_eplen'] = metrics["ep_length"]
            metrics['seen_objects_fail'] = len(self.seen_object_ids)
        
        self._metrics = metrics
        return metrics

class ObjectNavMappingSampler(Phone2ProcObjectNavTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_cache = dict()
        self.room_poly_cache = dict()
    
    def get_house_map(self):
        if (self.house_index in self.map_cache):
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]
        else:
            base_map, xyminmax, pixel_sizes = build_centered_map(self.house)
            self.map_cache[self.house_index] = {"map":base_map,"xyminmax":xyminmax, "pixel_sizes":pixel_sizes}
            self.room_poly_cache[self.house_index],_ = get_rooms_polymap_and_type(self.house)
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavMappingTask]:
        parent_task = super().next_task(force_advance_scene)
        parent_task.set_map(*self.get_house_map())
        parent_task.set_counter(self.obj_type_counter)
        self._last_sampled_task = parent_task
        return self._last_sampled_task

class ObjectNavValidationMappingSampler(Phone2ProcValidTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_cache = dict()
        self.room_poly_cache = dict()
    
    def get_house_map(self):
        if (self.house_index in self.map_cache):
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]
        else:
            base_map, xyminmax, pixel_sizes = build_centered_map(self.house)
            self.map_cache[self.house_index] = {"map":base_map,"xyminmax":xyminmax, "pixel_sizes":pixel_sizes}
            self.room_poly_cache[self.house_index],_ = get_rooms_polymap_and_type(self.house)
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavMappingTask]:
        parent_task = super().next_task(force_advance_scene)
        if parent_task is not None:
            parent_task.set_map(*self.get_house_map())
            parent_task.set_counter(self.obj_type_counter)
        self._last_sampled_task = parent_task
        return self._last_sampled_task

class ObjectNavRealMappingTask(ObjectNavRealTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregate_map = None
        self.instant_map = None
        self.room_polys = None
    
    def set_map(self,base_map, room_polys):
        self.aggregate_map = copy.deepcopy(base_map["map"])
        self.map_scaling_params = {key: base_map[key] for key in ["xyminmax","pixel_sizes"]}
        m,n,_ = self.aggregate_map.shape
        self.instant_map = np.concatenate((copy.deepcopy(base_map["map"]),np.zeros([m,n,11])),axis=2)
        self.room_polys = room_polys
    
    def _step(self, action: int):
        sr = super()._step(action)                
        current_pose = self.controller.last_event.metadata["agent"]
        heading_idx = int(np.round(current_pose["rotation"]["y"] / 30)) % 11
        possible_headings = [0.0]*12 
        possible_headings[heading_idx] = 1
        map_update = [current_pose['position']['x'],current_pose['position']['z'],1]
        self.aggregate_map = update_aggregate_map_blocky(self.aggregate_map,[map_update],**self.map_scaling_params)
        return sr
    
class ObjectNavRealMappingSampler(RealObjectNavTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_cache = dict()
        self.room_poly_cache = dict()
    
    def get_house_map(self):
        if (self.house_index in self.map_cache):
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]
        else:
            with open('local_scenes/all_back_apartment.json') as f:
                all_houses=json.load(f)
            base_map, xyminmax, pixel_sizes = build_centered_map(all_houses[0])
            self.map_cache[self.house_index] = {"map":base_map,"xyminmax":xyminmax, "pixel_sizes":pixel_sizes}
            self.room_poly_cache[self.house_index],_ = get_rooms_polymap_and_type(all_houses[0])
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]

    def next_task(self) -> Optional[ObjectNavRealMappingTask]:
        parent_task = super().next_task()
        if parent_task is not None:
            parent_task.set_map(*self.get_house_map())
        self._last_sampled_task = parent_task
        return self._last_sampled_task