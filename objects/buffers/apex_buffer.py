from libs.common import Common
from libs.torch_module import ExtendedDataset

import h5py
import numpy as np
import random
import math
import torch
from torch.utils.data.dataloader import default_collate
import json

class ApexBuffer (Common):
    def __init__(self, master_agent):
        super().__init__()

        # set config, executor, shared_resources and master_agent
        self.config = master_agent.config
        self.executor = master_agent.executor
        self.shared_resources = master_agent.shared_resources
        self.master_agent = master_agent

        # set properties
        self._initProps()

        self.sum_tree = SumTree(self)
        self.dataset = Dataset(self)
        return
    
    @property
    def change_flg(self):
        return self.new_data_count >= self.threshold
        
    @property
    def simulation_id(self):
        return self.master_agent.network.simulation.get('id')
    
    @property
    def finish_flg(self):
        return self.master_agent.get('finish_flg')

    @property
    def episode(self):
        return self.master_agent.get('episode')
    
    @property
    def total_priority(self):
        return self.sum_tree.get('total_priority')
    
    def _initProps(self):
        drl_info = self.config.get('drl_info')

        # set apex buffer information
        self.size = drl_info['framework']['apex']['buffer']['size']
        self.file_capacity = drl_info['framework']['apex']['buffer']['file_capacity']
        self.reset_flg = drl_info['framework']['apex']['buffer']['priority']['reset']['type'] is not None and drl_info['framework']['apex']['buffer']['priority']['reset']['type'] == 'episode'
        if self.reset_flg:
            self.reset_interval = drl_info['framework']['apex']['buffer']['priority']['reset']['episode']['interval']
        
        # set training information
        self.threshold = drl_info['training']['threshold']
        self.batch_size = drl_info['training']['batch']['size']
        self.num_batches = drl_info['training']['batch']['number']

        # set buffer_file_path_map
        save_dir_path_map = self.master_agent.get('save_dir_path_map')
        buffer_dir_path = save_dir_path_map['buffer']
        session_dir_path = save_dir_path_map['session']

        self.buffer_file_path_map = {
            'session': session_dir_path / 'session.json',
            'tree': buffer_dir_path / 'tree.h5',
            'data': {
                data_id: buffer_dir_path / f"data_{data_id}.h5" 
                for data_id in range(1, math.ceil(self.size / self.file_capacity) + 1)
            }
        }

        # set new_data_count and next_data_id
        if self.simulation_id > 1:
            saved_buffer = self.shared_resources.get('buffer')
            self.new_data_count = saved_buffer.get('new_data_count')
            self.next_data_id = saved_buffer.get('next_data_id')
            self.current_size = saved_buffer.get('current_size')

        elif self.buffer_file_path_map['session'].exists():
            with open(self.buffer_file_path_map['session'], 'r', encoding='utf-8') as f:
                session_json = json.load(f)
            self.new_data_count = session_json['buffer']['new_data_count']
            self.next_data_id = session_json['buffer']['next_data_id']
            self.current_size = session_json['buffer']['current_size']

        else:
            self.new_data_count = 0
            self.next_data_id = 0   
            self.current_size = 0
        
        return
    
    def _showInfo(self):
        print('==============================================')
        print(f"status: buffer update")
        print(f"number of new data: {self.new_data_count}/{self.threshold}")
        print(f"buffer size: {self.current_size}/{self.size}")
        return
    
    def sample(self, number=None, use_collate=False):
        if number is None:
            number = self.batch_size * self.num_batches

        data_id_list = self.sum_tree.sample(number)
        learning_data_list = [self.dataset[data_id] for data_id in data_id_list]

        if not use_collate:
            return data_id_list, learning_data_list
        
        learning_data = default_collate(learning_data_list)
        return data_id_list, learning_data
    
    def update(self, data_id_list=None, priority_list=None, learning_data_list=None):
        # get update_type
        if data_id_list is None and priority_list is None and learning_data_list is not None:
            update_type = 'add'
        elif data_id_list is not None and priority_list is not None and learning_data_list is None:
            update_type = 'priority'
        else:
            raise NotImplementedError(f"Not supported arguments pattern.")

        if update_type == 'add':
            # if there is no new data, return
            if len(learning_data_list) == 0:
                return

            # get data_id_list
            data_id_list = list(range(self.next_data_id, self.next_data_id + len(learning_data_list)))
            data_id_list = [data_id % self.size for data_id in data_id_list]

            # update dataset and sum_tree
            self.dataset.update(data_id_list, learning_data_list)
            self.sum_tree.update(data_id_list)

            # update next_data_id, new_data_count and current_size
            self.next_data_id = (self.next_data_id + len(learning_data_list)) % self.size
            self.new_data_count += len(learning_data_list)
            self.current_size = min(self.current_size + len(learning_data_list), self.size)
            
            # clear learning_data_list
            self.master_agent.clearLearningData()

            # show buffer update info
            self._showInfo()

        elif update_type == 'priority':
            self.sum_tree.update(data_id_list, priority_list)

        else:
            raise NotImplementedError(f"Not supported arguments pattern.")
        
        return
    
    def reset(self, property_name):
        if property_name == 'new_data_count':
            self.new_data_count %= self.threshold
        elif property_name == 'priority':
            self.sum_tree.reset()
        else:
            raise NotImplementedError(f"Not supported property_name: {property_name}")
    
    def save(self):
        self.dataset.flush()
        self.dataset.close()
        self.sum_tree.save()
        return
        
class SumTree (Common):
    def __init__(self, buffer):
        super().__init__()

        self.buffer = buffer
        self.config = buffer.config
        self.shared_resources = buffer.shared_resources 

        self._initProps()
        
        if self.reset_flg and self.episode % self.reset_interval == 0:
            self._resetPriority()
            self._showInfo('reset')

        return
    
    @property
    def dataset(self):
        return self.buffer.dataset
    
    @property
    def simulation_id(self):
        return self.buffer.get('simulation_id')
    
    @property
    def current_size(self):
        return self.buffer.get('current_size')
    
    @property
    def next_data_id(self):
        return self.buffer.get('next_data_id')
    
    @property
    def total_priority(self):
        return self.tree_array[0]

    @property
    def priority_list(self):
        return self.tree_array[self.num_leaves - 1 : self.num_leaves - 1 + self.current_size]
    
    @property
    def episode(self):
        return self.buffer.get('episode')
    
    def _initProps(self):
        # set initial priority and size
        drl_info = self.config.get('drl_info')
        self.initial_priority = drl_info['framework']['apex']['buffer']['priority']['initial_value']
        self.reset_flg = self.buffer.get('reset_flg')
        if self.reset_flg:
            self.reset_interval = self.buffer.get('reset_interval')
        self.size = self.buffer.get('size')

        # set tree_file_path
        buffer_file_path_map = self.buffer.get('buffer_file_path_map')
        self.tree_file_path = buffer_file_path_map['tree']

        # set num_leaves (the number of leaf nodes in tree_array)
        self.num_leaves = 2**math.ceil(math.log2(self.size))

        # initialize tree_array and next_data_id (0 <= next_data_id < size)
        if self.simulation_id > 1:
            saved_sum_tree = self.shared_resources.buffer.sum_tree
            self.tree_array = saved_sum_tree.get('tree_array')

        elif self.tree_file_path.exists():
            tree_obj = h5py.File(self.tree_file_path, 'r')
            self.tree_array = tree_obj['tree_array'][:]
            tree_obj.close()

        else:
            self.tree_array = np.zeros(2 * self.num_leaves - 1, dtype=np.float32)

        return
    
    def _resetPriority(self):
        # reset initial priority
        drl_info = self.config.get('drl_info')
        self.initial_priority = drl_info['framework']['apex']['buffer']['priority']['initial_value']

        # reset tree_array
        self.tree_array = np.zeros(2 * self.num_leaves - 1, dtype=np.float32)
        for data_id in range(self.current_size):
            tree_id = data_id + self.num_leaves - 1
            self.tree_array[tree_id] = self.initial_priority

            change = self.initial_priority
            self._propagate(tree_id, change)

        return
    
    def _showInfo(self, type='reset'):
        print('==============================================')
        if type == 'reset':
            print(f"status: buffer priority reset")
            print(f"buffer size: {self.current_size}/{self.size}")
            print(f"priority value: {self.initial_priority}")
        else:
            raise NotImplementedError(f"Not supported type: {type}")
        
        return

    def _propagate(self, tree_id, change):
        parent_id = (tree_id - 1) // 2
        self.tree_array[parent_id] += change

        if parent_id != 0:
            self._propagate(
                tree_id=parent_id, 
                change=change
            )
        return
    
    def _retrieve(self, tree_id, random_value):
        left_child_id = 2 * tree_id + 1
        right_child_id = left_child_id + 1

        # stop if arriving at a leaf node
        if left_child_id >= len(self.tree_array):
            return tree_id
        
        if random_value <= self.tree_array[left_child_id]:
            # if random_value is smaller than the left child node's value, go to the left child node
            return self._retrieve(left_child_id, random_value)
        else:
            # otherwise, go to the right child node and adjust random_value by subtracting the left child node's value
            return self._retrieve(right_child_id, random_value - self.tree_array[left_child_id])
        
    def reset(self):
        self._resetPriority()
        self._showInfo('reset')
        return
        
    def sample(self, number):
        if self.current_size < number:
            return list(range(self.current_size))
        
        sample_value_list = np.random.uniform(0, self.total_priority, number).tolist()
        tree_id_list = [self._retrieve(0, sample_value) for sample_value in sample_value_list]
    
        # get data_id_list
        data_id_list = []
        for tree_id in tree_id_list:
            data_id = tree_id - self.num_leaves + 1
            if data_id + 1 > self.current_size:
                data_id = random.randint(0, self.current_size - 1)
            data_id_list.append(data_id)
    
        return sorted(data_id_list)
    
    def update(self, data_id_list, priority_list=None):
        if priority_list is None:
            priority_list = [self.initial_priority] * len(data_id_list)
        
        for data_id, priority in zip(data_id_list, priority_list):
            # get tree_id
            tree_id = data_id + self.num_leaves - 1

            # get change and update leaf node value
            change = priority - self.tree_array[tree_id]
            self.tree_array[tree_id] = priority

            # update parent node values
            self._propagate(tree_id, change)

        return
    
    def save(self):
        # save tree_array
        with h5py.File(self.tree_file_path, 'w') as tree_obj:
            if 'tree_array' in tree_obj:
                tree_obj['tree_array'][:] = self.tree_array
            else:
                tree_obj.create_dataset('tree_array', data=self.tree_array, dtype=np.float32)

        return
    
class Dataset(ExtendedDataset):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self.config = buffer.config

        self._initProps()
        self._makeDataFiles()
        return
    
    @property
    def current_size(self):
        return self.buffer.get('current_size')
    
    @property
    def next_data_id(self):
        return self.buffer.get('next_data_id')
    
    def _initProps(self):
        self.size = self.buffer.get('size')
        self.file_capacity = self.buffer.get('file_capacity')

        self.data_file_path_map = self.buffer.get('buffer_file_path_map')['data']

        self.num_roads = self.buffer.master_agent.get('num_roads')
        self.num_lanes_map = self.buffer.master_agent.get('num_lanes_map')  

        drl_info = self.config.get('drl_info')
        self.num_vehicles = drl_info['state']['vehicle']['number']

        self.num_features_map = self.config.get('num_features_map') 

        self.hdf5_obj_map = {}  
        return
    
    def _makeDataFiles(self):
        data_file_ids = sorted(list(self.data_file_path_map.keys()))
        for data_file_id in data_file_ids:
            data_file_path = self.data_file_path_map[data_file_id]
            if data_file_path.exists():
                continue
            
            with h5py.File(data_file_path, 'a') as data_obj:
                if data_file_id != data_file_ids[-1]:
                    capacity = self.file_capacity
                else:
                    capacity = self.size - self.file_capacity * (len(data_file_ids) - 1)
                
                # state and next_state
                for state_type in ['state', 'next_state']:
                    state_group =data_obj.create_group(state_type)
                    state_group.create_dataset(
                        'intersection',
                        shape=(capacity, self.num_features_map['intersection'][self.num_roads]),
                        dtype=np.float32                    
                    )
                    roads_group = state_group.create_group('roads')
                    for road_id in range(1, self.num_roads + 1):
                        road_group = roads_group.create_group(f"road_{road_id}")
                        road_group.create_dataset(
                            'road',
                            shape=(capacity, self.num_features_map['road'][self.num_roads]),
                            dtype=np.float32
                        )

                        lanes_group = road_group.create_group('lanes')
                        for lane_id in range(1, self.num_lanes_map[road_id] + 1):
                            lane_group = lanes_group.create_group(f"lane_{lane_id}")
                            lane_group.create_dataset(
                                'lane',
                                shape=(capacity, self.num_features_map['lane']),
                                dtype=np.float32
                            )
                            lane_group.create_dataset(
                                'vehicles',
                                shape=(capacity, self.num_vehicles, self.num_features_map['vehicle'][self.num_roads]),
                                dtype=np.float32
                            )

                # action, reward and done_flg
                data_obj.create_dataset(
                    'action',
                    shape=(capacity, 1),
                    dtype=np.float32
                )

                data_obj.create_dataset(
                    'cumulative_reward',
                    shape=(capacity, 1),
                    dtype=np.float32
                )

                data_obj.create_dataset(
                    'done_flg',
                    shape=(capacity, 1),
                    dtype=np.float32
                )
        return
    
    def _toNumpy(self, data_list):
        if isinstance(data_list, dict):
            return {key: self._toNumpy(value) for key, value in data_list.items()}
        elif isinstance(data_list, list):
            return [self._toNumpy(item) for item in data_list]
        elif isinstance(data_list, torch.Tensor):
            return data_list.cpu().numpy()
        else:
            return data_list
    
    def __getitem__(self, id):
        file_id = id // self.file_capacity + 1
        tmp_id = id % self.file_capacity

        if file_id not in self.hdf5_obj_map:
            self.hdf5_obj_map[file_id] = h5py.File(self.data_file_path_map[file_id], 'a')
        
        data_obj = self.hdf5_obj_map[file_id]

        return {
            'state': {
                'intersection': data_obj['state/intersection'][tmp_id],
                'roads': {
                    f"road_{road_id}": {
                        'road': data_obj[f"state/roads/road_{road_id}/road"][tmp_id],
                        'lanes': {
                            f"lane_{lane_id}": {
                                'lane': data_obj[f"state/roads/road_{road_id}/lanes/lane_{lane_id}/lane"][tmp_id],
                                'vehicles': data_obj[f"state/roads/road_{road_id}/lanes/lane_{lane_id}/vehicles"][tmp_id]
                            } for lane_id in range(1, self.num_lanes_map[road_id] + 1)
                        }
                    } for road_id in range(1, self.num_roads + 1)
                }
            },
            'action': data_obj['action'][tmp_id],
            'cumulative_reward': data_obj['cumulative_reward'][tmp_id],
            'next_state': {
                'intersection': data_obj['next_state/intersection'][tmp_id],
                'roads': {
                    f"road_{road_id}": {
                        'road': data_obj[f"next_state/roads/road_{road_id}/road"][tmp_id],
                        'lanes': {
                            f"lane_{lane_id}": {
                                'lane': data_obj[f"next_state/roads/road_{road_id}/lanes/lane_{lane_id}/lane"][tmp_id],
                                'vehicles': data_obj[f"next_state/roads/road_{road_id}/lanes/lane_{lane_id}/vehicles"][tmp_id]
                            } for lane_id in range(1, self.num_lanes_map[road_id] + 1)
                        }
                    } for road_id in range(1, self.num_roads + 1)
                }
            },
            'done_flg': data_obj['done_flg'][tmp_id],   
        }
    
    def __setitem__(self, id, learning_data):
        file_id = id // self.file_capacity + 1
        tmp_id = id % self.file_capacity

        if file_id not in self.hdf5_obj_map:
            self.hdf5_obj_map[file_id] = h5py.File(self.data_file_path_map[file_id], 'a')
        
        data_obj = self.hdf5_obj_map[file_id]

        # update state and next_state
        for state_type in ['state', 'next_state']:
            data_obj[f"{state_type}/intersection"][tmp_id] = learning_data[state_type]['intersection']
            for road_id in range(1, self.num_roads + 1):
                data_obj[f"{state_type}/roads/road_{road_id}/road"][tmp_id] = learning_data[state_type]['roads'][f"road_{road_id}"]['road']
                for lane_id in range(1, self.num_lanes_map[road_id] + 1):
                    data_obj[f"{state_type}/roads/road_{road_id}/lanes/lane_{lane_id}/lane"][tmp_id] = learning_data[state_type]['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['lane']
                    data_obj[f"{state_type}/roads/road_{road_id}/lanes/lane_{lane_id}/vehicles"][tmp_id] = learning_data[state_type]['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['vehicles']
        
        # update action, reward and done_flg
        data_obj['action'][tmp_id] = learning_data['action']
        data_obj['cumulative_reward'][tmp_id] = learning_data['cumulative_reward']
        data_obj['done_flg'][tmp_id] = learning_data['done_flg']

        return
    
    def __len__(self):
        return self.current_size
    
    def __del__(self):
        self.flush()
        self.close()
        return
    
    def update(self, id_list, data_list):
        data_list = self._toNumpy(data_list)
        for id, data in zip(id_list, data_list):
            self[id] = data
        return
    
    def flush(self):
        for hdf5_obj in self.hdf5_obj_map.values():
            hdf5_obj.flush()
        return
    
    def close(self):
        for hdf5_obj in self.hdf5_obj_map.values():
            hdf5_obj.close()
        self.hdf5_obj_map = {}
        return
    

    
        