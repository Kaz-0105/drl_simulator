from libs.common import Common
from libs.torch_module import ExtendedDataset


import h5py
from tqdm import tqdm
import numpy as np
import random
import math

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
    
    def _initProps(self):
        # set size, file_size
        drl_info = self.config.get('drl_info')
        self.size = drl_info['framework']['apex']['buffer']['size']
        self.file_capacity = drl_info['framework']['apex']['buffer']['file_capacity']
        self.reset_flg = drl_info['framework']['apex']['buffer']['priority']['reset']['flg']
        if self.reset_flg:
            self.reset_interval = drl_info['framework']['apex']['buffer']['priority']['reset']['interval']
        
        self.threshold = drl_info['learning']['threshold']

        # set buffer_file_path_map
        buffer_dir_path = self.master_agent.get('save_dir_path_map')['buffer']
        self.buffer_file_path_map = {
            'tree': buffer_dir_path / 'tree.h5',
            'meta': buffer_dir_path / 'meta.h5',
            'data': {
                idx: buffer_dir_path / f"data_{idx}.h5" for idx in range(1, math.ceil(self.file_capacity / 1000) + 1)
            }
        }

        # set new_data_count and next_data_id
        if self.simulation_count > 1:
            saved_buffer = self.shared_resources.get('buffer')
            self.new_data_count = saved_buffer.get('new_data_count')
            self.next_data_id = saved_buffer.get('next_data_id')

        elif self.buffer_file_path_map['meta'].exists():
            meta_obj = h5py.File(self.buffer_file_path_map['meta'], 'r')
            self.new_data_count = meta_obj.attrs['new_data_count']
            self.next_data_id = meta_obj.attrs['next_data_id']
            meta_obj.close()

        else:
            self.new_data_count = 0
            self.next_data_id = 0   
        
        # initialize change_flg
        self.change_flg = False
        return
    
    def _resetPriority(self):
        if not self.priority_reset_flg:
            return
        
        if self.priority_reset_type == 'episode' and self.episode == self.priority_reset_episode:
            self.sum_tree.resetPriority()
        
        elif self.priority_reset_type == 'interval' and (self.episode % self.priority_reset_interval == 0):
            self.sum_tree.resetPriority()

        return

    def push(self, learning_data):
        for tmp_data in learning_data:
            self.sum_tree.add(tmp_data)
            self.new_data_count += 1

        return
            
    def sample(self):
        data, data_indices = self.sum_tree.sample(self.batch_size * self.num_batches)

        batch_data = []
        for idx in range(self.num_batches):
            if (idx + 1) * self.batch_size > len(data):
                tmp_data = data[idx * self.batch_size:]
                tmp_data_indices = data_indices[idx * self.batch_size:]
                batch_data.append((tmp_data, tmp_data_indices))
                break
            else:
                tmp_data = data[idx * self.batch_size: (idx + 1) * self.batch_size]
                tmp_data_indices = data_indices[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_data.append((tmp_data, tmp_data_indices))
            
            valid_data = []
            valid_indices = []
            for idx in range(len(tmp_data)):
                if tmp_data[idx] is not None:
                    valid_data.append(tmp_data[idx])
                    valid_indices.append(tmp_data_indices[idx])
        
        return batch_data
    
    def update(self):
        # reset new_data_count
        self.new_data_count %= self.threshold

        # get learning_data_list
        learning_data_list = self.master_agent.get('learning_data_list')

        


        # clear learning_data_list
        self.master_agent.clearLearningData()
        if change_flg:
            self.buffer.showInfo()
        return

    def update(self, indices, priorities):
        self.sum_tree.update_priority(indices, priorities)
        return

    def save(self):
        # update pickle files
        if self.finish_flg or self.simulation_count % self.save_interval == 0:
            with open(self.tree_file_path, 'wb') as f:
                saved_data = {
                    'tree': self.sum_tree.get('tree'),
                    'next_data_idx': self.sum_tree.get('next_data_idx'),
                    'current_size': self.sum_tree.get('current_size'),
                    'new_data_count': self.new_data_count,
                }
                pickle.dump(saved_data, f)

            data = self.sum_tree.get('data')
            for data_id in tqdm(range(1, len(self.data_file_path_map) + 1)):
                data_file_path = self.data_file_path_map[data_id]
                with open(data_file_path, 'wb') as f:
                    pickle.dump(data[1000 * (data_id - 1): 1000 * data_id], f)
        return 
    
    def showInfo(self):
        print('==============================================')
        print(f"status: buffer update")
        print(f"number of new data: {self.new_data_count}/{self.threshold}")
        print(f"current buffer size: {self.current_size}/{self.max_size}")
        return

    @property
    def current_size(self):
        return self.dataset.get('current_size')
    
    @property
    def enough_new_data_flg(self):
        if not self.change_flg:
            return False

        if self.new_data_count < self.num_new_data:
            return False

        if self.current_size < self.batch_size * self.num_batches:
            return False
        
        return True
        
    @property
    def simulation_count(self):
        master_agent = self.master_agent
        network = master_agent.network
        vissim = network.vissim
        return vissim.get('simulation_count')
    
    @property
    def finish_flg(self):
        return self.master_agent.get('finish_flg')

    
    @property
    def episode(self):
        return self.master_agent.get('episode')
        
        
class SumTree (Common):
    def __init__(self, buffer):
        super().__init__()

        self.buffer = buffer
        self.config = buffer.config
        self.shared_resources = buffer.shared_resources 

        self._initProps()
        return
    
    @property
    def dataset(self):
        return self.buffer.dataset
    
    @property
    def current_size(self):
        return self.buffer.dataset.get('current_size')
    
    @property
    def next_data_id(self):
        return self.buffer.get('next_data_id')
    
    @property
    def total_priority(self):
        return self.tree_array[0]
    
    def _initProps(self):
        # set initial priority and size
        drl_info = self.config.get('drl_info')
        self.initial_priority = drl_info['framework']['apex']['buffer']['priority']['initial_value']
        self.size = self.buffer.get('size')

        self.buffer_file_path_map = self.buffer.get('buffer_file_path_map')

        # set num_leaves (the number of leaf nodes in tree_array)
        self.num_leaves = 2**math.ceil(math.log2(self.size))

        # initialize tree_array and next_data_id (0 <= next_data_id < size)
        if self.buffer.get('simulation_count') > 1:
            saved_sum_tree = self.shared_resources.buffer.sum_tree
            self.tree_array = saved_sum_tree.get('tree_array')

        elif self.buffer_file_path_map['tree'].exists() and self.buffer_file_path_map['meta'].exists():
            tree_obj = h5py.File(self.buffer_file_path_map['tree'], 'r')
            self.tree_array = tree_obj[:]
            tree_obj.close()

        else:
            self.tree_array = np.zeros(2 * self.num_leaves - 1, dtype=np.float32)
    
        return

    def _propagate(self, tree_id, change):
        parent_id = (tree_id - 1) // 2
        self.tree_array[parent_id] += change

        if parent_id != 0:
            self._propagate(
                tree_idx=parent_id, 
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
    
    def add(self, tmp_data, priority = None):
        # if priority is not specified, use the initial priority
        if priority is None:
            priority = self.initial_priority

        # get the tree index for the new data priority
        target_tree_id = self.next_data_id + self.num_leaves - 1

        # calculate the change
        change = priority - self.tree_array[target_tree_id]

        # update tree
        self.tree_array[target_tree_id] = priority
        self._propagate(target_tree_id, change)

        # push data to dataset
        self.dataset[self.next_data_id] = tmp_data

        # update current size
        if self.current_size < self.size:
            self.current_size += 1

        # update next data index
        self.next_data_id = (self.next_data_id + 1) % self.size
        return
    
    def sample(self, number):
        # if the current size is smaller than the number of samples, return all data and corresponding indices
        if self.current_size < number:
            data = self.dataset[:self.current_size]
            data_indices = list(range(self.current_size))
            return data, data_indices

        # ランダムに0-total_priorityの範囲でサンプリング
        sample_values = np.random.uniform(0, self.total_priority, number)

        # 対応するデータのツリーのインデックスを取得
        tree_indices = [self._retrieve(0, sample_value) for sample_value in sample_values]

        # データのインデックスを取得
        data_indices = []
        for tree_idx in tree_indices:
            data_idx = tree_idx - self.num_leaves + 1
            if data_idx + 1 > self.current_size:
                data_idx = random.randint(0, self.current_size - 1)
            data_indices.append(data_idx)

        # ツリーのインデックスからデータを取得
        data = []
        for data_idx in data_indices:
            data.append(self.dataset[data_idx])


        return data, data_indices
    
    def update_priority(self, data_indices, new_priorities):
        for data_idx, new_priority in zip(data_indices, list(new_priorities)):
            # validation
            if data_idx < 0 or data_idx >= self.current_size:
                continue
            
            # ツリーのインデックスを計算
            tree_idx = data_idx + self.num_leaves - 1

            # 差分を計算後に葉ノードの値を更新
            change = new_priority.item() - self.tree_array[tree_idx].item()
            self.tree_array[tree_idx] = new_priority.item()

            # 親ノードの値を順に更新
            self._propagate(tree_idx, change)

    def resetPriority(self):
        for data_idx in range(self.current_size):
            tree_idx = data_idx + self.num_leaves - 1
            change = self.initial_priority - self.tree_array[tree_idx]
            self.tree_array[tree_idx] = self.initial_priority

            self._propagate(tree_idx, change.item())
        
        return
    
class Dataset(ExtendedDataset):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self.config = buffer.config

        self._initProps()
        self._makeDataFiles()
        return
    
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
        self.current_size = 0  
        return
    
    def _makeDataFiles(self):
        data_file_ids = sorted(list(self.data_file_path_map.keys()))
        for data_file_id in data_file_ids:
            data_file_path = self.data_file_path_map[data_file_id]
            if data_file_path.exists():
                continue
            
            with h5py.File(data_file_path, 'w') as data_obj:
                if data_file_id != data_file_ids[-1]:
                    capacity = self.file_capacity
                else:
                    capacity = self.size - self.file_capacity * (len(data_file_ids) - 1)
                
                data_obj.create_dataset(
                    'phases',
                    shape=(capacity, self.num_features_map['phase'][self.num_roads]),
                    dtype=np.float32                    
                )
                roads_group = data_obj.create_group('roads')
                for road_id in range(1, self.num_roads + 1):
                    road_group = roads_group.create_group(f"road_{road_id}")
                    road_group.create_dataset(
                        'road',
                        shape=(capacity, self.num_features_map['road']),
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
        return
    
    def __getitem__(self, id):
        file_id = id // self.file_capacity + 1
        tmp_id = id % self.file_capacity

        if file_id not in self.hdf5_obj_map:
            self.hdf5_obj_map[file_id] = h5py.File(self.data_file_path_map[file_id], 'r')
        
        data_obj = self.hdf5_obj_map[file_id]

        return {
            'phases': data_obj['phases'][tmp_id],
            'roads': {
                f"road_{road_id}": {
                    'road': data_obj[f"roads/road_{road_id}/road"][tmp_id],
                    'lanes': {
                        f"lane_{lane_id}": {
                            'lane': data_obj[f"roads/road_{road_id}/lanes/lane_{lane_id}/lane"][tmp_id],
                            'vehicles': data_obj[f"roads/road_{road_id}/lanes/lane_{lane_id}/vehicles"][tmp_id]
                        } for lane_id in range(1, self.num_lanes_map[road_id] + 1)
                    }
                } for road_id in range(1, self.num_roads + 1)
            }
        }
    
    def __setitem__(self, id, data):
        file_id = id // self.file_capacity + 1
        tmp_id = id % self.file_capacity

        if file_id not in self.hdf5_obj_map:
            self.hdf5_obj_map[file_id] = h5py.File(self.data_file_path_map[file_id], 'a')
        
        data_obj = self.hdf5_obj_map[file_id]

        data_obj['phases'][tmp_id] = data['phases']
        for road_id in range(1, self.num_roads + 1):
            data_obj[f"roads/road_{road_id}/road"][tmp_id] = data['roads'][f"road_{road_id}"]['road']
            for lane_id in range(1, self.num_lanes_map[road_id] + 1):
                data_obj[f"roads/road_{road_id}/lanes/lane_{lane_id}/lane"][tmp_id] = data['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['lane']
                data_obj[f"roads/road_{road_id}/lanes/lane_{lane_id}/vehicles"][tmp_id] = data['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['vehicles']
        return
    
    def __len__(self):
        return self.current_size
    
    def close(self):
        for hdf5_obj in self.hdf5_obj_map.values():
            hdf5_obj.flush()
            hdf5_obj.close()
        
        self.hdf5_obj_map = {}
        return
    

    
        