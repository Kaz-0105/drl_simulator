from libs.common import Common

import pickle
from tqdm import tqdm
import numpy as np
import random
import math

class ReplayBuffer (Common):
    def __init__(self, master_agent):
        super().__init__()

        # set config, executor, shared_resources and master_agent
        self.config = master_agent.config
        self.executor = master_agent.executor
        self.shared_resources = master_agent.shared_resources
        self.master_agent = master_agent

        # connect to model
        self.model = master_agent.model
        self.model.set('replay_buffer', self)

        # set properties
        self._initProps()
        return
    
    def _initProps(self):
        # set apex information
        apex_info = self.config.get('apex_info')
        self.num_new_data = apex_info['buffer']['num_new_data']
        self.max_size = apex_info['buffer']['size']
        self.num_batches = apex_info['buffer']['batch']['number']
        self.batch_size = apex_info['buffer']['batch']['size']
        self.initial_priority = apex_info['buffer']['initial_priority']
        self.priority_reset_flg = apex_info['buffer']['priority_reset']['flg']
        self.priority_reset_type = apex_info['buffer']['priority_reset']['type']
        self.priority_reset_episode = apex_info['buffer']['priority_reset']['episode']
        self.priority_reset_interval = apex_info['buffer']['priority_reset']['interval']

        drl_info = self.config.get('drl_info')
        self.learning_flg = drl_info['learning_flg']
        self.save_interval = drl_info['save_interval']

        simulator_info = self.config.get('simulator_info')
        self.num_simulations = simulator_info['num_simulations']

        # set buffer paths
        buffer_dir_path = self.master_agent.get('save_dir_path_map')['buffer']
        self.tree_file_path = buffer_dir_path / 'tree.pkl'
        self.data_file_path_map = {idx + 1: buffer_dir_path / f"data_{idx + 1}.pkl" for idx in range(math.ceil(self.max_size / 1000))}

        self.change_flg = False
        return
    
    def load(self):
        if self.simulation_count > 1:
            self.sum_tree = self.shared_resources.get('sum_tree')
            self.new_data_count = self.shared_resources.get('new_data_count')
            self._resetPriority()
            return
        
        self.sum_tree = SumTree(self)
        self.new_data_count = 0

        if not self.tree_file_path.exists():
            return
        
        if not self.learning_flg:
            return
        
        with open(self.tree_file_path, 'r', encoding='utf-8') as f:
            loaded_data = pickle.load(f)

        self.sum_tree.set('tree', loaded_data['tree'])
        self.sum_tree.set('next_data_idx', loaded_data['next_data_idx'])
        self.sum_tree.set('current_size', loaded_data['current_size'])
        self.new_data_count = loaded_data['new_data_count']

        data = []
        for idx in tqdm(range(len(self.data_file_path_map))):
            data_file_path = self.data_file_path_map[idx + 1]
            if not data_file_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_file_path}")
            
            with open(data_file_path, 'r', encoding='utf-8') as f:
                loaded_data = pickle.load(f)
            
            data.extend(loaded_data['data'])
        self.sum_tree.set('data', data)

        self._resetPriority()

        return
    
    def _resetPriority(self):
        if not self.priority_reset_flg:
            return
        
        if self.priority_reset_type == 'episode' and self.episode == self.priority_reset_episode:
            self.sum_tree.resetPriority()
        
        elif self.priority_reset_type == 'interval' and (self.episode % self.priority_reset_interval == 0):
            self.sum_tree.resetPriority()

        return
    
    def resetNewDataCount(self):
        self.new_data_count %= self.num_new_data
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

    def update(self, indices, priorities):
        self.sum_tree.update_priority(indices, priorities)
        return

    def save(self):
        if not self.learning_flg:
            return
        
        # pklファイルを更新
        if self.finish_flg or self.simulation_count % self.save_interval == 0:
            with self.path_map['tree'].open('wb') as f:
                saved_data = {
                    'tree': self.sum_tree.get('tree'),
                    'next_data_idx': self.sum_tree.get('next_data_idx'),
                    'current_size': self.sum_tree.get('current_size'),
                    'new_data_count': self.new_data_count,
                }
                pickle.dump(saved_data, f)

            data = self.sum_tree.get('data')
            for idx in tqdm(range(len(self.path_map['data']))):
                data_path = self.path_map['data'][idx]
                with data_path.open('wb') as f:
                    if idx == len(self.path_map['data']) - 1:
                        saved_data = {
                            'data': data[1000 * idx:]
                        }
                    else:
                        saved_data = {
                            'data': data[1000 * idx: 1000 * (idx + 1)]
                        }
                    pickle.dump(saved_data, f)
        
        # shared_resourcesオブジェクトに保存
        self.shared_resources.set('sum_tree', self.sum_tree)
        self.shared_resources.set('new_data_count', self.new_data_count)      
        return 
    
    def showInfo(self):
        print('==============================================')
        print(f"status: buffer update")
        print(f"number of new data: {self.new_data_count}/{self.num_new_data}")
        print(f"current buffer size: {self.current_size}/{self.max_size}")
        return

    @property
    def current_size(self):
        return self.sum_tree.get('current_size')
    
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
    def __init__(self, replay_buffer):
        # 継承
        super().__init__()

        self.replay_buffer = replay_buffer  
        self.initial_priority = self.replay_buffer.get('initial_priority')
        
        # ツリーに格納するデータ数の最大値を設定
        self.capacity = self.replay_buffer.get('max_size')
        self.num_leaves = 2**math.ceil(math.log2(self.capacity)) # 完全二分木のサイズに調整

        # ツリーのノードの重みを保存する配列を定義
        self.tree = np.zeros(2 * self.num_leaves - 1, dtype=np.float32)

        # 経験データ自体を格納する配列を定義
        self.data = [None] * self.capacity

        # 現在のデータ数を初期化
        self.current_size = 0

        # 次に経験を格納する位置
        self.next_data_idx = 0
        return

    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change

        if parent != 0: 
            self._propagate(parent, change)
    
    def _retrieve(self, tree_idx, random_value):
        left_child = 2 * tree_idx + 1
        right_child = left_child + 1

        if left_child >= len(self.tree):
            return tree_idx

        if random_value <= self.tree[left_child]:
            return self._retrieve(left_child, random_value)
        else:
            return self._retrieve(right_child, random_value - self.tree[left_child])
    
    def add(self, tmp_data, priority = None):
        # 優先度が指定されていない場合は直近のデータの平均をつかう
        if priority is None:
            priority = self.initial_priority

        # 優先度を更新するツリーのインデックスを計算
        tree_idx = self.next_data_idx + self.num_leaves - 1

        # 差分を計算後に葉ノードの値を更新
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # 親ノードの値を順に更新
        self._propagate(tree_idx, change.item())

        # データを保存
        self.data[self.next_data_idx] = tmp_data

        # 現在のデータ数を更新
        if self.current_size < self.capacity:
            self.current_size += 1

        # 次にデータを格納する位置を更新
        self.next_data_idx += 1
        self.next_data_idx %= self.capacity
        return
    
    def sample(self, size):
        # サンプリングサイズが現在のサイズより大きい場合は，全データを返す
        if self.current_size < size:
            data = self.data[:self.current_size]
            data_indices = list(range(self.current_size))
            return data, data_indices

        # ランダムに0-total_priorityの範囲でサンプリング
        sample_values = np.random.uniform(0, self.total_priority, size)

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
            data.append(self.data[data_idx])


        return data, data_indices
    
    def update_priority(self, data_indices, new_priorities):
        for data_idx, new_priority in zip(data_indices, list(new_priorities)):
            # validation
            if data_idx < 0 or data_idx >= self.current_size:
                continue
            
            # ツリーのインデックスを計算
            tree_idx = data_idx + self.num_leaves - 1

            # 差分を計算後に葉ノードの値を更新
            change = new_priority.item() - self.tree[tree_idx].item()
            self.tree[tree_idx] = new_priority.item()

            # 親ノードの値を順に更新
            self._propagate(tree_idx, change)

    def resetPriority(self):
        for data_idx in range(self.current_size):
            tree_idx = data_idx + self.num_leaves - 1
            change = self.initial_priority - self.tree[tree_idx]
            self.tree[tree_idx] = self.initial_priority

            self._propagate(tree_idx, change.item())
        
        return
    
    @property
    def total_priority(self):
        return self.tree[0]