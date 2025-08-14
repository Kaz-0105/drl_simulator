from libs.common import Common
from libs.sum_tree import SumTree

import pickle
from tqdm import tqdm
import numpy as np

class ReplayBuffer (Common):
    def __init__(self, master_agent):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトを取得
        self.config = master_agent.config
        self.executor = master_agent.executor
        self.shared_resources = master_agent.shared_resources

        # 上位の紐づくオブジェクトを取得
        self.master_agent = master_agent

        # ネットワークと紐づける
        self.model = master_agent.model
        self.model.set('replay_buffer', self)

        # バッファのサイズとバッチサイズを取得
        self._getBufferInfo()

        # バッファのパスを取得
        self.path_map = self.master_agent.get('replay_buffer_path_map')

        # バッファのデータをロード
        self._loadData()
        return
    
    def _getBufferInfo(self):
        apex_info = self.config.get('apex_info')
        self.num_data_for_learning = apex_info['buffer']['num_data_for_learning']
        self.max_size = apex_info['buffer']['size']
        self.num_batches = apex_info['buffer']['batch']['number']
        self.batch_size = apex_info['buffer']['batch']['size']
        self.initial_priority = apex_info['buffer']['initial_priority']
        self.priority_reset_flg = apex_info['buffer']['priority_reset_flg']
        return

    def _loadData(self):
        # 最初のエピソードかどうかで分岐
        if self.simulation_count == 1:
            # データのコンテナを初期化
            self.sum_tree = SumTree(self.max_size, self.initial_priority)

            # カウンタを初期化
            self.new_data_count = 0

            # バッファが保存されていない場合はロードは必要ない
            if not self.path_map['tree'].exists():
                return
            
            # 学習を行わない場合はロードしない
            if not self.master_agent.get('learning_flg'):
                return
            
            with self.path_map['tree'].open('rb') as f:
                loaded_data = pickle.load(f)
                self.sum_tree.set('tree', loaded_data['tree'])
                self.sum_tree.set('next_data_idx', loaded_data['next_data_idx'])
                self.sum_tree.set('current_size', loaded_data['current_size'])
                self.new_data_count = loaded_data['new_data_count']

            data = []
            for data_path in tqdm(self.path_map['data']):
                with data_path.open('rb') as f:
                    loaded_data = pickle.load(f)
                    data.extend(loaded_data['data'])
            self.sum_tree.set('data', data)

        else:
            # shared_resourcesオブジェクトからデータを取得
            self.sum_tree = self.shared_resources.get('sum_tree')
            self.new_data_count = self.shared_resources.get('new_data_count')

        # 優先度のリセットを行う（フラグが立っている場合）
        if self.priority_reset_flg:
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

    def update(self, indices, priorities):
        self.sum_tree.update_priority(indices, priorities)
        return

    def save(self):
        # 学習を行わない場合は保存しない
        if not self.master_agent.get('learning_flg'):
            return
        
        # pklファイルを更新
        simulator_info = self.config.get('simulator_info')
        drl_info = self.config.get('drl_info')
        if self.simulation_count == simulator_info['simulation_count'] or self.simulation_count % drl_info['buffer_save_interval'] == 0:
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
    
    def updateInitialPriority(self, losses):
        max_loss = np.max(losses)
        self.sum_tree.set('initial_priority', max_loss * 1.1)
        return

    @property
    def current_size(self):
        return self.sum_tree.get('current_size')
    
    @property
    def should_learn_flg(self):
        print(f"ReplayBuffer: New data count[{self.new_data_count}/{self.num_data_for_learning}], Data size[{self.current_size}/{self.max_size}]")

        # 新しいデータが十分に溜まったら学習を行う
        if self.new_data_count >= self.num_data_for_learning:
            self.new_data_count %= self.num_data_for_learning
            return True
        else:
            return False
        
    @property
    def simulation_count(self):
        # vissimオブジェクトを取得
        master_agent = self.master_agent
        network = master_agent.network
        vissim = network.vissim

        return vissim.get('simulation_count')
        
        
