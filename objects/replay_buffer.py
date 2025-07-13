from libs.common import Common
from libs.sum_tree import SumTree

import pickle

class ReplayBuffer (Common):
    def __init__(self, master_agent):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトと上位の紐づくオブジェクトを取得
        self.config = master_agent.config
        self.executor = master_agent.executor
        self.master_agent = master_agent

        # ネットワークと紐づける
        self.model = master_agent.model
        self.model.set('replay_buffer', self)

        # バッファのサイズとバッチサイズを取得
        self._getBufferInfo()

        # データのコンテナを初期化
        self.sum_tree = SumTree(self.max_size)

        # バッファのパスを取得
        self.path_map = self.master_agent.get('replay_buffer_path_map')
        self._load()

        return
    
    def _getBufferInfo(self):
        apex_info = self.config.get('apex_info')
        self.num_data_for_learning = apex_info['buffer']['num_data_for_learning']
        self.max_size = apex_info['buffer']['size']
        self.num_batches = apex_info['buffer']['batch']['number']
        self.batch_size = apex_info['buffer']['batch']['size']
        return

    def _load(self):
        # バッファーのファイルが存在する場合は読み込む
        if not self.path_map['tree'].exists():
            return
        
        with self.path_map['tree'].open('rb') as f:
            loaded_data = pickle.load(f)
            self.sum_tree.set('tree', loaded_data['tree'])
            self.sum_tree.set('next_data_idx', loaded_data['next_data_idx'])
            self.sum_tree.set('current_size', loaded_data['current_size'])
            self.new_data_count = loaded_data['new_data_count']

        data = []
        for data_path in self.path_map['data']:
            with data_path.open('rb') as f:
                loaded_data = pickle.load(f)
                data.extend(loaded_data['data'])
        self.sum_tree.set('data', data)
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
        with self.path_map['tree'].open('wb') as f:
            saved_data = {
                'tree': self.sum_tree.get('tree'),
                'next_data_idx': self.sum_tree.get('next_data_idx'),
                'current_size': self.sum_tree.get('current_size'),
                'new_data_count': self.new_data_count,
            }
            pickle.dump(saved_data, f)

        data = self.sum_tree.get('data')
        for idx in range(len(self.path_map['data'])):
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
        return 

    @property
    def current_size(self):
        return self.sum_tree.get('current_size')
    
    @property
    def should_learn_flg(self):
        print(f"ReplayBuffer: New data count[{self.new_data_count}/{self.num_data_for_learning}]")

        # 新しいデータが十分に溜まったら学習を行う
        if self.new_data_count >= self.num_data_for_learning:
            self.new_data_count %= self.num_data_for_learning
            return True
        else:
            return False
        
        
