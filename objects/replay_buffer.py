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
        self.path = self.master_agent.get('replay_buffer_path')
        self._load()
    
    def _getBufferInfo(self):
        apex_info = self.config.get('apex_info')
        self.max_size = apex_info['buffer']['size']
        self.batch_size = apex_info['buffer']['batch_size']
        return

    def _load(self):
        # バッファーのファイルが存在する場合は読み込む
        if not self.path.exists():
            return
        
        with self.path.open('rb') as f:
            loaded_data = pickle.load(f)
            self.sum_tree.set('tree', loaded_data['tree'])
            self.sum_tree.set('data', loaded_data['data'])
            self.sum_tree.set('next_data_idx', loaded_data['next_data_idx'])
            self.sum_tree.set('current_size', loaded_data['current_size'])
        return
    
    def push(self, learning_data):
        for tmp_data in learning_data:
            self.sum_tree.add(tmp_data)
        return
            
    def sample(self):
        return self.sum_tree.sample(self.batch_size)

    def update(self, indices, priorities):
        self.sum_tree.update_priority(indices, priorities)
        return

    def save(self):
        with self.path.open('wb') as f:
            saved_data = {
                'tree': self.sum_tree.get('tree'),
                'data': self.sum_tree.get('data'),
                'next_data_idx': self.sum_tree.get('next_data_idx'),
                'current_size': self.sum_tree.get('current_size')
            }
            pickle.dump(saved_data, f)
        return 

    @property
    def current_size(self):
        return self.sum_tree.get('current_size')
        
