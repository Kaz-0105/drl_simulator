from libs.common import Common
from libs.sum_tree import SumTree

class ReplayBuffer (Common):
    def __init__(self, controller):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトと上位の紐づくオブジェクトを取得
        self.config = controller.config
        self.executor = controller.executor
        self.controller = controller

        # ネットワークと紐づける
        self.model = controller.model
        self.model.set('replay_buffer', self)

        # バッファのサイズとバッチサイズを取得
        apex_info = self.config.get('apex_info')
        self.max_size = apex_info['buffer']['size']
        self.batch_size = apex_info['buffer']['batch_size']

        # データのコンテナを初期化
        self.sum_tree = SumTree(self.max_size)
    
    @property
    def current_size(self):
        return self.sum_tree.current_size
    
    def push(self, learning_data):
        for tmp_data in learning_data:
            self.sum_tree.add(tmp_data)
            
    def sample(self):
        return self.sum_tree.sample(self.batch_size)

    def update(self, indices, priorities):
        self.sum_tree.update_priority(indices, priorities)