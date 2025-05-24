from collections import deque
import random

class ReplayBuffer:
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
        drl_info = self.config.get('drl_info')
        self.size = drl_info['buffer']['size']
        self.batch_size = drl_info['buffer']['batch_size']

        # データのコンテナを初期化
        self.container = deque(maxlen=self.size)
    
    def push(self, data):
        if self.model.__class__.__name__ == 'ApeXNet':
            self.container.append((data['state'], data['action'], data['reward'], data['next_state'], data['done']))
    
    def sample(self):
        return random.sample(self.container, self.batch_size)