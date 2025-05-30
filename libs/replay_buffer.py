from libs.common import Common
from libs.sum_tree import SumTree

from pathlib import Path

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
        apex_info = self.config.get('apex_info')
        self.max_size = apex_info['buffer']['size']
        self.batch_size = apex_info['buffer']['batch_size']
        self.alpha = apex_info['buffer']['alpha']
        self.epsilon = apex_info['buffer']['epsilon']

        # データのコンテナを初期化
        self.sum_tree = SumTree(self.max_size)

        self.loadBuffer()
    
    def loadBuffer(self):
        # バッファーのファイルパスを取得
        num_lanes_str = ''
        num_lanes_map = self.master_agent.get('num_lanes_map')
        for num_lanes in num_lanes_map.values():
            num_lanes_str += str(num_lanes)
        num_vehs_str = str(self.master_agent.get('num_vehicles'))
        self.buffer_path = Path('buffers/apex_buffer_' + num_lanes_str + '_' + num_vehs_str + '.pkl')

        # バッファーのファイルが存在する場合は読み込む
        if self.buffer_path.exists():
            self.sum_tree.load(self.buffer_path)
    
    @property
    def current_size(self):
        return self.sum_tree.current_size
    
    def push(self, learning_data):
        for tmp_data in learning_data:
            self.sum_tree.add(tmp_data)
            
    def sample(self):
        return self.sum_tree.sample(self.batch_size)

    def update(self, indices, priorities):
        priorities = (priorities + self.epsilon) ** self.alpha
        self.sum_tree.update_priority(indices, priorities)

    def save(self):
        self.sum_tree.save(self.buffer_path)
        
