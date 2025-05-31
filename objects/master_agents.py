from libs.container import Container
from libs.object import Object
from libs.replay_buffer import ReplayBuffer
from objects.intersections import Intersections
from objects.local_agents import LocalAgents
from neural_networks.apex_net import QNet
from neural_networks.apex_net2 import QNet2
from neural_networks.apex_net3 import QNet3

from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn

class MasterAgents(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトを取得
        self.config = network.config
        self.executor = network.executor

        # 上位の紐づくオブジェクトを取得
        self.network = network

        # 要素オブジェクトを初期化
        self.makeElements()
    
    def makeElements(self):
        # intersectionsオブジェクトを取得
        intersections = self.network.intersections
        self.intersections_map = {}
        for intersection in intersections.getAll():
            # 車線数のリストを取得
            num_lanes_turple = intersection.getNumLanesTurple()

            if num_lanes_turple not in self.intersections_map:
                # 車線数のリストをキーにしてMasterAgentオブジェクトを初期化
                self.intersections_map[num_lanes_turple] = []
            
            self.intersections_map[num_lanes_turple].append(intersection)
        
        for num_lanes_turple in self.intersections_map.keys():
            # master_agentオブジェクトを初期化
            self.add(MasterAgent(self, num_lanes_turple))

    def saveLearningData(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.saveLearningData)
        
        self.executor.wait()
    
    def train(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.train)
        
        self.executor.wait()
    
    def updateLocalAgents(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.updateLocalAgents)
        
        self.executor.wait()
    
    def saveNetworkAndBuffer(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.saveNetworkAndBuffer)
        
        self.executor.wait()

class MasterAgent(Object):
    def __init__(self, master_agents, num_lanes_turple):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = master_agents.config
        self.executor = master_agents.executor

        # 上位オブジェクトを取得
        self.master_agents = master_agents

        # IDを設定
        self.id = self.master_agents.count() + 1

        # intersectionsオブジェクトと紐づける
        self._makeIntersectionConnections(num_lanes_turple)

        # 車線数の情報と自動車台数の情報を取得
        self._makeNumLanesMap(num_lanes_turple)
        drl_info = self.config.get('drl_info')
        self.num_vehicles = drl_info['num_vehicles']

        # 強化学習関連のハイパーパラメータを取得
        apex_info = self.config.get('apex_info')
        self.update_interval = apex_info['update_interval']
        self.gamma = apex_info['gamma']
        self.learning_rate = apex_info['learning_rate']
        self.td_steps = apex_info['td_steps']

        # 使用する強化学習の手法で分岐
        self._makeModel()
        
        # LocalAgentオブジェクトを初期化
        self.local_agents = LocalAgents(self)

        # 更新回数を初期化
        self.update_count = 0

    def _makeIntersectionConnections(self, num_lanes_turple):
        # intersection_listを取得
        intersection_list = self.master_agents.intersections_map[num_lanes_turple]

        # intersectionsオブジェクトを初期化
        self.intersections = Intersections(self)

        # intersectionオブジェクトと紐づける
        for intersection in intersection_list:
            self.intersections.add(intersection)
            intersection.set('master_agent', self)

    def _makeNumLanesMap(self, num_lanes_turple):
        # 車線数のリストを少し整形
        num_lanes_map = {}
        for num_lanes in num_lanes_turple:
            num_lanes_map[len(num_lanes_map) + 1] = num_lanes

        self.num_lanes_map = num_lanes_map

    def _makeModel(self):
        if self.config.get('drl_info')['method'] =='apex':
            # モデルを初期化（学習用にセット）
            # self.model = QNet(self.config, self.num_vehicles, self.num_lanes_map)
            # self.model = QNet2(self.config, self.num_vehicles, self.num_lanes_map)
            self.model = QNet3(self.config, self.num_lanes_map)
            self.model.train()

            # 過去に学習済みの場合はそれを読み込む
            self._loadModel()

            # ターゲットモデルを初期化（学習用と同期，推論用にセット）
            # self.target_model = QNet(self.config, self.num_vehicles, self.num_lanes_map)
            # self.target_model = QNet2(self.config, self.num_vehicles, self.num_lanes_map)
            self.target_model = QNet3(self.config, self.num_lanes_map)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

            # 最適化手法と評価関数を定義
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # 経験再生用のバッファを初期化
            self.replay_buffer = ReplayBuffer(self)

    def _loadModel(self):
        # model_pathを定義
        num_lanes_str = ''
        for num_lanes in self.num_lanes_map.values():
            num_lanes_str += str(num_lanes)
        num_vehs_str = str(self.num_vehicles)
        # self.model_path = Path('models/apex_qnet_'+ num_lanes_str + '_' + num_vehs_str + '.pth')
        self.model_path = Path('models/apex_qnet2_' + num_lanes_str + '_' + num_vehs_str + '.pth')

        # 存在する場合は読み込む
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path))

    def saveLearningData(self):
        # ローカルエージェントを走査
        for local_agent in self.local_agents.getAll():
            # 学習データを取得
            learning_data = local_agent.get('learning_data')
            
            # 学習データがない場合はスキップ
            if not learning_data:
                continue

            # バッファーにデータを保存
            self.replay_buffer.push(learning_data)

            # データをクリア
            local_agent.set('learning_data', [])
    
    def train(self):
        drl_info = self.config.get('drl_info')
        if drl_info['method'] == 'apex':
            # バッファーのサイズが十分でない場合は学習しない
            if self.replay_buffer.get('current_size') < self.replay_buffer.get('batch_size'):
                return
            
            # バッファーからデータを取得
            batch_data, batch_data_indices = self.replay_buffer.sample()

            # 勾配を初期化
            self.optimizer.zero_grad()

            # Q値を計算
            actions = torch.tensor([tmp_data[1] - 1 for tmp_data in batch_data], dtype=torch.int64).unsqueeze(1)
            q_values_all = self.model([tmp_data[0] for tmp_data in batch_data])
            q_values = q_values_all.gather(1, actions) 
            
            # TDターゲットを計算（Double DQNの実装）
            self.model.eval()
            with torch.no_grad():
                max_actions = torch.argmax(self.model([tmp_data[3] for tmp_data in batch_data]), dim=1)
                target_q_values_all = self.target_model([tmp_data[3] for tmp_data in batch_data])
                target_q_values = target_q_values_all.gather(1, max_actions.unsqueeze(1))
                dones = torch.tensor([tmp_data[4] for tmp_data in batch_data], dtype=torch.float32).unsqueeze(1)
                td_targets = (1 - dones) * (self.gamma ** self.td_steps) * target_q_values
                td_targets += torch.tensor([tmp_data[2] for tmp_data in batch_data], dtype=torch.float32).unsqueeze(1)           
            self.model.train()

            # 損失を計算
            loss = self.criterion(q_values, td_targets)

            # 勾配を計算
            loss.backward()

            # パラメータを更新
            self.optimizer.step()

            # 更新回数をインクリメント
            self.update_count = (self.update_count + 1) % self.update_interval

            # ターゲットモデルを更新
            if self.update_count == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # 優先度を計算（経験再生用のバッファーに保存するため）
            priorities = torch.abs(q_values - td_targets).detach().numpy()
            self.replay_buffer.update(batch_data_indices, priorities)
            
    def updateLocalAgents(self):
        drl_info = self.config.get('drl_info')
        if drl_info['method'] == 'apex':
            # 同期のタイミングではないときはスキップ
            if self.update_count != 0:
                return
            
            # ローカルエージェントを走査
            for local_agent in self.local_agents.getAll():
                local_agent.model.load_state_dict(self.model.state_dict())
    
    def saveNetworkAndBuffer(self):
        # モデルを保存
        torch.save(self.model.state_dict(), self.model_path)

        # バッファーを保存
        self.replay_buffer.save()




            
    




