from libs.common import Common
from libs.container import Container
from libs.object import Object
from libs.replay_buffer import ReplayBuffer
from objects.intersections import Intersections
from objects.local_agents import LocalAgents
from neural_networks.apex_net import QNet

from pathlib import Path
import torch

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
        self.makeIntersectionConnections(num_lanes_turple)

        # 車線数の情報と自動車台数の情報を取得
        self.makeNumLanesMap(num_lanes_turple)
        drl_info = self.config.get('drl_info')
        self.num_vehicles = drl_info['num_vehicles']

        # 使用する強化学習の手法で分岐
        self.makeModel()

        # 強化学習関連のハイパーパラメータを取得
        apex_info = self.config.get('apex_info')
        self.gamma = apex_info['gamma']
        self.learning_rate = apex_info['learning_rate']
        
        # LocalAgentオブジェクトを初期化
        self.local_agents = LocalAgents(self)

    def makeIntersectionConnections(self, num_lanes_turple):
        # intersection_listを取得
        intersection_list = self.master_agents.intersections_map[num_lanes_turple]

        # intersectionsオブジェクトを初期化
        self.intersections = Intersections(self)

        # intersectionオブジェクトと紐づける
        for intersection in intersection_list:
            self.intersections.add(intersection)
            intersection.set('master_agent', self)

    def makeNumLanesMap(self, num_lanes_turple):
        # 車線数のリストを少し整形
        num_lanes_map = {}
        for num_lanes in num_lanes_turple:
            num_lanes_map[len(num_lanes_map) + 1] = num_lanes

        self.num_lanes_map = num_lanes_map

    def makeModel(self):
        if self.config.get('drl_info')['method'] =='apex':
            # モデルを初期化
            self.model = QNet(self.config, self.num_vehicles, self.num_lanes_map)

            # 学習用にセット
            self.model.train()

            # 過去に学習済みの場合はそれを読み込む
            self.loadModel()
            
            # 経験再生用のバッファを初期化
            self.replay_buffer = ReplayBuffer(self)

    def loadModel(self):
        # model_pathを定義
        num_lanes_str = ''
        for num_lanes in self.num_lanes_map.values():
            num_lanes_str += str(num_lanes)
        num_vehs_str = str(self.num_vehicles)
        self.model_path = Path('models/apex_'+ num_lanes_str + '_' + num_vehs_str + '.pth')

        # 存在する場合は読み込む
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path))

    




