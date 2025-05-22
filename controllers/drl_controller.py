from controllers.base_controller import BaseController
from objects.links import Lanes
import torch
import torch.nn as nn
from pathlib import Path
from libs.neural_network import NeuralNetwork
from collections import deque
import random

class DRLController(BaseController):
    def __init__(self, controllers, intersection):
        # 継承
        super().__init__(controllers)

        # intersectionオブジェクトと紐づける
        self.intersection = intersection
        intersection.set('controller', self)

        # roadオブジェクトおよびlaneオブジェクトと紐づける（一方通行）
        self.roads = self.intersection.input_roads
        self.makeRoadLanesMap()

        # drl_infoを取得
        self.drl_info = self.config.get('drl_info')

        # モデルの定義
        if self.config.get('drl_info')['method'] =='apex':
            self.model = ApeXNet(self)
            model_path = Path('models/apex.pth')

            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path))
            
            self.replay_buffer = ReplayBuffer(self)

    def makeRoadLanesMap(self):
        # road_lanes_mapを初期化
        road_lanes_map = {}

        # 道路を走査
        for road_order_id in self.roads.getKeys(container_flg=True):
            road = self.intersection.input_roads[road_order_id]
            lanes = Lanes(self)

            for link in road.links.getAll():
                if link.get('type') != 'right':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            for link in road.links.getAll():
                if link.get('type') != 'main':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            for link in road.links.getAll():
                if link.get('type') != 'left':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            road_lanes_map[road_order_id] = lanes

        self.road_lanes_map = road_lanes_map

    def run(self):
        # 状態量の取得
        self.makeStates()

        # 行動の選択
        q_values = self.model([self.states])

        print('test')

    def makeStates(self):
        if self.config.get('drl_info')['method'] == 'apex':
            # 状態量を初期化
            states = {}

            # 道路群の状態量を初期化
            roads_states = {}

            # 道路を走査
            for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
                # roadオブジェクトを取得
                road = self.roads[road_order_id]

                # 道路の状態量を初期化
                road_states = {}
                
                # 車線群の状態量を初期化
                lanes_states = {}

                # lanesオブジェクトを取得
                lanes = self.road_lanes_map[road_order_id]

                # 車線を走査
                for lane_order_id in lanes.getKeys(container_flg=True, sorted_flg=True):
                    # laneオブジェクトを取得
                    lane = lanes[lane_order_id]

                    # 車線の状態量を初期化
                    lane_states = {}

                    # vehicle_dataを位置情報でソート
                    vehicle_data = lane.get('vehicle_data')
                    vehicle_data.sort_values(by='position', ascending=False, inplace=True)
                    vehicle_data.reset_index(drop=True, inplace=True)

                    # 各車線で状態に使う自動車台数を取得
                    num_vehicles = self.drl_info['num_vehicles']
                    vehicle_data = vehicle_data.head(num_vehicles)

                    # 距離情報を信号との距離に変換
                    length_info = lane.get('length_info')
                    vehicle_data['position'] = length_info['length'] - vehicle_data['position']

                    # 車両に関する状態を取得
                    vehicles_states = {}
                    feature_names = ['position', 'speed', 'in_queue', 'direction']
                    for index in range(num_vehicles):
                        if index < vehicle_data.shape[0]:
                            # レコードを取得
                            vehicle = vehicle_data.iloc[index]

                            # 車両の状態量を初期化
                            vehicle_states = []

                            # 特徴量を走査
                            for feature_name in feature_names:
                                # 使わない状態量はスキップ
                                if self.drl_info['features']['vehicle'][feature_name] == False:
                                    continue

                                # 方向に関する状態量はone-hotベクトルに変換，それ以外はそのまま追加
                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.intersection.get('num_roads') - 1)
                                    direction_vector[int(vehicle['direction_id']) - 1] = 1
                                    vehicle_states.extend(direction_vector)
                                else: 
                                    vehicle_states.append(int(vehicle[feature_name]))
                            
                            # 自動車が存在するかどうかのフラグの状態量を追加
                            vehicle_states.append(1) 

                            # テンソルに変換してからvehicles_statesに追加  
                            vehicles_states[len(vehicles_states) + 1] = torch.tensor(vehicle_states).float()                    
                        else:
                            # 車両の状態量を初期化
                            vehicle_states = []

                            # 特徴量を走査
                            for feature_name in feature_names:
                                # 使わない状態量はスキップ
                                if self.drl_info['features']['vehicle'][feature_name] == False:
                                    continue
                                
                                # 方向に関する状態量はone-hotベクトルに変換，それ以外はそのまま追加
                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.intersection.get('num_roads') - 1)
                                    vehicle_states.extend(direction_vector)
                                else: 
                                    vehicle_states.append(0)
                            
                            # 自動車が存在するかどうかのフラグの状態量を追加
                            vehicle_states.append(0)

                            # テンソルに変換してからvehicles_statesに追加
                            vehicles_states[len(vehicles_states) + 1] = torch.tensor(vehicle_states).float()
                    
                    # 車線の状態量に追加
                    lane_states['vehicles'] = dict(sorted(vehicles_states.items()))

                    # 評価指標に関する状態量を取得
                    lane_states['metric'] = torch.tensor([lane.get('num_vehicles')], dtype=torch.float32)
                    
                    # 車線情報に関する状態量を取得（長さ，メインリンクかサブリンクか）
                    if lane.link.get('type') == 'main':
                        lane_states['shape'] = torch.tensor([int(length_info['length']), 1, 0], dtype=torch.float32)
                    elif lane.link.get('type') == 'right' or lane.link.get('type') == 'left':
                        lane_states['shape'] = torch.tensor([int(length_info['length']), 0, 1], dtype=torch.float32)

                    # lanes_statesにlane_statesを追加
                    lanes_states[lane_order_id] = lane_states
                
                # road_statesに車線の状態量を追加
                road_states['lanes'] = dict(sorted(lanes_states.items()))

                # 評価指標の状態量について
                metric_states = []
                metric_states.append(int(road.get('max_queue_length')))
                metric_states.append(int(road.get('average_delay')))

                # road_statesに評価指標の状態量を追加
                road_states['metric'] = torch.tensor(metric_states, dtype=torch.float32)

                # roads_statesにroad_statesを追加
                roads_states[road_order_id] = road_states
            
            # statesに道路の状態量を追加
            states['roads'] = dict(sorted(roads_states.items()))

            # 交差点の状態量について
            current_phase_id = self.intersection.get('current_phase_id')
            intersection_states = [0] * (self.intersection.get('num_phases'))
            intersection_states[current_phase_id - 1] = 1

            # statesに交差点の状態量を追加
            states['intersection'] = torch.tensor(intersection_states, dtype=torch.float32)

            # 状態量をインスタンス変数に保存
            self.states = states

    def getAction(self):
        pass

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

class ApeXNet(NeuralNetwork):
    def __init__(self, controller):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = controller.config
        self.executor = controller.executor
        self.controller = controller

        # 各サブネットワークを定義
        self.vehicle_net = ApeXVehicleNet(self)
        self.vehicles_net = ApeXVehiclesNet(self)
        self.lane_shape_net = ApeXLaneShapeNet(self)
        self.lane_metric_net = ApeXLaneMetricNet(self)
        self.lane_net = ApeXLaneNet(self)
        self.road_metric_net = ApeXRoadMetricNet(self)
        self.makeNumLaneRoadNetMap()
        self.phase_net = ApeXPhaseNet(self)
        self.intersection_net = ApeXIntersectionNet(self, self.controller.roads.count())

        # ネットワークの定義
        self.input_size = self.intersection_net.get('output_size')
        self.hidden_sizes = [256, 128]
        self.makeOutputSize()
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[1], self.output_size),
            nn.ReLU(),
        )
    
    def makeNumLaneRoadNetMap(self):
        num_lanes_road_net_map = {}

        for road_order_id in self.controller.roads.getKeys(container_flg=True):
            lanes = self.controller.road_lanes_map[road_order_id]

            if lanes.count() not in num_lanes_road_net_map:
                num_lanes_road_net_map[lanes.count()] = ApeXRoadNet(self, lanes.count())
        
        self.num_lanes_road_net_map = num_lanes_road_net_map
    
    def makeOutputSize(self):
        # フェーズの数を取得
        num_roads_phases_map = self.config.get('num_roads_phases_map')  
        phases_map = num_roads_phases_map[self.controller.roads.count()]
        num_phases = phases_map.shape[0]

        # 出力サイズを設定
        self.output_size = num_phases

    def forward(self, x):
        # 自動車の情報について
        vehicle_states = []
        for states in x:
            roads_states = states['roads']
            for road_order_id, road_states in roads_states.items():
                lanes_states = road_states['lanes']
                for lane_order_id, lane_states in lanes_states.items():
                    vehicle_states.extend([tmp_states for _, tmp_states in lane_states['vehicles'].items()])

        # 車両の情報をテンソルに変換
        vehicle_states = torch.stack(vehicle_states)

        # 車両の情報をネットワークに通す
        vehicle_outputs = self.vehicle_net(vehicle_states)

        # 次元を変更
        # (batch_size * num_roads * num_lanes, num_vehicles_features)
        vehicle_outputs = vehicle_outputs.view(-1, self.vehicles_net.num_features)
        vehicle_outputs = self.vehicles_net(vehicle_outputs)

        # 車線形状について
        lane_shape_states = []

        for states in x:
            roads_states = states['roads']
            for road_order_id, road_states in roads_states.items():
                lanes_states = road_states['lanes']
                for lane_order_id, lane_states in lanes_states.items():
                    lane_shape_states.append(lane_states['shape'])
        
        lane_shape_states = torch.stack(lane_shape_states)    
        lane_shape_outputs = self.lane_shape_net(lane_shape_states)

        # 車線ごとの評価指標について
        lane_metric_states = []

        for states in x:
            roads_states = states['roads']
            for road_order_id, road_states in roads_states.items():
                lanes_states = road_states['lanes']
                for lane_order_id, lane_states in lanes_states.items():
                    lane_metric_states.append(lane_states['metric'])
        
        lane_metric_states = torch.stack(lane_metric_states)
        lane_metric_outputs = self.lane_metric_net(lane_metric_states)

        # 3つの特徴量を結合，配列の次元は（batch_size * num_roads * num_lanes, num_features）
        lane_states = torch.cat((vehicle_outputs, lane_shape_outputs, lane_metric_outputs), dim=1)

        # 車線の情報をネットワークに通す
        lane_outputs = self.lane_net(lane_states)

        # 道路ごとに車線の情報を分割する
        num_batch = len(x)
        lane_outputs_map = self.makeLaneOutputsMap(lane_outputs, num_batch)
        
        # 道路の評価指標について
        road_metric_states = []
        for states in x:
            roads_states = states['roads']
            for road_order_id, road_states in roads_states.items():
                road_metric_states.append(road_states['metric'])
        
        road_metric_states = torch.stack(road_metric_states)
        road_metric_outputs = self.road_metric_net(road_metric_states)

        road_metric_outputs_map = self.makeRoadMetricOutputsMap(road_metric_outputs, num_batch)

        # 道路ごとのネットワークに通す
        road_outputs = None
        for road_order_id in self.controller.roads.getKeys(container_flg=True, sorted_flg=True):
            # lanesオブジェクトを取得
            lanes = self.controller.road_lanes_map[road_order_id]

            # 車線の数が一致するroad_netを取得
            road_net = self.num_lanes_road_net_map[lanes.count()]

            # 道路情報をネットワークに通す
            tmp_road_outputs = road_net(torch.cat((lane_outputs_map[road_order_id], road_metric_outputs_map[road_order_id]), dim=1))
            
            if road_outputs is None:
                road_outputs = tmp_road_outputs
            else:
                road_outputs = torch.cat((road_outputs, tmp_road_outputs), dim=1)
        
        # フェーズの情報をネットワークに通す
        phase_states = []
        for states in x:
            phase_states.append(states['intersection'])
        
        phase_states = torch.stack(phase_states)
        phase_outputs = self.phase_net(phase_states)

        # 交差点の情報をネットワークに通す
        intersection_states = torch.cat((road_outputs, phase_outputs), dim=1)
        intersection_outputs = self.intersection_net(intersection_states)

        # Q値の確率を計算
        q_values = self.net(intersection_outputs)

        return q_values

    def makeLaneOutputsMap(self, lane_outputs, num_batch):
        lane_outputs_map = {}

        lane_outputs = lane_outputs.view(num_batch, -1)

        lane_output_size = self.lane_net.get('output_size')

        end_col = -1
        for road_order_id in self.controller.roads.getKeys(container_flg=True, sorted_flg=True):
            lanes = self.controller.road_lanes_map[road_order_id]
            num_lanes = lanes.count()

            start_col = end_col + 1
            end_col = start_col + num_lanes * lane_output_size - 1

            lane_outputs_map[road_order_id] = lane_outputs[:, start_col:end_col + 1]
        
        return lane_outputs_map

    def makeRoadMetricOutputsMap(self, road_metric_outputs, num_batch):
        road_metric_outputs_map = {}
        road_metric_outputs = road_metric_outputs.view(num_batch, -1)

        road_metric_output_size = self.road_metric_net.get('output_size')

        end_col = -1
        for road_order_id in self.controller.roads.getKeys(container_flg=True, sorted_flg=True):
            start_col = end_col + 1
            end_col = start_col + road_metric_output_size - 1
            road_metric_outputs_map[road_order_id] = road_metric_outputs[:, start_col:end_col + 1]
        
        return road_metric_outputs_map

class ApeXVehicleNet(NeuralNetwork):
    def __init__(self, apex_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = apex_net.config
        self.executor = apex_net.executor
        self.apex_net = apex_net

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
        )

    def makeNumFeatures(self):
        # 強化学習に関する設定を取得
        drl_info = self.config.get('drl_info')

        # 使用する状態量を確認しカウント
        num_features = 0
        for feature_name, feature_flg in drl_info['features']['vehicle'].items():
            if feature_flg == False:
                continue
            if feature_name == 'direction':
                intersection = self.apex_net.controller.intersection
                num_features += (intersection.get('num_roads') - 1)
            else:
                num_features += 1

        # 存在するかどうかのフラグ分を追加
        num_features += 1

        # インスタンス変数として保存
        self.num_features = num_features
    
    def forward(self, x):
        # xは（batch_size × num_vehicles × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)
        
class ApeXVehiclesNet(NeuralNetwork):
    def __init__(self, apex_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = apex_net.config
        self.executor = apex_net.executor
        self.apex_net = apex_net

        # vehicle_netを取得
        self.vehicle_net = apex_net.vehicle_net

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
        )
    
    def makeNumFeatures(self):
        # 強化学習に関する設定を取得
        drl_info = self.config.get('drl_info')
        
        # vehicle_netの出力サイズに自動車台数をかけたものが特徴量の数
        self.num_features = self.vehicle_net.output_size * drl_info['num_vehicles']
    
    def forward(self, x):
        # xは（batch_size × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)

class ApeXLaneShapeNet(NeuralNetwork):
    def __init__(self, apex_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = apex_net.config
        self.executor = apex_net.executor
        self.apex_net = apex_net

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.output_size = 8
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            nn.ReLU(),
        )   

    def makeNumFeatures(self):
        # 強化学習に関する設定を取得
        drl_info = self.config.get('drl_info')

        # 状態の数を初期化
        self.num_features = 0

        # 使用する状態量を確認してカウント
        for feature_name, feature_flg in drl_info['features']['lane']['shape'].items():
            if feature_flg == False:
                continue

            if feature_name == 'length':
                self.num_features += 1
            elif feature_name == 'type':
                self.num_features += 2

    def forward(self, x):
        # xは（batch_size × num_roads × num_lanes, num_features）のテンソル
        return self.net(x)

class ApeXLaneMetricNet(NeuralNetwork):
    def __init__(self, apex_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = apex_net.config
        self.executor = apex_net.executor
        self.apex_net = apex_net

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.output_size = 8
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            nn.ReLU(),
        )     

    def makeNumFeatures(self):
        # 強化学習に関する設定を取得
        drl_info = self.config.get('drl_info')

        # 状態の数を初期化
        self.num_features = 0

        # 使用する状態量を確認してカウント  
        for feature_name, feature_flg in drl_info['features']['lane']['metric'].items():
            if feature_flg == False:
                continue

            if feature_name == 'num_vehicles':
                self.num_features += 1    

    def forward(self, x):
        # xは（batch_size × num_roads × num_lanes, num_features）のテンソル
        return self.net(x)
    
class ApeXLaneNet(NeuralNetwork):
    def __init__(self, apex_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = apex_net.config
        self.executor = apex_net.executor
        self.apex_net = apex_net

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
        )
    
    def makeNumFeatures(self):
        num_vehicle_features = self.apex_net.vehicles_net.get('output_size')
        num_shape_features = self.apex_net.lane_shape_net.get('output_size')
        num_metric_features = self.apex_net.lane_metric_net.get('output_size')

        self.num_features = num_vehicle_features + num_shape_features + num_metric_features
    
    def forward(self, x):
        return self.net(x)
    
class ApeXRoadMetricNet(NeuralNetwork):
    def __init__(self, apex_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトと上位の紐づくオブジェクトを取得
        self.config = apex_net.config
        self.executor = apex_net.executor
        self.apex_net = apex_net

        # 特徴量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.output_size = 8
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            nn.ReLU(),
        )

    def makeNumFeatures(self):
        # 強化学習に関する設定を取得
        drl_info = self.config.get('drl_info')

        # 状態の数を初期化
        num_features = 0

        # 使用する状態量を確認してカウント
        for feature_name, feature_flg in drl_info['features']['road']['metric'].items():
            if feature_flg == False:
                continue
            
            if feature_name == 'queue_length':
                num_features += 1
            elif feature_name == 'delay':
                num_features += 1
            
        # インスタンス変数として保存
        self.num_features = num_features
    
    def forward(self, x):
        return self.net(x)
    
class ApeXRoadNet(NeuralNetwork):
    def __init__(self, apex_net, num_lanes):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトと上位の紐づくオブジェクトを取得
        self.config = apex_net.config
        self.executor = apex_net.executor
        self.apex_net = apex_net

        # 車線の数を取得
        self.num_lanes = num_lanes

        # 特徴量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
        )

    def makeNumFeatures(self):
        # 車線の状態量の数を取得
        num_lane_features = self.apex_net.lane_net.get('output_size')

        # 道路の評価指標の状態量の数を取得
        num_road_metric_features = self.apex_net.road_metric_net.get('output_size')

        # 道路の状態量の数を取得
        self.num_features = num_lane_features * self.num_lanes + num_road_metric_features
    
    def forward(self, x):
        return self.net(x)
class ApeXPhaseNet(NeuralNetwork):
    def __init__(self, apex_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトと上位の紐づくオブジェクトを取得
        self.config = apex_net.config
        self.executor = apex_net.executor
        self.apex_net = apex_net

        # 特徴量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
        )

    def makeNumFeatures(self):
        # フェーズの情報を取得
        num_roads = self.apex_net.controller.roads.count()

        # フェーズ情報の設定を取得
        phases_map = self.config.get('num_roads_phases_map')[num_roads]

        # フェーズの数を取得
        num_phases = phases_map.shape[0]

        # 状態量の数を設定
        self.num_features = num_phases
    
    def forward(self, x):
        return self.net(x)

class ApeXIntersectionNet(NeuralNetwork):
    def __init__(self, apex_net, num_roads):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトと上位の紐づくオブジェクトを取得
        self.config = apex_net.config
        self.executor = apex_net.executor
        self.apex_net = apex_net

        # 道路の数を取得
        self.num_roads = num_roads

        # 特徴量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
        )

    def makeNumFeatures(self):
        # 状態量の数を初期化
        num_features = 0

        # 道路の状態量を取得
        for road_order_id in self.apex_net.controller.roads.getKeys(container_flg=True):
            lanes = self.apex_net.controller.road_lanes_map[road_order_id]
            road_net = self.apex_net.num_lanes_road_net_map[lanes.count()]
            num_features += road_net.get('output_size')
        
        # フェーズの状態量を取得
        num_features += self.apex_net.phase_net.get('output_size')

        self.num_features = num_features
    
    def forward(self, x):
        return self.net(x)


