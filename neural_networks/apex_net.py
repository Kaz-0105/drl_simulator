from libs.neural_network import NeuralNetwork

import torch
import torch.nn as nn

class QNet(NeuralNetwork):
    def __init__(self, config, num_vehicles, num_lanes_map):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # ネットワーク関連のハイパーパラメータを取得
        self.num_vehicles = num_vehicles
        self.num_lanes_map = num_lanes_map
        self.num_roads = len(self.num_lanes_map)

        # 各サブネットワークを定義
        self.makeSubNetworkMap()
        
        # ネットワークの定義（Dueling Network）        
        self.input_size = self.sub_network_map['intersection'].get('output_size')
        self.hidden_sizes = [self.input_size // 2, self.input_size // 2]
        self.makeOutputSize()
        self.value_stream = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[1], 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[1], self.output_size),
        )

    
    def makeSubNetworkMap(self):
        # サブネットワークのマップを初期化
        self.sub_network_map = nn.ModuleDict()

        # 自動車のサブネットワークを定義
        self.sub_network_map['vehicle'] = VehicleNet(self.config, self.num_roads - 1)

        # 自動車群のサブネットワークを定義
        vehicle_features = self.sub_network_map['vehicle'].get('output_size')
        self.sub_network_map['vehicles'] = VehiclesNet(self.config, vehicle_features, self.num_vehicles)

        # 車線の形状と評価指標のサブネットワークを定義
        self.sub_network_map['lane_shape'] = LaneShapeNet(self.config)
        self.sub_network_map['lane_metric'] = LaneMetricNet(self.config)

        # 車線のサブネットワークを定義
        vehicles_features = self.sub_network_map['vehicles'].get('output_size')
        lane_shape_features = self.sub_network_map['lane_shape'].get('output_size')
        lane_metric_features = self.sub_network_map['lane_metric'].get('output_size')
        self.sub_network_map['lane'] = LaneNet(self.config, vehicles_features, lane_shape_features, lane_metric_features)
        
        # 車線群のサブネットワークを定義
        self.makeLanesNetMap()

        # 道路の評価指標のサブネットワークを定義
        self.sub_network_map['road_metric'] = RoadMetricNet(self.config)

        # 道路のサブネットワークを定義
        self.makeRoadNetMap()

        # 道路群のサブネットワークを定義
        self.makeRoadsNet()

        # フェーズのサブネットワークを定義
        self.sub_network_map['phase'] = PhaseNet(self.config, self.num_roads)

        # 交差点のサブネットワークを定義
        num_roads_features = self.sub_network_map['roads'].get('output_size')
        num_phase_features = self.sub_network_map['phase'].get('output_size')
        self.sub_network_map['intersection'] = IntersectionNet(self.config, num_roads_features, num_phase_features)
    
    def makeLanesNetMap(self):
        lanes_net_map = nn.ModuleDict()
        
        for num_lanes in self.num_lanes_map.values():
            if num_lanes not in lanes_net_map:
                num_lane_features = self.sub_network_map['lane'].get('output_size')
                lanes_net_map[str(num_lanes)] = LanesNet(self.config, num_lane_features, num_lanes)
        
        self.sub_network_map['lanes'] = lanes_net_map
        
    def makeRoadNetMap(self):
        road_net_map = nn.ModuleDict()

        for num_lanes, lanes_net in self.sub_network_map['lanes'].items():
            num_lanes_features = lanes_net.get('output_size')
            num_road_metric_features = self.sub_network_map['road_metric'].get('output_size')
            road_net_map[str(num_lanes)] = RoadNet(self.config, num_lanes_features, num_road_metric_features)
        
        self.sub_network_map['road'] = road_net_map
    
    def makeRoadsNet(self):
        # 道路ごとの特徴量の数を取得
        num_road_features_map = {}
        for road_order_id, num_lanes in self.num_lanes_map.items():
            road_net = self.sub_network_map['road'][str(num_lanes)]
            num_road_features_map[road_order_id] = road_net.get('output_size')
        
        self.sub_network_map['roads'] = RoadsNet(self.config, num_road_features_map)

    def makeOutputSize(self):
        # フェーズの数を取得
        num_roads_phases_map = self.config.get('num_roads_phases_map')  
        phases_map = num_roads_phases_map[self.num_roads]
        num_phases = phases_map.shape[0]

        # 出力サイズを設定
        self.output_size = num_phases

    def forward(self, x):
        #　バッチのサイズを取得
        batch_size = len(x)

        # 自動車の状態量をテンソルに変換（batch_size * num_roads * num_lanes * num_vehs, num_vehicle_features）
        vehicle_inputs = []
        for states in x:
            roads_state = states['roads']
            for road_state in roads_state.values():
                lanes_state = road_state['lanes']
                for lane_state in lanes_state.values():
                    vehicle_inputs.extend([tmp_state for _, tmp_state in lane_state['vehicles'].items()])
        vehicle_inputs = torch.stack(vehicle_inputs)

        # vehicle_netに通す
        vehicle_net = self.sub_network_map['vehicle']
        vehicle_outputs = vehicle_net(vehicle_inputs)

        # テンソルの次元を変更 (batch_size * num_roads * num_lanes, num_vehicles_features)
        vehicles_net = self.sub_network_map['vehicles']
        vehicles_inputs = vehicle_outputs.view(-1, vehicles_net.get('num_features'))

        # vehicles_netに通す
        vehicles_outputs = vehicles_net(vehicles_inputs)

        # 車線形状の状態量をテンソルに変換（batch_size * num_roads * num_lanes, num_lane_shape_features）
        lane_shape_inputs = []
        for states in x:
            roads_state = states['roads']
            for road_state in roads_state.values():
                lanes_state = road_state['lanes']
                for lane_state in lanes_state.values():
                    lane_shape_inputs.append(lane_state['shape'])
        lane_shape_inputs = torch.stack(lane_shape_inputs)    

        # lane_shape_netに通す
        lane_shape_net = self.sub_network_map['lane_shape']
        lane_shape_outputs = lane_shape_net(lane_shape_inputs)

        # 車線評価指標の状態量をテンソルに変換（batch_size * num_roads * num_lanes, num_lane_metric_features）
        lane_metric_inputs = []
        for states in x:
            roads_state = states['roads']
            for road_state in roads_state.values():
                lanes_state = road_state['lanes']
                for lane_state in lanes_state.values():
                    lane_metric_inputs.append(lane_state['metric'])
        lane_metric_inputs = torch.stack(lane_metric_inputs)

        # lane_metric_netに通す
        lane_metric_net = self.sub_network_map['lane_metric']
        lane_metric_outputs = lane_metric_net(lane_metric_inputs)

        # 車線の特徴量を結合する（batch_size * num_roads * num_lanes, num_lane_features）
        lane_inputs = torch.cat((vehicles_outputs, lane_shape_outputs, lane_metric_outputs), dim=1)

        # lane_netに通す
        lane_net = self.sub_network_map['lane']
        lane_outputs = lane_net(lane_inputs)

        # 車線群ごとの特徴量に分割｛road_order_id =>（batch_size, num_lanes_features）｝
        lanes_inputs_map = {}
        lane_outputs = lane_outputs.view(batch_size, -1) 
        end_col = -1
        for road_order_id, num_lanes in self.num_lanes_map.items():
            start_col = end_col + 1
            end_col += num_lanes * lane_net.get('output_size')

            lanes_inputs_map[road_order_id] = lane_outputs[:, start_col:end_col + 1]
        
        # lanes_netに通す
        lanes_outputs_map = {}
        for road_order_id, lanes_inputs in lanes_inputs_map.items():
            lanes_net = self.sub_network_map['lanes'][str(self.num_lanes_map[road_order_id])]
            lanes_outputs_map[road_order_id] = lanes_net(lanes_inputs)
        
        # 道路の評価指標の状態量をテンソルに変換（batch_size * num_roads, num_road_metric_features）
        road_metric_inputs = []
        for states in x:
            roads_state = states['roads']
            for road_state in roads_state.values():
                road_metric_inputs.append(road_state['metric'])
        road_metric_inputs = torch.stack(road_metric_inputs)

        # road_metric_netに通す
        road_metric_net = self.sub_network_map['road_metric']
        road_metric_outputs = road_metric_net(road_metric_inputs)

        # 道路ごとに評価指標の特徴量を整理｛road_order_id =>（batch_size, num_road_metric_features）｝
        road_metric_outputs_map = {}
        road_metric_outputs = road_metric_outputs.view(batch_size, -1)
        end_col = -1
        for road_order_id in self.num_lanes_map.keys():
            start_col = end_col + 1
            end_col += road_metric_net.get('output_size')
            road_metric_outputs_map[road_order_id] = road_metric_outputs[:, start_col:end_col + 1]
        
        # 道路の特徴量を結合する｛road_order_id => (batch_size, num_road_features)｝
        road_inputs_map = {}
        for road_order_id in self.num_lanes_map.keys():
            # 車線の特徴量を取得
            lane_outputs = lanes_outputs_map[road_order_id]

            # 道路の評価指標の特徴量を取得
            road_metric_outputs = road_metric_outputs_map[road_order_id]

            # 車線と道路の評価指標の特徴量を結合
            road_inputs_map[road_order_id] = torch.cat((lane_outputs, road_metric_outputs), dim=1)

        # road_netに通す
        road_outputs_map = {}
        for road_order_id, road_inputs in road_inputs_map.items():
            road_net = self.sub_network_map['road'][str(self.num_lanes_map[road_order_id])]
            road_outputs_map[road_order_id] = road_net(road_inputs)
        
        # 道路群の特徴量に統合する
        roads_inputs = None
        for road_outputs in road_outputs_map.values():
            if roads_inputs is None:
                roads_inputs = road_outputs
            else:
                roads_inputs = torch.cat((roads_inputs, road_outputs), dim=1)
        
        # roads_netに通す
        roads_net = self.sub_network_map['roads']
        roads_outputs = roads_net(roads_inputs)

        # フェーズの状態量をテンソルに変換（batch_size, num_phases）
        phase_inputs = []
        for states in x:
            phase_inputs.append(states['phase'])
        phase_inputs = torch.stack(phase_inputs)

        # phase_netに通す
        phase_net = self.sub_network_map['phase']
        phase_outputs = phase_net(phase_inputs)

        # 交差点の状態量を結合する（batch_size, num_intersection_features）
        intersection_inputs = torch.cat((roads_outputs, phase_outputs), dim=1)

        # intersection_netに通す
        intersection_net = self.sub_network_map['intersection']
        intersection_outputs = intersection_net(intersection_inputs)

        # Dueling Networkの出力を計算
        state_value = self.value_stream(intersection_outputs)
        advantages = self.advantage_stream(intersection_outputs)

        # Q値を計算
        q_values = state_value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

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


class VehicleNet(NeuralNetwork):
    def __init__(self, config, num_routes):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # ネットワーク関連のハイパーパラメータを取得
        self.num_routes = num_routes

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features // 2
        self.output_size = self.num_features // 2
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
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
                num_features += (self.num_routes + 1) # 方向が分からない場合もあるので、+1
            else:
                num_features += 1

        # 存在するかどうかのフラグ分を追加
        num_features += 1

        # インスタンス変数として保存
        self.num_features = num_features
    
    def forward(self, x):
        # xは（batch_size × num_vehicles × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)
        
class VehiclesNet(NeuralNetwork):
    def __init__(self, config, num_vehicle_features, num_vehicles):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # 状態量の数を取得する
        self.num_features = num_vehicle_features * num_vehicles

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features // 2
        self.output_size = self.num_features // 2
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
    
    def forward(self, x):
        # xは（batch_size × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)

class LaneShapeNet(NeuralNetwork):
    def __init__(self, config):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features 
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
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

class LaneMetricNet(NeuralNetwork):
    def __init__(self, config):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
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
    
class LaneNet(NeuralNetwork):
    def __init__(self, config, num_vehicles_features, num_shape_features, num_metric_features):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # ネットワーク関連のパラメータを取得
        self.num_vehicles_features = num_vehicles_features
        self.num_shape_features = num_shape_features
        self.num_metric_features = num_metric_features

        # 状態量の数を取得する
        self.num_features = self.num_vehicles_features + self.num_shape_features + self.num_metric_features

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
    
    def forward(self, x):
        return self.net(x)

class LanesNet(NeuralNetwork):
    def __init__(self, config, num_lane_features, num_lanes):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # ネットワーク関連のパラメータを取得
        self.num_lane_features = num_lane_features
        self.num_lanes = num_lanes

        # 状態量の数を取得する
        self.num_features = self.num_lane_features * self.num_lanes

        # ネットワークの定義
        self.input_size = self.num_features 
        self.hidden_size = self.num_features // 2
        self.output_size = self.num_features // 2
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x):
        return self.net(x)
    
class RoadMetricNet(NeuralNetwork):
    def __init__(self, config):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # 特徴量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
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
    
class RoadNet(NeuralNetwork):
    def __init__(self, config, num_lanes_features, num_road_metric_features):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # ネットワーク関連のパラメータを取得
        self.num_lanes_features = num_lanes_features
        self.num_road_metric_features = num_road_metric_features

        # 特徴量の数を取得する
        self.num_features = self.num_lanes_features + self.num_road_metric_features

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features // 2
        self.output_size = self.num_features // 2
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
    
    def forward(self, x):
        return self.net(x)

class RoadsNet(NeuralNetwork):
    def __init__(self, config, num_road_features_map):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # ネットワーク関連のパラメータを取得
        self.num_road_features_map = num_road_features_map

        # 特徴量の数を取得する
        self.num_features = sum(self.num_road_features_map.values())

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features // 2
        self.output_size = self.num_features // 2
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x):
        return self.net(x)

class PhaseNet(NeuralNetwork):
    def __init__(self, config, num_roads):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # ネットワーク関連のパラメータを取得
        self.num_roads = num_roads
        self.makeNumPhases()

        # 特徴量の数を取得する
        self.num_features = self.num_phases

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features // 2
        self.output_size = self.num_features // 2
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
    
    def makeNumPhases(self):
        # フェーズ情報の設定を取得
        phases_map = self.config.get('num_roads_phases_map')[self.num_roads]

        # フェーズの数を取得
        self.num_phases = phases_map.shape[0]
    
    def forward(self, x):
        return self.net(x)

class IntersectionNet(NeuralNetwork):
    def __init__(self, config, num_roads_features, num_phase_features):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config

        # ネットワーク関連のパラメータを取得
        self.num_roads_features = num_roads_features
        self.num_phase_features = num_phase_features

        # 特徴量の数を取得する
        self.num_features = self.num_roads_features + self.num_phase_features

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = self.num_features // 2
        self.output_size = self.num_features // 2
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
    
    def forward(self, x):
        return self.net(x)


