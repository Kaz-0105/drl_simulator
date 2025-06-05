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

        # requires_grad_flgを設定
        self.requires_grad = True

        # 各サブネットワークを定義
        self._makeSubNetworkMap()
        
        # ネットワークの定義（Dueling Network）        
        self.input_size = self.sub_network_map['intersection'].get('output_size')
        self.hidden_sizes = [128, 64]
        self._makeOutputSize()
        self.value_stream = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_sizes[1], 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_sizes[1], self.output_size),
        )

        # ネットワークのパラメータの初期化を行う
        self._initialize_networks()

    
    def _makeSubNetworkMap(self):
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
        self._makeLanesNetMap()

        # 道路の評価指標のサブネットワークを定義
        self.sub_network_map['road_metric'] = RoadMetricNet(self.config)

        # 道路のサブネットワークを定義
        self._makeRoadNetMap()

        # 道路群のサブネットワークを定義
        self._makeRoadsNet()

        # フェーズのサブネットワークを定義
        self.sub_network_map['phase'] = PhaseNet(self.config, self.num_roads)

        # 交差点のサブネットワークを定義
        num_roads_features = self.sub_network_map['roads'].get('output_size')
        num_phase_features = self.sub_network_map['phase'].get('output_size')
        self.sub_network_map['intersection'] = IntersectionNet(self.config, num_roads_features, num_phase_features)
    
    def _makeLanesNetMap(self):
        lanes_net_map = nn.ModuleDict()
        
        for num_lanes in self.num_lanes_map.values():
            if num_lanes not in lanes_net_map:
                num_lane_features = self.sub_network_map['lane'].get('output_size') 
                lanes_net_map[str(num_lanes)] = LanesNet(self.config, num_lane_features, num_lanes)
        
        self.sub_network_map['lanes'] = lanes_net_map
        
    def _makeRoadNetMap(self):
        road_net_map = nn.ModuleDict()

        for num_lanes, lanes_net in self.sub_network_map['lanes'].items():
            num_lanes_features = lanes_net.get('output_size')
            num_road_metric_features = self.sub_network_map['road_metric'].get('output_size')
            road_net_map[str(num_lanes)] = RoadNet(self.config, num_lanes_features, num_road_metric_features)
        
        self.sub_network_map['road'] = road_net_map
    
    def _makeRoadsNet(self):
        # 道路ごとの特徴量の数を取得
        num_road_features_map = {}
        for road_order_id, num_lanes in self.num_lanes_map.items():
            road_net = self.sub_network_map['road'][str(num_lanes)]
            num_road_features_map[road_order_id] = road_net.get('output_size')
        
        self.sub_network_map['roads'] = RoadsNet(self.config, num_road_features_map)

    def _makeOutputSize(self):
        # フェーズの数を取得
        num_roads_phases_map = self.config.get('num_roads_phases_map')  
        phases_map = num_roads_phases_map[self.num_roads]
        num_phases = phases_map.shape[0]

        # 出力サイズを設定
        self.output_size = num_phases

    def _initialize_networks(self):
        pass

    def forward(self, x):
        #　バッチのサイズを取得
        batch_size = len(x)

        # vehicleのサブネットワークを取得
        vehicle_net = self.sub_network_map['vehicle']

        # 自動車の状態量をテンソルで取得（batch_size * num_roads * num_lanes * num_vehicles, num_vehicles_features）
        vehicle_inputs = self._extractVehicleInputs(x)

        # 出力を計算
        vehicle_outputs = vehicle_net(vehicle_inputs)

        # vehiclesのサブネットワークを取得
        vehicles_net = self.sub_network_map['vehicles']

        # 自動車群の状態量をテンソルで取得（batch_size * num_roads * num_lanes, num_vehicles_features）
        vehicles_inputs = self._reshapeVehiclesInputs(vehicle_outputs, vehicles_net.get('num_features'))

        # 出力を計算
        vehicles_outputs = vehicles_net(vehicles_inputs)

        # lane_shapeのサブネットワークを取得
        lane_shape_net = self.sub_network_map['lane_shape']

        # 車線形状の状態量をテンソルで取得（batch_size * num_roads * num_lanes, num_lane_shape_features）
        lane_shape_inputs = self._extractLaneShapeInputs(x)

        # 出力を計算
        lane_shape_outputs = lane_shape_net(lane_shape_inputs)

        # lane_metricのサブネットワークを取得
        lane_metric_net = self.sub_network_map['lane_metric']

        # 車線評価指標の状態量をテンソルで取得（batch_size * num_roads * num_lanes, num_lane_metric_features）
        lane_metric_inputs = self._extractLaneMetricInputs(x)

        # 出力を計算
        lane_metric_outputs = lane_metric_net(lane_metric_inputs)

        # laneのサブネットワークを取得
        lane_net = self.sub_network_map['lane']

        # 車線に関する入力を統合する（batch_size * num_roads * num_lanes, num_lane_features）
        lane_inputs = torch.cat((vehicles_outputs, lane_shape_outputs, lane_metric_outputs), dim=1)

        # 出力を計算
        lane_outputs = lane_net(lane_inputs)

        # lanesのサブネットワークを取得
        lanes_net_map = self.sub_network_map['lanes']

        # 車線群ごとの特徴量に分割
        # 例：{road_order_id: (batch_size, num_lanes_features)}
        road_lanes_inputs_map = self._makeRoadLanesInputsMap(lane_outputs, batch_size, lane_net.get('output_size'))
        
        # 出力を計算
        road_lanes_outputs_map = {}
        for road_order_id, lanes_inputs in road_lanes_inputs_map.items():
            # 車線数に応じたネットワークを取得
            num_lanes = self.num_lanes_map[road_order_id]
            lanes_net = lanes_net_map[str(num_lanes)]

            # 出力を計算して辞書に格納
            road_lanes_outputs_map[road_order_id] = lanes_net(lanes_inputs)
        
        # road_metricのサブネットワークを取得
        road_metric_net = self.sub_network_map['road_metric']

        # 道路の評価指標の状態量をテンソルで取得（batch_size * num_roads, num_road_metric_features）
        road_metric_inputs = self._extractRoadMetricInputs(x)

        # 出力を計算
        road_metric_outputs = road_metric_net(road_metric_inputs)

        # 道路ごとに評価指標の特徴量を整理
        # 例：｛road_order_id =>（batch_size, num_road_metric_features）｝
        road_metric_outputs_map = self._makeRoadMetricOutputsMap(road_metric_outputs, batch_size, road_metric_net.get('output_size'))
        
        # roadのサブネットワークのマップを取得
        road_net_map = self.sub_network_map['road']

        # 道路の特徴量を結合する
        # 例：｛road_order_id => (batch_size, num_road_features)｝
        road_inputs_map = self._makeRoadInputsMap(road_lanes_outputs_map, road_metric_outputs_map)

        # 出力を計算
        road_outputs_map = {}
        for road_order_id, road_inputs in road_inputs_map.items():
            road_net = road_net_map[str(self.num_lanes_map[road_order_id])]
            road_outputs_map[road_order_id] = road_net(road_inputs)
        
        # roadsのサブネットワークを取得
        roads_net = self.sub_network_map['roads']

        # 道路の特徴量を統合する
        roads_inputs = self._makeRoadsInputs(road_outputs_map)
        
        # 出力を計算
        roads_outputs = roads_net(roads_inputs)

        # phaseのサブネットワークを取得
        phase_net = self.sub_network_map['phase']

        # フェーズの状態量をテンソルに変換（batch_size, num_phases）
        phase_inputs = self._extractPhaseInputs(x)

        # 出力を計算
        phase_outputs = phase_net(phase_inputs)

        # intersectionのサブネットワークを取得
        intersection_net = self.sub_network_map['intersection']

        # 交差点の状態量を結合する（batch_size, num_intersection_features）
        intersection_inputs = torch.cat((roads_outputs, phase_outputs), dim=1)

        # 出力を計算
        intersection_outputs = intersection_net(intersection_inputs)

        # Dueling Networkの出力を計算
        state_value = self.value_stream(intersection_outputs)
        advantages = self.advantage_stream(intersection_outputs)

        # Q値を計算
        q_values = state_value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

    def _extractVehicleInputs(self, x):
        vehicle_inputs = []
        for states in x:
            roads = states['roads']
            for road in roads.values():
                lanes = road['lanes']
                for lane in lanes.values():
                    vehicles = lane['vehicles']
                    vehicle_inputs.extend([vehicle for _, vehicle in vehicles.items()])
        
        vehicle_inputs = torch.stack(vehicle_inputs)
        vehicle_inputs.requires_grad_(self.requires_grad)

        return vehicle_inputs

    def _reshapeVehiclesInputs(self, vehicle_outputs, num_vehicles_features):
        return vehicle_outputs.view(-1, num_vehicles_features)

    def _extractLaneShapeInputs(self, x):
        lane_shape_inputs = []
        for states in x:
            roads = states['roads']
            for road in roads.values():
                lanes = road['lanes']
                lane_shape_inputs.extend([lane['shape'] for lane in lanes.values()])
        
        lane_shape_inputs = torch.stack(lane_shape_inputs)
        lane_shape_inputs.requires_grad_(self.requires_grad)  

        return lane_shape_inputs

    def _extractLaneMetricInputs(self, x):
        lane_metric_inputs = []
        for states in x:
            roads = states['roads']
            for road in roads.values():
                lanes = road['lanes']
                lane_metric_inputs.extend([lane['metric'] for lane in lanes.values()])
        
        lane_metric_inputs = torch.stack(lane_metric_inputs)
        lane_metric_inputs.requires_grad_(self.requires_grad)

        return lane_metric_inputs

    def _makeRoadLanesInputsMap(self, lane_outputs, batch_size, lane_output_size):
        # キーが道路の順序ID，値が道路を構成する車線群の特徴量のテンソルとなる辞書を初期化
        # 例：{road_order_id: (batch_size, num_lanes_features)}
        road_lanes_inputs_map = {}

        # バッチごとに車線の特徴量を一行に並べる
        lane_outputs = lane_outputs.view(batch_size, -1) 

        # 道路の車線数を参照して，区切っていく
        end_col = -1
        for road_order_id, num_lanes in self.num_lanes_map.items():
            start_col = end_col + 1
            end_col += num_lanes * lane_output_size

            road_lanes_inputs_map[road_order_id] = lane_outputs[:, start_col:end_col + 1]
        
        return road_lanes_inputs_map

    def _extractRoadMetricInputs(self, x):
        road_metric_inputs = []
        for states in x:
            roads = states['roads']
            road_metric_inputs.extend([road['metric'] for road in roads.values()])
            
        road_metric_inputs = torch.stack(road_metric_inputs)
        road_metric_inputs.requires_grad_(self.requires_grad)

        return road_metric_inputs
    
    def _makeRoadMetricOutputsMap(self, road_metric_outputs, batch_size, output_size):
        # 道路の順序IDをキー，評価指標の特徴量を値とする辞書を初期化
        # 例：{road_order_id: (batch_size, num_road_metric_features)}
        road_metric_outputs_map = {}

        # バッチごとに評価指標の特徴量を一行に並べる
        road_metric_outputs = road_metric_outputs.view(batch_size, -1)
        
        # 道路ごとに区切っていく
        end_col = -1
        for road_order_id in self.num_lanes_map.keys():
            start_col = end_col + 1
            end_col += output_size

            road_metric_outputs_map[road_order_id] = road_metric_outputs[:, start_col:end_col + 1]
        
        return road_metric_outputs_map

    def _makeRoadInputsMap(self, road_lanes_outputs_map, road_metric_outputs_map):
        # 道路の順序IDをキー，車線の特徴量と道路の評価指標の特徴量を結合したものを値とする辞書を初期化
        # 例：{road_order_id: (batch_size, num_road_features)}
        road_inputs_map = {}

        # 道路を走査
        for road_order_id in self.num_lanes_map.keys():
            # 車線の特徴量を取得
            lane_outputs = road_lanes_outputs_map[road_order_id]

            # 道路の評価指標の特徴量を取得
            road_metric_outputs = road_metric_outputs_map[road_order_id]

            # 車線と道路の評価指標の特徴量を結合
            road_inputs_map[road_order_id] = torch.cat((lane_outputs, road_metric_outputs), dim=1)
        
        return road_inputs_map
    
    def _makeRoadsInputs(self, road_outputs_map):
        roads_inputs = None
        for road_outputs in road_outputs_map.values():
            if roads_inputs is None:
                roads_inputs = road_outputs
            else:
                roads_inputs = torch.cat((roads_inputs, road_outputs), dim=1)
        
        return roads_inputs
    
    def _extractPhaseInputs(self, x):
        phase_inputs = []
        for states in x:
            phase_inputs.append(states['phase'])

        phase_inputs = torch.stack(phase_inputs)
        phase_inputs.requires_grad_(self.requires_grad)

        return phase_inputs

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
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
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
        self.hidden_size = self.num_features
        self.output_size = self.num_features
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_size, self.output_size),
        )
    
    def forward(self, x):
        return self.net(x)


