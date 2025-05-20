from controllers.base_controller import BaseController
from objects.links import Lanes
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np

class DRLController(BaseController):
    def __init__(self, controllers, intersection):
        # 継承
        super().__init__(controllers)

        # intersectionオブジェクトと紐づける
        self.intersection = intersection
        intersection.set('controller', self)

        # drl_infoを取得
        self.drl_info = self.config.get('drl_info')

        # モデルの定義
        if self.config.get('drl_info')['method'] =='a2c':
            self.model = A2CNet(self)
            model_path = Path('models/a2c.pth')

            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path))

        self.roads = self.intersection.input_roads
        self.makeRoadLanesMap()
    
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
        self.action = self.model([self.states])

    def makeStates(self):
        if self.config.get('drl_info')['method'] == 'a2c':
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

class A2CNet(nn.Module):
    def __init__(self, controller):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = controller.config
        self.executor = controller.executor
        self.controller = controller

        self.vehicle_net = A2CVehicleNet(self)
        self.vehicles_net = A2CVehiclesNet(self)
        self.lane_shape_net = A2CLaneShapeNet(self)
        self.lane_metric_net = A2CLaneMetricNet(self)
        self.lane_net = A2CLaneNet(self)
        # self.road_net = A2CRoadNet(self)

        
    def get(self, property_name):
        if hasattr(self, property_name) == False:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{property_name}'")
        
        return getattr(self, property_name)
        
    def set(self, property_name, value):
        if hasattr(self, property_name) == False:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{property_name}'")
        
        setattr(self, property_name, value)

    
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

        print('test')

class A2CVehicleNet(nn.Module):
    def __init__(self, a2c_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = a2c_net.config
        self.executor = a2c_net.executor
        self.a2c_net = a2c_net

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = 32
        self.output_size = 32
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
                intersection = self.a2c_net.controller.intersection
                num_features += (intersection.get('num_roads') - 1)
            else:
                num_features += 1

        # 存在するかどうかのフラグ分を追加
        num_features += 1

        # インスタンス変数として保存
        self.num_features = num_features
    
    def get(self, property_name):
        if hasattr(self, property_name) == False:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{property_name}'")
        
        return getattr(self, property_name)
        
    def set(self, property_name, value):
        if hasattr(self, property_name) == False:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{property_name}'")
        
        setattr(self, property_name, value)
    
    def forward(self, x):
        # xは（batch_size × num_vehicles × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)
        
class A2CVehiclesNet(nn.Module):
    def __init__(self, a2c_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = a2c_net.config
        self.executor = a2c_net.executor
        self.a2c_net = a2c_net

        # vehicle_netを取得
        self.vehicle_net = a2c_net.vehicle_net

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = 64
        self.output_size = 64
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
    
    def get(self, property_name):
        if hasattr(self, property_name) == False:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{property_name}'")
        
        return getattr(self, property_name)
        
    def set(self, property_name, value):
        if hasattr(self, property_name) == False:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{property_name}'")
        
        setattr(self, property_name, value)
    
    def forward(self, x):
        # xは（batch_size × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)

class A2CLaneShapeNet(nn.Module):
    def __init__(self, a2c_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = a2c_net.config
        self.executor = a2c_net.executor
        self.a2c_net = a2c_net

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
    
    def get(self, property_name):
        if hasattr(self, property_name) == False:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{property_name}'")
        
        return getattr(self, property_name)
    
    def set(self, property_name, value):
        if hasattr(self, property_name) == False:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{property_name}'")
        
        setattr(self, property_name, value)

    def forward(self, x):
        # xは（batch_size × num_roads × num_lanes, num_features）のテンソル
        return self.net(x)

class A2CLaneMetricNet(nn.Module):
    def __init__(self, a2c_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = a2c_net.config
        self.executor = a2c_net.executor
        self.a2c_net = a2c_net

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
    
class A2CLaneNet(nn.Module):
    def __init__(self, a2c_net):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = a2c_net.config
        self.executor = a2c_net.executor
        self.a2c_net = a2c_net

        # 状態量の数を取得する
        self.makeNumFeatures()

        # ネットワークの定義
        self.input_size = self.num_features
        self.hidden_size = 128
        self.output_size = 128
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
        )
    
    def makeNumFeatures(self):
        num_vehicle_features = self.a2c_net.vehicles_net.output_size
        num_shape_features = self.a2c_net.lane_shape_net.output_size
        num_metric_features = self.a2c_net.lane_metric_net.output_size

        self.num_features = num_vehicle_features + num_shape_features + num_metric_features
    
    def forward(self, x):
        return self.net(x)

        


