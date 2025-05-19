from controllers.base_controller import BaseController
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

        self.num_roads = self.intersection.get('num_roads')
        self.makeRoadNumLanesMap()

    def makeRoadNumLanesMap(self):
        road_num_lanes_map = {}
        for road_order_id in self.intersection.input_roads.getKeys(container_flg=True):
            road = self.intersection.input_roads[road_order_id]
            num_lanes = 0
            for link in road.links.getAll():
                if link.get('type') == 'connector':
                    continue

                num_lanes += link.lanes.count()
            
            road_num_lanes_map[road_order_id] = num_lanes
        
        self.road_num_lanes_map = road_num_lanes_map

    def run(self):
        # 状態量の取得
        self.makeStates()

        # 行動の選択
        self.action = self.model([self.states])

    def makeStates(self):
        if self.config.get('drl_info')['method'] == 'a2c':
            states = {'roads': {}, 'intersection': []}

            for road_order_id in self.intersection.input_roads.getKeys(container_flg=True):
                road = self.intersection.input_roads[road_order_id]
                road_states ={'lanes': [], 'metric': []}
                # 車線ごとの状態量について
                for link in road.links.getAll():
                    # connectorは除外
                    if link.get('type') == 'connector':
                        continue
                   
                    for lane in link.lanes.getAll():
                        # 車線の状態量をまとめる
                        lane_states = {}

                        # 位置でソート
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
                        vehicles_states = []
                        feature_names = ['position', 'speed', 'in_queue', 'direction']
                        for index in range(num_vehicles):
                            if index < vehicle_data.shape[0]:
                                vehicle = vehicle_data.iloc[index]
                                vehicle_states = []
                                for feature_name in feature_names:
                                    if self.drl_info['features']['vehicle'][feature_name] == True:
                                        if feature_name == 'direction':
                                            direction_vector = [0] * (self.intersection.get('num_roads') - 1)
                                            direction_vector[int(vehicle['direction_id']) - 1] = 1
                                            vehicle_states.extend(direction_vector)
                                        else: 
                                            vehicle_states.append(int(vehicle[feature_name]))
                                vehicle_states.append(1)                          
                                vehicles_states.append(torch.tensor(vehicle_states).float())
                            else:
                                vehicle_states = []
                                for feature_name in feature_names:
                                    if self.drl_info['features']['vehicle'][feature_name] == True:
                                        if feature_name == 'direction':
                                            direction_vector = [0] * (self.intersection.get('num_roads') - 1)
                                            vehicle_states.extend(direction_vector)
                                        else: 
                                            vehicle_states.append(0)
                                vehicle_states.append(0)
                                vehicles_states.append(torch.tensor(vehicle_states).float())
                        
                        # 車線の状態量に追加
                        lane_states['vehicles'] = vehicles_states

                        # 評価指標に関する状態量を取得
                        lane_states['metric'] = torch.tensor([lane.get('num_vehicles')], dtype=torch.float32)
                        
                        # 車線情報に関する状態量を取得
                        if link.get('type') == 'main':
                            lane_states['shape'] = torch.tensor([int(length_info['length']), 1, 0], dtype=torch.float32)
                        elif link.get('type') == 'right' or link.get('type') == 'left':
                            lane_states['shape'] = torch.tensor([int(length_info['length']), 0, 1], dtype=torch.float32)

                        road_states['lanes'].append(lane_states)

                # 道路ごとの評価指標の状態量について
                metric_states = []
                metric_states.append(int(road.get('max_queue_length')))
                metric_states.append(int(road.get('average_delay')))

                road_states['metric'] = torch.tensor(metric_states, dtype=torch.float32)

                # 全体の状態量に追加
                states['roads'][road_order_id] = road_states

            # 交差点の状態量について
            current_phase_id = self.intersection.get('current_phase_id')
            intersection_states = [0] * (self.intersection.get('num_phases'))
            intersection_states[current_phase_id - 1] = 1
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
            for road_order_id, road_states in sorted(roads_states.items()):
                print(road_order_id)
                lanes_states = road_states['lanes']
                for lane_states in lanes_states:
                    vehicle_states.extend(lane_states['vehicles'])

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
            for road_order_id, road_states in sorted(roads_states.items()):
                lanes_states = road_states['lanes']
                for lane_states in lanes_states:
                    lane_shape_states.append(lane_states['shape'])
        
        lane_shape_states = torch.stack(lane_shape_states)    
        lane_shape_outputs = self.lane_shape_net(lane_shape_states)

        # 車線ごとの評価指標について
        lane_metric_states = []

        for states in x:
            roads_states = states['roads']
            for road_order_id, road_states in sorted(roads_states.items()):
                lanes_states = road_states['lanes']
                for lane_states in lanes_states:
                    lane_metric_states.append(lane_states['metric'])
        
        lane_metric_states = torch.stack(lane_metric_states)
        lane_metric_outputs = self.lane_metric_net(lane_metric_states)

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
        pass

        


