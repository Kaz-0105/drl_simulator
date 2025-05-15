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

        # モデルのロードor初期化
        if self.config.get('drl_info')['method'] =='a2c':
            model_path = Path('models/a2c.pth')

            # モデルの初期化
            if model_path.exists():
                self.model = torch.load(model_path)
            else:
                self.model = A2CNet(self)

    def run(self):
        # 状態量の取得
        state = self.getStates()

    def getStates(self):
        if self.config.get('drl_info')['method'] == 'a2c':
            states = {}

            for road in self.intersection.input_roads.getAll():
                road_states ={}
                for link in road.links.getAll():
                    # connectorはsubと一緒に扱うためここではスキップ
                    if link.get('type') == 'connector':
                        continue

                    if link.get('type') == 'main':
                        for lane in link.lanes.getAll():
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
                            vehicles_states = {}
                            feature_names = ['position', 'speed', 'in_queue', 'direction']
                            for index in range(num_vehicles):
                                if index < len(vehicle_data):
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
                                    vehicles_states[index + 1] = np.array(vehicle_states, dtype=np.float32)
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
                                    vehicles_states[index + 1] = np.round(vehicle_states).astype(np.float32)
                            
                            # 車線の状態量に追加
                            lane_states['vehicles'] = vehicles_states

                            # 評価指標に関する状態量を取得
                            
                            # 車線情報に関する状態量を取得
                            


        

class A2CNet(nn.Module):
    def __init__(self, controller):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = controller.config
        self.executor = controller.executor
        self.controller = controller
    
    def forward(self, x):
        # xの形状は（バッチサイズ，
        pass

# class A2CRoadNet(nn.Module):
#     def __init__(self, intersection_net):
#         # 継承
#         super().__init__()

#         # 設定オブジェクトと上位の紐づくオブジェクトを取得
#         self.config = intersection_net.config
#         self.executor = intersection_net.executor
#         self.intersection_net = intersection_net

#         # 下位のネットワークを取得
#         self.link_net = A2CLinkNet(self)

# class A2CLinkNet(nn.Module):
#     def __init__(self, road_net, num_lanes):
#         # 継承
#         super().__init__()

#         # 設定オブジェクトと上位の紐づくオブジェクトを取得
#         self.config = road_net.config
#         self.executor = road_net.executor
#         self.road_net = road_net

#         # 下位のネットワークを取得
#         self.lane_net = A2CLaneNet(self)

#         # 車線数を取得
#         self.num_lanes = num_lanes

#         # 各種パラメータを取得
#         self.feature_dim = num_lanes * self.lane_net.output_dim
#         self.input_dim = self.feature_dim
#         self.hidden_dim = 64
#         self.output_dim = self.feature_dim

#         self.net = nn.Sequential(
#             nn.Linear(self.lane_net.input_dim, self.lane_net.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.lane_net.hidden_dim, self.lane_net.output_dim),
#             nn.ReLU(),
#         )
    
#     def forward(self, x):
#         # xの形状は（バッチサイズ，特徴量数の辞書型配列）のリスト
#         pass

# class A2CLaneNet(nn.Module):
#     def __init__(self, link_net):
#         # 継承
#         super().__init__()

#         # 設定オブジェクトと上位の紐づくオブジェクトを取得
#         self.config = link_net.config
#         self.executor = link_net.executor
#         self.link_net = link_net

#         # 下位のネットワークを取得
#         self.vehicles_net = A2CVehiclesNet(self)
#         self.lane_shape_net = A2CLaneShapeNet(self)
#         self.lane_metric_net = A2CLaneMetricNet(self)

#         # 各種パラメータを取得
#         self.feature_dim = self.vehicles_net.output_dim + self.lane_shape_net.output_dim + self.lane_metric_net.output_dim
#         self.input_dim = self.feature_dim
#         self.hidden_dim = 64
#         self.output_dim = self.feature_dim
        
#         self.net = nn.Sequential(
#             nn.Linear(self.input_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.output_dim),
#             nn.ReLU(),
#         )
    
#     def forward(self, x):
#         # xの形状は{vehicles: (バッチサイズ，自動車台数，特徴量数), lane_shape: (バッチサイズ，特徴量数), lane_metric: (バッチサイズ，特徴量数)}
#         x['vehicles'] = self.vehicles_net(x['vehicles'])
#         x['lane_shape'] = self.lane_shape_net(x['lane_shape'])
#         x['lane_metric'] = self.lane_metric_net(x['lane_metric'])

#         # xを（バッチサイズ，特徴量数）に変換
#         x = torch.cat(x['vehicles'], x['lane_shape'], x['lane_metric'], dim=1)

#         # ネットワークを通す
#         return self.net(x)

# class A2CVehiclesNet(nn.Module):
#     def __init__(self, lane_net):
#         # 継承
#         super().__init__()

#         # 設定オブジェクトと上位の紐づくオブジェクトを取得
#         self.config = lane_net.config
#         self.executor = lane_net.executor
#         self.lane_net = lane_net

#         # 自動車台数を取得
#         drl_info = self.config.get('drl_info')
#         self.num_vehicles = drl_info['num_vehicles']

#         # 下位のネットワークを取得
#         self.vehicle_net = A2CVehicleNet(self)

#         # 各種パラメータを取得
#         self.feature_dim = self.num_vehicles * self.vehicle_net.output_dim
#         self.input_dim = self.feature_dim
#         self.hidden_dim = 64
#         self.output_dim = self.feature_dim

#         self.net = nn.Sequential(
#             nn.Linear(self.input_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.output_dim),
#             nn.ReLU(),
#         )
    
#     def forward(self, x):
#         # xの形状は（バッチサイズ，自動車台数，特徴量数）
#         # xを（バッチサイズ * 自動車台数, 特徴量数）に変換
#         x = x.view(-1, self.vehicle_net.input_dim)

#         # 各車両の特徴量を抽出
#         output = self.vehicle_net(x)

#         # xを（バッチサイズ, 自動車台数 * 特徴量数）に変換
#         output = output.view(-1, self.input_dim)

#         # ネットワークを通す
#         return self.net(output)
        

# class A2CVehicleNet(nn.Module):
#     def __init__(self, vehicles_net):
#         # 継承
#         super().__init__()

#         # 設定オブジェクトと上位の紐づくオブジェクトを取得
#         self.config = vehicles_net.config
#         self.executor = vehicles_net.executor
#         self.vehicles_net = vehicles_net

#         # 各種パラメータを取得
#         drl_info = self.config.get('drl_info')
#         self.feature_dim = 0
#         for value in drl_info['features']['vehicle'].values():
#             if value == True:
#                 self.feature_dim += 1

#         self.input_dim = self.feature_dim
#         self.hidden_dim = 64
#         self.output_dim = self.feature_dim

#         self.net = nn.Sequential(
#             nn.Linear(self.input_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.output_dim),
#             nn.ReLU(),
#         )
    
#     def forward(self, x):
#         # xの形状は（バッチサイズ * 自動車台数，特徴量数）
#         return self.net(x)
    
# class A2CLaneShapeNet(nn.Module):
#     def __init__(self, lane_net):
#         # 継承
#         super().__init__()

#         # 設定オブジェクトと上位の紐づくオブジェクトを取得
#         self.config = lane_net.config
#         self.executor = lane_net.executor
#         self.lane_net = lane_net

#         # 各種パラメータを取得
#         intersection = lane_net.link_net.road_net.intersection_net.intersection
#         self.num_roads = intersection.input_roads.count()

#         self.feature_dim = (self.num_roads - 1) + 1
#         self.input_dim = self.feature_dim
#         self.hidden_dim = 64
#         self.output_dim = self.feature_dim

#         self.net = nn.Sequential(
#             nn.Linear(self.input_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.output_dim),
#             nn.ReLU(),
#         )
    
#     def forward(self, x):
#         # xの形状は（バッチサイズ，特徴量数）
#         return self.net(x)

# class A2CLaneMetricNet(nn.Module):
#     def __init__(self, lane_net):
#         # 継承
#         super().__init__()

#         # 設定オブジェクトと上位の紐づくオブジェクトを取得
#         self.config = lane_net.config
#         self.executor = lane_net.executor
#         self.lane_net = lane_net

#         # 各種パラメータを取得
#         drl_info = self.config.get('drl_info')
#         self.feature_dim = 0
#         for value in drl_info['features']['lane'].values():
#             if value == True:
#                 self.feature_dim += 1
        
#         self.input_dim = self.feature_dim
#         self.hidden_dim = 64
#         self.output_dim = self.feature_dim

#         self.net = nn.Sequential(
#             nn.Linear(self.input_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.output_dim),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         # xの形状は（バッチサイズ，特徴量数）
#         return self.net(x)



        

        


