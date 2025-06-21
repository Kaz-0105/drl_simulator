from libs.common import Common
from objects.links import Lanes
from neural_networks.q_net_1 import QNet1
from neural_networks.q_net_2 import QNet2
from neural_networks.q_net_3 import QNet3

from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class BcAgent(Common):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = network.config
        self.executor = network.executor

        # 上位の紐づくオブジェクトを取得
        self.network = network

        # intersectionオブジェクトと紐づける
        self.intersection = self.network.intersections[1]
        self.intersection.set('bc_agent', self)

        # signal_controllerオブジェクトと紐づける
        self.signal_controller = self.intersection.signal_controller

        # roadsオブジェクトと紐づける
        self.roads = self.intersection.input_roads
        self.num_roads = self.roads.count()

        # パラメータを取得
        self._getDrlParameters()
        self._getBcParameters()

        # 車線数情報を取得
        self._makeNumLanesMap()

        # 道路から車線群へのマップを作成
        self._makeRoadLanesMap()

        # 保存先のパス，学習データのパスを作成
        self._makeSavePath()
        self._makeBufferPathList()

        # モデルを初期化
        self._makeModel()

        # 専門家のデータセットとデータローダーを作成
        self.expert_dataset = ExpertDataset(self.buffer_path_list)
        self.dataloader = DataLoader(
            self.expert_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0, 
            collate_fn=collate_fn
        )

        # 損失関数と最適化手法を設定
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # total_rewardを初期化
        self.total_reward = 0

    def _getDrlParameters(self):
        drl_info = self.config.get('drl_info')
        self.duration_steps = drl_info['duration_steps']
        self.num_vehicles = drl_info['num_vehicles']
        self.reward_id = drl_info['reward_id']
        self.features_info = drl_info['features']
        return
    
    def _getBcParameters(self):
        bc_info = self.config.get('bc_info')
        self.learning_flg = bc_info['learning_flg']
        self.network_id = bc_info['network_id']
        self.learning_rate = bc_info['learning_rate']
        self.batch_size = bc_info['batch_size']
        self.num_epochs = bc_info['num_epochs']
        return
    
    def _makeNumLanesMap(self):
        self.num_lanes_map = {}
        for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
            road = self.roads[road_order_id]
            num_lanes = 0
            for link in road.links.getAll():
                if link.get('type') == 'connector':
                    continue

                num_lanes += link.lanes.count()
            
            self.num_lanes_map[road_order_id] = num_lanes
        return

    def _makeRoadLanesMap(self):
        self.road_lanes_map = {}

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
            
            self.road_lanes_map[road_order_id] = lanes
        return
    
    def _makeSavePath(self):
        lanes_str = ''
        for road_order_id in range(1, self.num_roads + 1):
            lanes_str += str(self.num_lanes_map[road_order_id])

        self.model_path = Path(f"models/bc_q_net_{self.network_id}_{lanes_str}_{self.num_vehicles}.pth")
        return

    def _makeBufferPathList(self):
        self.buffer_path_list = []
        
        lanes_str = ''
        for road_order_id in range(1, self.num_roads + 1):
            lanes_str += str(self.num_lanes_map[road_order_id])
        
        common_path_str = f"buffers/bc_buffer_{self.network_id}_{lanes_str}_{self.num_vehicles}"
        
        idx = 1
        while True:
            buffer_path = Path(f"{common_path_str}_{idx}.pkl")
            if not buffer_path.exists():
                break
            self.buffer_path_list.append(buffer_path)
            idx += 1
        
        if not self.buffer_path_list:
            raise FileNotFoundError(f"No buffer files found in {common_path_str}_*.pkl. \n Please conduct expert data collection first.")
        
        return


    def _makeModel(self):
        if self.network_id == 1:
            self.model = QNet1(self.config, self.num_vehicles, self.num_lanes_map)
        elif self.network_id == 2:
            self.model = QNet2(self.config, self.num_vehicles, self.num_lanes_map)
        elif self.network_id == 3:
            self.model = QNet3(self.config, self.num_lanes_map)
        self.model.train()

        if self.model_path.exists():
            self.model.load(self.model_path)

        return

    def cloneExpert(self):
        if not self.learning_flg:
            return
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            for tmp_states, tmp_actions in self.dataloader:
                # 状態と行動をインスタンス変数に保存
                self.tmp_states = tmp_states
                self.tmp_actions = tmp_actions
                
                # 行動をone-hotベクトルに変換する
                self._makeActionOneHot()

                self.optimizer.zero_grad()

                self.model.set('requires_grad', True)

                q_values = self.model(self.tmp_states)

                loss = self.criterion(q_values, torch.stack(self.tmp_actions))

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
        
        return
    
    def _updateVehicleData(self):
        lane_str_vehicle_data_map = {}
        
        # 道路を走査
        for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
            # lanesオブジェクトを取得
            lanes = self.road_lanes_map[road_order_id]

            # direction_signal_value_mapを取得（信号待ちの状態量が必要な場合）
            if self.features_info['vehicle']['wait_flg']:
                direction_signal_value_map = self.roads[road_order_id].get('direction_signal_value_map')

            # 車線を走査
            for lane_order_id in lanes.getKeys(container_flg=True, sorted_flg=True):
                lane_str = f"{road_order_id}-{lane_order_id}"

                # laneオブジェクトを取得
                lane = lanes[lane_order_id]

                # vehicle_dataを位置情報でソート
                vehicle_data = lane.get('vehicle_data').copy()
                vehicle_data.sort_values(by='position', ascending=False, inplace=True)
                vehicle_data.reset_index(drop=True, inplace=True)

                # 先頭からnum_vehicles台の車両を取得
                vehicle_data = vehicle_data.head(self.num_vehicles).copy()

                # 距離情報を信号との距離に変換
                length_info = lane.get('length_info')
                vehicle_data['position'] = length_info['length'] - vehicle_data['position']

                # near_flgを追加（交差点に近いかどうか）
                if self.features_info['vehicle']['near_flg'] or self.features_info['vehicle']['wait_flg']:
                    near_flgs = []
                    for _, row in vehicle_data.iterrows():
                        if row['position'] <= 100:
                            near_flgs.append(True)
                        else:
                            near_flgs.append(False)
                    
                    vehicle_data['near_flg'] = near_flgs

                direction_ids = vehicle_data['direction_id']

                # wait_flgを追加（信号待ちの状況かどうか）
                if self.features_info['vehicle']['wait_flg']:
                    # wait_flgを初期化
                    wait_flgs = []
                    for _, row in vehicle_data.iterrows():
                        # 交差点に近くない自動車はスコープから外す
                        if row['position'] > 100:
                            wait_flgs.append(False)
                            continue

                        # 信号が赤の場合は信号待ち
                        signal_value = 3 if row['direction_id'] == 0 else direction_signal_value_map[row['direction_id']]
                        if signal_value == 1:
                            wait_flgs.append(True)
                            continue
                        
                        # 先頭車の場合
                        if len(wait_flgs) == 0:
                            wait_flgs.append(False)
                            continue

                        # 先頭車でない場合は進路が異なる先行車を探す
                        found_flg = False
                        for i in range(len(wait_flgs) - 1, - 1, -1):
                            if direction_ids[i] != row['direction_id']:
                                wait_flgs.append(True if wait_flgs[i] else False)
                                found_flg = True
                                break
                        
                        # 先行車が見つからないとき
                        if not found_flg:
                            wait_flgs.append(False)
                            
                        
                    # wait_flgsをvehicle_dataに追加
                    vehicle_data['wait_flg'] = wait_flgs
                
                lane_str_vehicle_data_map[lane_str] = vehicle_data
        
        self.lane_str_vehicle_data_map = lane_str_vehicle_data_map  
        return
    
    def updateState(self):
        if not self.infer_flg:
            return
        
        # 車両のデータを更新
        self._updateVehicleData()
        
        if self.network_id == 1:
            # 状態量を初期化
            self.state = {}

            # 道路群の状態量を初期化
            roads_state = {}

            # 道路を走査
            for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
                # roadオブジェクトを取得
                road = self.roads[road_order_id]

                # 道路の状態量を初期化
                road_state = {}
                
                # 車線群の状態量を初期化
                lanes_state = {}

                # lanesオブジェクトを取得
                lanes = self.road_lanes_map[road_order_id]

                # 車線を走査
                for lane_order_id in lanes.getKeys(container_flg=True, sorted_flg=True):
                    # laneオブジェクトを取得
                    lane = lanes[lane_order_id]

                    # 車線の状態量を初期化
                    lane_state = {}

                    # 自動車のデータを取得
                    vehicle_data = self.lane_str_vehicle_data_map.get(f"{road_order_id}-{lane_order_id}")
                    
                    # 車両に関する状態を取得
                    vehicles_state = {}
                    for index in range(self.num_vehicles):
                        if index < vehicle_data.shape[0]:
                            # レコードを取得
                            vehicle = vehicle_data.iloc[index]

                            # 車両の状態量を初期化
                            vehicle_state = []

                            # 特徴量を走査
                            for feature_name, feature_flg in self.features_info['vehicle'].items():
                                # 使わない状態量はスキップ
                                if feature_flg == False:
                                    continue

                                # 方向に関する状態量はone-hotベクトルに変換，それ以外はそのまま追加
                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.intersection.get('num_roads'))
                                    direction_vector[int(vehicle['direction_id'])] = 1
                                    vehicle_state.extend(direction_vector)
                                else: 
                                    vehicle_state.append(float(vehicle[feature_name]))
                            
                            # 自動車が存在するかどうかのフラグの状態量を追加
                            vehicle_state.append(1) 

                            # テンソルに変換してからvehicles_stateに追加  
                            vehicles_state[len(vehicles_state) + 1] = torch.tensor(vehicle_state).float()                    
                        else:
                            # 車両の状態量を初期化
                            vehicle_state = []

                            # 特徴量を走査
                            for feature_name, feature_flg in self.features_info['vehicle'].items():
                                # 使わない状態量はスキップ
                                if feature_flg == False:
                                    continue
                                
                                # 方向に関する状態量はone-hotベクトルに変換，それ以外はそのまま追加
                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.intersection.get('num_roads'))
                                    vehicle_state.extend(direction_vector)
                                else: 
                                    vehicle_state.append(0.0)
                            
                            # 自動車が存在するかどうかのフラグの状態量を追加
                            vehicle_state.append(0)

                            # テンソルに変換してからvehicles_stateに追加
                            vehicles_state[len(vehicles_state) + 1] = torch.tensor(vehicle_state, dtype=torch.float32)
                    
                    # 車線の状態量に追加
                    lane_state['vehicles'] = dict(sorted(vehicles_state.items()))

                    # 評価指標に関する状態量を取得
                    lane_state['metric'] = torch.tensor([lane.get('num_vehicles')], dtype=torch.float32)

                    # 道路の情報を取得
                    length_info = lane.get('length_info')
                    
                    # 車線情報に関する状態量を取得（長さ，メインリンクかサブリンクか）
                    if lane.link.get('type') == 'main':
                        lane_state['shape'] = torch.tensor([int(length_info['length']), 1, 0], dtype=torch.float32)
                    elif lane.link.get('type') == 'right' or lane.link.get('type') == 'left':
                        lane_state['shape'] = torch.tensor([int(length_info['length']), 0, 1], dtype=torch.float32)

                    # lanes_stateにlane_stateを追加
                    lanes_state[lane_order_id] = lane_state
                
                # road_stateに車線の状態量を追加
                road_state['lanes'] = dict(sorted(lanes_state.items()))

                # 評価指標の状態量について
                metric_state = []
                metric_state.append(int(road.get('max_queue_length')))
                metric_state.append(int(road.get('average_delay')))

                # road_stateに評価指標の状態量を追加
                road_state['metric'] = torch.tensor(metric_state, dtype=torch.float32)

                # roads_stateにroad_stateを追加
                roads_state[road_order_id] = road_state
            
            # statesに道路の状態量を追加
            self.state['roads'] = dict(sorted(roads_state.items()))

            # フェーズに関する状態量を取得
            current_phase_id = self.intersection.get('current_phase_id')
            phase_state = [0] * (self.intersection.get('num_phases'))
            if current_phase_id is not None:
                phase_state[current_phase_id - 1] = 1
            else:
                phase_state[0] = 1

            # statesに交差点の状態量を追加
            self.state['phase'] = torch.tensor(phase_state, dtype=torch.float32)
        
        return

    def updateAction(self):
        if not self.infer_flg:
            return

        with torch.no_grad():
            if self.network_id == 1:
                self.model.set('requires_grad', False)
                q_values = self.model([self.state])
                self.action = torch.argmax(q_values).item() + 1
        
        self.signal_controller.setNextPhases([self.action] * self.duration_steps)

        return     

    def updateReward(self):
        if not self.evaluate_flg:
            return  
        
        if self.reward_id == 1:
            # 信号待ちの自動車の数をカウント
            score = 0
            for _, vehicle_data in self.lane_str_vehicle_data_map.items():
                if vehicle_data.shape[0] == 0:
                    continue

                for _, row in vehicle_data.iterrows():
                    if not row['wait_flg']:
                        score += 1
                    else:
                        score -= 1

            # 報酬を計算（-1から1の範囲に正規化）
            self.reward = score
        
        elif self.reward_id == 2:
            # 信号待ちの自動車の数をカウント
            score = 0
            for _, vehicle_data in self.lane_str_vehicle_data_map.items():
                if vehicle_data.shape[0] == 0:
                    continue

                for _, row in vehicle_data.iterrows():
                    if not row['wait_flg']:
                        score += 1
                    else:
                        score -= 1

            # 前回の行動からの交差点の通過車両の数をカウント
            for road in self.roads.getAll():
                for data_collection_point in road.data_collection_points.getAll():
                    if data_collection_point.get('type') != 'intersection':
                        continue

                    for data_collection_measurement in data_collection_point.data_collection_measurements.getAll():
                        if data_collection_measurement.get('type') == 'multiple':
                            continue
                        
                        num_vehs_record = data_collection_measurement.get('num_vehs_record')
                        num_vehs_list = num_vehs_record['num_vehs'].tail(self.duration_steps).tolist()
                        score += sum(num_vehs_list)
        
            # rewardを更新
            self.reward = score
        
        self.total_reward += self.reward
        return
    
    def _makeActionOneHot(self):
        # 行動の数を取得
        num_actions = self.model.get('output_size')

        # 行動をone-hotベクトルに変換
        for idx in range(len(self.tmp_actions)):
            action = self.tmp_actions[idx]
            one_hot_action = [0] * num_actions
            one_hot_action[action - 1] = 1
            self.tmp_actions[idx] = torch.tensor(one_hot_action, dtype=torch.float32)
        return

    def showTotalReward(self):
        print(f"Total Reward: {self.total_reward}")
        return
    
    def saveModel(self):
        if not  self.learning_flg:
            return
        
        # モデルを保存
        torch.save(self.model.state_dict(), self.model_path)
        return
    
    
    @property
    def infer_flg(self):
        # 現在残っている将来のフェーズを取得
        future_phase_ids = self.signal_controller.get('future_phase_ids')
        return len(future_phase_ids) <= 1

    @property
    def evaluate_flg(self):
        return self.infer_flg
    

class ExpertDataset(Dataset):
    def __init__(self, buffer_path_list):
        self.buffer_path_list = buffer_path_list
        self._makeLearningData()
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
    
    def _makeLearningData(self):
        self.states = []
        self.actions = []

        for buffer_path in self.buffer_path_list:
            with open(buffer_path, 'rb') as f:
                loaded_data = pickle.load(f)
                for _, tmp_data in enumerate(loaded_data):
                    state = tmp_data['state']
                    action = tmp_data['action']

                    self.states.append(state)
                    self.actions.append(action)
            
        return

def collate_fn(batch):
    # 状態と行動をリストで取得
    states, actions = zip(*batch)
    states = list(states)
    actions = list(actions)
    
    return states, actions

        

    