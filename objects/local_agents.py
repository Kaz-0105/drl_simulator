from libs.container import Container
from libs.object import Object
from objects.links import Lanes
from neural_networks.q_net_1 import QNet1

import torch
import random
from collections import deque
import pandas as pd
import time

class LocalAgents(Container):
    def __init__(self, upper_object, device=None):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

        # 引継ぎデータ格納用のオブジェクトを取得
        self.shared_resources = upper_object.shared_resources

        # 上位オブジェクトによって分岐
        if upper_object.__class__.__name__ == 'Network':
            # デバイスを設定
            self.device = device

            # 上位の紐づくオブジェクトを取得
            self.network = upper_object

            # 要素オブジェクトを作成
            self._makeElements()

        elif upper_object.__class__.__name__ == 'MasterAgent':
            # 上位の紐づくオブジェクトを取得
            self.master_agent = upper_object
    
    def _makeElements(self):
        intersections = self.network.intersections
        for intersection in intersections.getAll(sorted_flg=True):
            self.add(LocalAgent(self, intersection))
    
    def getState(self):
        # 非同期で状態量を取得
        for agent in self.getAll():
            self.executor.submit(agent.getState)
            
        # 全ての状態量取得が終わるまで待機
        self.executor.wait()
        return
    
    def getAction(self):
        # 非同期で行動を取得
        for agent in self.getAll():
            self.executor.submit(agent.getAction)

        # 全ての行動取得が終わるまで待機
        self.executor.wait()
        return
    
    def getReward(self):
        # 非同期で報酬を取得
        for agent in self.getAll():
            self.executor.submit(agent.getReward)

        # 全ての報酬取得が終わるまで待機
        self.executor.wait()

    def makeLearningData(self):
        # データを送信
        for agent in self.getAll():
            self.executor.submit(agent.makeLearningData)
        
        # 全てのデータ保存が終わるまで待機
        self.executor.wait()
    
    @property
    def done_flg(self):
        for agent in self.getAll():
            if agent.done_flg:
                return True
    
        return False
    
class LocalAgent(Object):
    def __init__(self, local_agents, intersection):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = local_agents.config
        self.executor = local_agents.executor

        # 引継ぎデータ格納用のオブジェクトを取得
        self.shared_resources = local_agents.shared_resources

        # デバイスを設定
        self.device = local_agents.device

        # 上位オブジェクトを取得
        self.local_agents = local_agents

        # IDを設定
        self.id = self.local_agents.count() + 1

        # intersectionオブジェクトと紐づける
        self.intersection = intersection
        self.intersection.set('local_agent', self)

        # signal_controllerオブジェクトと紐づける
        self.signal_controller = self.intersection.signal_controller

        # networkオブジェクトと紐づける
        self.network = self.local_agents.network

        # master_agentと紐づける
        self._makeMasterAgentConnections()

        # roadオブジェクトおよびlaneオブジェクトと紐づける（一方通行）
        self.roads = self.intersection.input_roads
        self._makeRoadLanesMap()

        # DRL共通のパラメータを設定
        self._initDrlParameters()

        # ε-greedyで選ばれる行動の確率を設定
        self._makeRandomPhaseProbs()

        # APEXに関するパラメータを設定
        self._initApeXParameters()

        # ネットワークを作成してマスターと同期させる
        self._makeModel()
        self._syncModel()

        # 状態量，行動，報酬，終了フラグを初期化
        self.current_state = None
        self.current_action = None
        self.current_reward = None
        self.done_flg = False

        # トータルのリワードを初期化
        self.total_reward = 0

        # バッファーに送る学習データを格納するためのリストを初期化
        self.learning_data = []

        # 状態，行動，報酬を一時的にストックするための変数を初期化
        self.state_record = deque(maxlen=self.td_steps + 1)
        self.action_record = deque(maxlen=self.td_steps)
        self.reward_record = deque(maxlen=self.td_steps)

        # 計算時間の記録を初期化
        self._initCalculationTimeRecord()

        return
            
    def _makeMasterAgentConnections(self):
        self.master_agent = self.intersection.get('master_agent')
        self.master_agent.local_agents.add(self)
        return
    
    def _makeRoadLanesMap(self):
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

    def _initDrlParameters(self):
        drl_info = self.config.get('drl_info')
        self.network_id = drl_info['network_id']
        self.features_info = drl_info['features']
        self.duration_steps = drl_info['duration_steps']
        self.num_vehicles = drl_info['num_vehicles']
        self.num_lanes_map = self.master_agent.num_lanes_map
        self.reward_id = drl_info['reward_id']
        return
    
    def _makeRandomPhaseProbs(self):
        num_roads_phases_map = self.config.get('num_roads_phases_map')
        phases = num_roads_phases_map[self.num_roads]
        self.random_phase_probs = {}
    
        for _, row in phases.iterrows():
            self.random_phase_probs[int(row['id'])] = float(row['random_prob'])
        
        # 確率の合計が1になるように正規化
        total_prob = sum(self.random_phase_probs.values())
        for phase_id in self.random_phase_probs:
            self.random_phase_probs[phase_id] /= total_prob

        return
    
    def _initApeXParameters(self):
        apex_info = self.config.get('apex_info')
        self.td_steps = apex_info['td_steps']
        self.epsilon = apex_info['epsilon']
        self.gamma = apex_info['gamma']
        return
    
    def _initCalculationTimeRecord(self):
        records_info = self.config.get('records_info')
        self.calc_time_flg = records_info['metric']['calc_time_flg']
        if self.calc_time_flg:
            self.calc_time_record = pd.DataFrame(columns=['time', 'calculation_time'])
        return
    
    def _makeModel(self):
        # モデルを初期化
        if (self.network_id == 1):
            self.model = QNet1(self.config, self.device, self.master_agent.num_vehicles, self.master_agent.num_lanes_map)

        self.model.eval()
        self.model.to(self.device)
        return
        
    def _syncModel(self):
        master_agent_model = self.master_agent.get('model')
        self.model.load_state_dict(master_agent_model.state_dict())
        return

    def _updateVehicleData(self):
        # 車両データのマップを初期化
        self.lane_str_vehicle_data_map = {} 

        # 信号付近の定義で利用するため最大キュー長を取得
        max_queue_length = self.intersection.get('max_queue_length')

        # 道路を走査
        for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
            # roadオブジェクトとlanesオブジェクトを取得
            road = self.roads[road_order_id]
            lanes = self.road_lanes_map[road_order_id]

            # direction_signal_value_mapを取得（信号待ちの状態量が必要な場合）
            direction_signal_value_map = self.roads[road_order_id].get('direction_signal_value_map')

            # 信号付近の距離を定義
            v_max = road.get('max_speed')
            near_length = max_queue_length if max_queue_length > v_max else v_max

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
                near_flgs = []
                for _, row in vehicle_data.iterrows():
                    if row['position'] <=  near_length:
                        near_flgs.append(True)
                    else:
                        near_flgs.append(False)
                
                vehicle_data['near_flg'] = near_flgs

                direction_ids = vehicle_data['direction_id']

                
                # wait_flgを初期化
                wait_flgs = []
                for idx, row in vehicle_data.iterrows():
                    # 交差点に近くない自動車はスコープから外す
                    if not near_flgs[idx]:
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
                
                self.lane_str_vehicle_data_map[lane_str] = vehicle_data 
        return

    def getState(self):
        # 状態量を取得するタイミングかどうかを確認
        if not self.infer_flg:
            return
        
        if self.network_id == 1:
            # 自動車に関する情報を更新
            self._updateVehicleData()

            # 状態量を初期化
            state = {}

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
                                    direction_vector = [0] * (self.num_roads)
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
                                    direction_vector = [0] * (self.num_roads)
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
            state['roads'] = dict(sorted(roads_state.items()))

            # フェーズに関する状態量を取得
            current_phase_id = self.intersection.get('current_phase_id')
            phase_state = [0] * (self.intersection.get('num_phases'))
            if current_phase_id is not None:
                phase_state[current_phase_id - 1] = 1
            else:
                phase_state[0] = 1

            # statesに交差点の状態量を追加
            state['phase'] = torch.tensor(phase_state, dtype=torch.float32)

        # 状態量をインスタンス変数に保存
        self.current_state = state
        self.state_record.append(state)
        return
    
    def getAction(self):
        # 推論の必要がないときはスキップ
        if not self.infer_flg:
            return
        
        # ε-greedy法に従って行動を選択
        if random.random() < self.epsilon:
            action = random.choices(
                list(self.random_phase_probs.keys()),
                weights=list(self.random_phase_probs.values()),
                k=1
            )[0]
        else:
            # 計算時間の測定開始
            if self.calc_time_flg:
                start_time = time.time()

            # 推論を行う
            with torch.no_grad():
                self.model.set('requires_grad_flg', False)
                action_values = self.model([self.current_state])
                action = torch.argmax(action_values).item() + 1
            
            # 計算時間の測定終了して記録
            if self.calc_time_flg:
                end_time = time.time()
                calc_time = end_time - start_time
                self.calc_time_record.loc[len(self.calc_time_record)] = [self.current_time, calc_time]

        # 行動をインスタンス変数に保存
        self.current_action = action
        self.action_record.append(action)

        # 信号機の将来のフェーズに追加
        self.signal_controller.setNextPhases([self.current_action] * self.duration_steps)
            

        return
    
    def getReward(self):
        # 報酬を計算するタイミングかどうかを確認
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
            self.current_reward = score
        
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
        
            # current_rewardを更新
            self.current_reward = score

        # 記録する
        self.reward_record.append(self.current_reward)
        self.total_reward += self.current_reward 
       
    def makeLearningData(self):
        # 推論の必要がないときはスキップ
        if self.infer_flg == False:
            return
        
        # データが溜まっていない場合はスキップ
        if len(self.state_record) != self.td_steps + 1:
            return
        
        # 状態について
        state = self.state_record[0]
        next_state = self.state_record[-1]

        # 行動について
        action = self.action_record[0]

        # 累積報酬について
        cumulative_reward = 0
        for reward in list(reversed(self.reward_record)):
            cumulative_reward = reward + self.gamma * cumulative_reward

        # 終了フラグについて
        done = int(self.done_flg)

        # マスターに送るデータを作成
        self.learning_data.append((state, action, cumulative_reward, next_state, done))
    
    @property
    def infer_flg(self):
        # 現在残っている将来のフェーズを取得
        future_phase_ids = self.signal_controller.get('future_phase_ids')
        return len(future_phase_ids) <= 1

    @property
    def evaluate_flg(self):
        return self.infer_flg

    @property
    def current_time(self):
        return self.network.simulation.get('current_time')

    @property
    def num_roads(self):
        return self.intersection.get('num_roads')
    

