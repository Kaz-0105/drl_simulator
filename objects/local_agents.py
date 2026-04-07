from libs.container import Container
from libs.object import Object
from objects.links import Lanes
from neural_networks.q_net_1 import QNet1

import torch
import random
from collections import deque
import pandas as pd
import numpy as np
import time

class LocalAgents(Container):
    def __init__(self, upper_object, device=None):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor
        self.shared_resources = upper_object.shared_resources

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object 
            self.device = device
            self._initElements()

        elif upper_object.__class__.__name__ == 'MasterAgent':
            self.master_agent = upper_object 
        
        else:
            raise ValueError(f"Not supported upper_object: {upper_object.__class__.__name__}")

        return
    
    def _initElements(self):
        for intersection in self.network.intersections.getAll(sorted_flg=True):
            self.add(LocalAgent(self, intersection))
    
    def getState(self):
        for agent in self.getAll():
            self.executor.submit(agent.getState) 
        self.executor.wait()
        return

    def getAction(self):
        for agent in self.getAll():
            self.executor.submit(agent.getAction)
        self.executor.wait()
        return
    
    def getReward(self):
        for agent in self.getAll():
            self.executor.submit(agent.getReward)
        self.executor.wait()
        return

    def makeLearningData(self):
        for agent in self.getAll():
            self.executor.submit(agent.makeLearningData)
        self.executor.wait()
        return
    
    @property
    def done_flg(self):
        for agent in self.getAll():
            if agent.get('done_flg'):
                return True
        return False
    
class LocalAgent(Object):
    TOTALLY_RANDOM = 1
    NUM_VEHICLES_RANDOM = 2
    RED = 1
    GREEN = 3

    def __init__(self, local_agents, intersection):
        super().__init__()

        self.config = local_agents.config
        self.executor = local_agents.executor
        self.shared_resources = local_agents.shared_resources
        self.local_agents = local_agents
        self.network = local_agents.network
        self.device = local_agents.device

        self.id = self.local_agents.count() + 1

        # connect intersection
        self.intersection = intersection
        self.intersection.set('local_agent', self)

        # set roads, signal_controller
        self.roads = self.intersection.input_roads
        self.signal_controller = self.intersection.signal_controller

        # connect master_agent
        self.master_agent = self.intersection.get('master_agent')
        self.master_agent.local_agents.add(self)

        self._initProps()

        # road_lanes_mapを作成
        self._makeRoadLanesMap()

        # フェーズ情報を取得
        self._makePhases()

        # DNN初期化
        self._makeModel()
        self._syncModel()
        return
    
    def _initProps(self):
        self.num_roads = self.intersection.get('num_roads')

        self.symmetry_phase_map = self.master_agent.get('symmetry_phase_map')
        self.epsilon = self.master_agent.get('epsilon')
        self.num_vehicles = self.master_agent.get('num_vehicles')
        self.num_lanes_map = self.master_agent.get('num_lanes_map') 
        self.random_phase_prob_map = self.master_agent.get('random_phase_prob_map')

        drl_info = self.config.get('drl_info')
        self.network_id = drl_info['network_id']
        self.reward_id = drl_info['reward_id']
        self.done_reward = drl_info['done_reward']
        self.state_id = drl_info['state_id']
        self.features_info = drl_info['features']
        self.data_augmentation_flg = drl_info['data_augmentation_flg']
        self.duration_steps = drl_info['duration_steps']
        self.num_vehicles = drl_info['num_vehicles']

        apex_info = self.config.get('apex_info')
        self.td_steps = apex_info['td_steps']
        self.gamma = apex_info['gamma']
        self.epsilon = self.master_agent.get('epsilon')
        self.random_action_type = apex_info['random_action_type']

        self.state_record = deque(maxlen=self.td_steps + 1)
        self.action_record = deque(maxlen=self.td_steps)
        self.reward_record = deque(maxlen=self.td_steps)
        self.calc_time_record = []
        self.current_state, self.current_action, self.current_reward = None, None, None
        self.done_flg = False
        self.total_reward = 0
        self.learning_data = []

        return
    
    # キー：道路ID，値：lanesオブジェクトの辞書を作成するメソッド
    # lanesオブジェクトに格納されるlaneオブジェクトは右分岐車線から順番にラベル付けされる
    def _makeRoadLanesMap(self):
        self.road_lanes_map = {}
        for road_order_id in range(1, self.num_roads + 1):
            road = self.roads[road_order_id]
            lanes = Lanes(self)

            # 右折分岐車線
            for link in road.links.getAll():
                if link.get('type') != 'right':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            # 中央車線
            for link in road.links.getAll():
                if link.get('type') != 'main':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            # 左折分岐車線
            for link in road.links.getAll():
                if link.get('type') != 'left':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            self.road_lanes_map[road_order_id] = lanes

        return
    
    def _makePhases(self):
        # フェーズ情報を取得
        self.phases = self.signal_controller.get('phases', type='copy')

        for phase_id in list(self.phases.keys()):
            phase_prob = self.random_phase_prob_map[phase_id]
            if phase_prob == 0:
                del self.phases[phase_id]

        return 
    
    # DNNを初期化するメソッド
    def _makeModel(self):
        if (self.network_id == 1):
            self.model = QNet1(self.config, self.device, self.num_vehicles, self.num_lanes_map)

        self.model.eval()
        self.model.to(self.device)
        return
    
    # master_agentのQネットワークと同期するメソッド
    def _syncModel(self):
        master_agent_model = self.master_agent.get('model')
        self.model.load_state_dict(master_agent_model.state_dict())
        return

    # 車両情報を更新するメソッド
    def _updateVehiclesDf(self):
        self.lane_str_vehicles_df_map = {} 
        for road_order_id in range(1, self.num_roads + 1):
            road = self.roads[road_order_id]
            lanes = self.road_lanes_map[road_order_id]

            # get needed information for making vehicle_data
            if self.reward_id in [1, 2]:
                direction_signal_color_map = road.get('direction_signal_color_map')
                v_max = road.get('max_speed')
                max_queue_length = self.intersection.get('max_queue_length')
                near_length = max_queue_length if max_queue_length > v_max else v_max

            for lane_order_id in lanes.getKeys(container_flg=True, sorted_flg=True):
                lane_str = f"{road_order_id}-{lane_order_id}"
                lane = lanes[lane_order_id]
                
                vehicles_df = lane.get('vehicles_df').copy()
                vehicles_df = vehicles_df.sort_values(by='position', ascending=False)
                vehicles_df = vehicles_df.reset_index(drop=True)

                # positionの定義
                length_info = lane.get('length_info')
                vehicles_df['position'] = length_info['length'] - vehicles_df['position']

                if self.reward_id in [1, 2]:
                    # define near_flg
                    near_flgs = []
                    for _, vehicle_row in vehicles_df.iterrows():
                        near_flgs.append(True if vehicle_row['position'] <= near_length else False)
                    vehicles_df['near_flg'] = near_flgs

                    # define red_flgs
                    red_flgs = []
                    for _, vehicle_row in vehicles_df.iterrows():
                        signal_color = direction_signal_color_map[vehicle_row['direction_id']] if vehicle_row['direction_id'] != 0 else 'red'
                        red_flgs.append(True if signal_color == self.RED else False)
                    vehicles_df['red_flg'] = red_flgs
                    
                    # define wait_flg
                    wait_flgs = []
                    direction_ids = vehicles_df['direction_id']
                    for idx, vehicle_row in vehicles_df.iterrows():
                        if not near_flgs[idx]:
                            wait_flgs.append(False)
                            continue

                        if red_flgs[idx]:
                            wait_flgs.append(True)
                            continue
                        
                        if len(wait_flgs) == 0:
                            wait_flgs.append(False)
                            continue

                        found_flg = False
                        for tmp_idx in reversed(range(len(wait_flgs))):
                            if direction_ids[tmp_idx] == vehicle_row['direction_id']:
                                continue

                            wait_flgs.append(True if red_flgs[tmp_idx] else False)
                            found_flg = True
                            break
                        
                        if found_flg:
                            continue
                        
                        wait_flgs.append(False)
                    vehicles_df['wait_flg'] = wait_flgs
                
                self.lane_str_vehicles_df_map[lane_str] = vehicles_df 
        return

    def _updateRoadMaxQueueMap(self):
        self.road_max_queue_map = {}
        for road_order_id in range(1, self.num_roads + 1):
            max_queue_length = 0
            lanes = self.road_lanes_map[road_order_id]
            for lane_order_id in range(1, lanes.count() + 1):
                vehicle_data = self.lane_str_vehicles_df_map[f"{road_order_id}-{lane_order_id}"]
                for _, row in vehicle_data[::-1].iterrows():
                    if row['speed'] < 10.0:
                        position = row['position']
                        max_queue_length = position if position > max_queue_length else max_queue_length
                        break

            self.road_max_queue_map[road_order_id] = max_queue_length
        return

    # 状態を取得するメソッド
    def getState(self):
        if not self.infer_flg:
            return
        
        # 自動車に関する情報を更新
        self._updateVehiclesDf()
        self._updateRoadMaxQueueMap()

        if self.network_id == 1:
            # 状態を作成
            state = {}
            current_phase_id = self.intersection.get('current_phase_id')
            phase_state = [0] * (self.intersection.get('num_phases'))
            if current_phase_id is not None:
                phase_state[current_phase_id - 1] = 1
            else:
                phase_state[0] = 1
            state['phase'] = torch.tensor(phase_state).float()

            # 道路の状態を作成
            roads_state = {}
            for road_order_id in range(1, self.num_roads + 1):
                road = self.roads[road_order_id]
                road_state = {}
                metric_state = []
                metric_state.append(int(self.road_max_queue_map[road_order_id]))
                metric_state.append(0 if np.isnan(road.get('average_delay')) else int(road.get('average_delay')))
                road_state['metric'] = torch.tensor(metric_state, dtype=torch.float32)
                
                # 車線の状態を作成
                lanes_state = {}
                lanes = self.road_lanes_map[road_order_id]
                for lane_order_id in range(1, lanes.count() + 1):
                    lane = lanes[lane_order_id]
                    lane_state = {}
                    lane_state['metric'] = torch.tensor([lane.get('num_vehicles')]).float()
                    length_info = lane.get('length_info')
                    if lane.link.get('type') == 'main':
                        lane_state['shape'] = torch.tensor([int(length_info['length']), 1, 0]).float()
                    elif lane.link.get('type') == 'right' or lane.link.get('type') == 'left':
                        lane_state['shape'] = torch.tensor([int(length_info['length']), 0, 1]).float()

                    # 自動車の状態を作成
                    vehicle_data = self.lane_str_vehicles_df_map.get(f"{road_order_id}-{lane_order_id}")
                    vehicles_state = {}
                    for index in range(self.num_vehicles):
                        if index < vehicle_data.shape[0]:
                            vehicle = vehicle_data.iloc[index]
                            vehicle_state = []
                            for feature_name, feature_flg in self.features_info['vehicle'].items():
                                if feature_flg == False:
                                    continue

                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.num_roads)
                                    direction_vector[int(vehicle['direction_id'])] = 1
                                    vehicle_state.extend(direction_vector)
                                else: 
                                    vehicle_state.append(float(vehicle[feature_name]))
                            vehicle_state.append(1)                   
                        else:
                            vehicle_state = []
                            for feature_name, feature_flg in self.features_info['vehicle'].items():
                                if feature_flg == False:
                                    continue

                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.num_roads)
                                    vehicle_state.extend(direction_vector)
                                else: 
                                    vehicle_state.append(0.0)
                            vehicle_state.append(0)
                        vehicles_state[len(vehicles_state) + 1] = torch.tensor(vehicle_state).float()  

                    lane_state['vehicles'] = dict(sorted(vehicles_state.items()))
                    lanes_state[lane_order_id] = lane_state

                road_state['lanes'] = dict(sorted(lanes_state.items()))
                roads_state[road_order_id] = road_state
            
            state['roads'] = dict(sorted(roads_state.items()))

        # 状態を保存
        self.current_state = state
        self.state_record.append(state)
        return

    # 行動を取得するメソッド
    def getAction(self):
        if not self.infer_flg:
            return
        
        # ε-greedy法
        if random.random() < self.epsilon:
            action = self._getRandomAction()
            
        else:
            start_time = time.time()

            with torch.no_grad():
                self.model.set('requires_grad_flg', False)
                action_values = self.model([self.current_state])
                action = torch.argmax(action_values).item() + 1
            
            end_time = time.time()
            calc_time = end_time - start_time
            self.calc_time_record.append({'time': self.current_time, 'calc_time': calc_time})

        # 記録
        self.current_action = action
        self.action_record.append(action)
        self.signal_controller.setNextPhases([self.current_action] * self.duration_steps)
        return
    
    def _getRandomAction(self):
        if self.random_action_type == LocalAgent.TOTALLY_RANDOM:
            action = random.choices(
                list(self.random_phase_prob_map.keys()),
                weights=list(self.random_phase_prob_map.values()),
                k=1
            )[0]
        
        elif self.random_action_type == LocalAgent.NUM_VEHICLES_RANDOM:
            # 各フェーズに何台の自動車が待っているかどうかを調べる
            signal_num_vehs_map = {route_id: 0 for route_id in range(1, self.num_roads * (self.num_roads - 1) + 1)}
            for road_order_id in range(1, self.num_roads + 1):
                lanes = self.road_lanes_map[road_order_id]
                for lane_order_id in range(1, lanes.count() + 1):
                    lane = lanes[lane_order_id]
                    vehicle_data = lane.get('vehicle_data', type='reference')
                    for _, row in vehicle_data.iterrows():
                        direction_id = row['direction_id']
                        if direction_id == 0:
                            continue
                        signal_num_vehs_map[(road_order_id - 1) * (self.num_roads - 1) + direction_id] += 1

            phase_num_vehs_map = {}
            for phase_id, phase_list in self.phases.items():
                tmp_num_vehs = 0
                for signal_id in phase_list:
                    tmp_num_vehs += signal_num_vehs_map[signal_id]
                phase_num_vehs_map[phase_id] = tmp_num_vehs

            # 全てのフェーズで0台の場合は完全ランダム
            if sum(phase_num_vehs_map.values()) == 0:
                action = random.choices(
                    list(self.random_phase_prob_map.keys()),
                    weights=list(self.random_phase_prob_map.values()),
                    k=1
                )[0]
            else:
                action = random.choices(
                    list(phase_num_vehs_map.keys()),
                    weights=list(phase_num_vehs_map.values()),
                    k=1
                )[0]
        else: 
            raise ValueError('not defined random_action_type.')
        
        return action

    
    # 報酬を取得するメソッド
    def getReward(self):
        if not self.evaluate_flg:
            return
        
        if self.reward_id == 1:
            # (the number of not waiting vehicles) - (the number of waiting vehicles)
            reward = 0
            num_vehs = 0

            for _, vehicles_df in self.lane_str_vehicles_df_map.items():
                # skip if there is no vehicle in the lane
                if vehicles_df.shape[0] == 0:
                    continue

                num_vehs += vehicles_df.shape[0]
                reward += (~vehicles_df['wait_flg']).sum()
                reward -= vehicles_df['wait_flg'].sum()

            # normalize the reward
            self.current_reward = reward / num_vehs if num_vehs > 0 else 0
        
        elif self.reward_id == 2:
            # (the number of not-waiting vehicles) - (the number of waiting vehicles) + (the number of passing vehicles)
            reward = 0
            num_vehs = 0

            # the number of waiting and not-waiting vehicles
            for _, vehicles_df in self.lane_str_vehicles_df_map.items():
                if vehicles_df.shape[0] == 0:
                    continue

                num_vehs += vehicles_df.shape[0]
                reward += (~vehicles_df['wait_flg']).sum()
                reward -= vehicles_df['wait_flg'].sum()

            # the number of passing vehicles
            for road in self.roads.getAll():
                for data_collection_point in road.data_collection_points.getAll():
                    if data_collection_point.get('type') != 'intersection':
                        continue

                    for data_collection_measurement in data_collection_point.data_collection_measurements.getAll():
                        if data_collection_measurement.get('type') == 'multiple':
                            continue
                        
                        num_vehs_record = data_collection_measurement.get('num_vehs_record')
                        num_pass_vehs = num_vehs_record['num_vehs'].tail(self.duration_steps).sum()
                        reward += num_pass_vehs
                        num_vehs += num_pass_vehs
            
            # normalize the reward
            self.current_reward = reward / num_vehs if num_vehs > 0 else 0
        
        elif self.reward_id == 3:
            # 一定速度以上の自動車台数 + 通過自動車台数
            self.current_reward = 0
            for lane_str, vehicle_data in self.lane_str_vehicles_df_map.items():
                if vehicle_data.shape[0] == 0:
                    continue

                road_order_id, _ = map(int, lane_str.split('-'))
                road = self.roads[road_order_id]
                v_max = road.get('max_speed')

                for _, row in vehicle_data.iterrows():
                    if row['speed'] > v_max / 2:
                        self.current_reward += 1

            for road in self.roads.getAll():
                for data_collection_point in road.data_collection_points.getAll():
                    if data_collection_point.get('type') != 'intersection':
                        continue

                    for data_collection_measurement in data_collection_point.data_collection_measurements.getAll():
                        if data_collection_measurement.get('type') == 'multiple':
                            continue
                        
                        num_vehs_record = data_collection_measurement.get('num_vehs_record')
                        num_vehs_list = num_vehs_record['num_vehs'].tail(self.duration_steps).tolist()
                        self.current_reward += sum(num_vehs_list)
                        
        elif self.reward_id == 4:
            # 法定速度の半分以上の自動車台数 - 法定速度の半分以下の自動車台数 + 通過自動車台数
            self.current_reward = 0
            for lane_str, vehicle_data in self.lane_str_vehicles_df_map.items():
                if vehicle_data.shape[0] == 0:
                    continue

                road_order_id, _ = map(int, lane_str.split('-'))
                road = self.roads[road_order_id]
                v_max = road.get('max_speed')

                for _, row in vehicle_data.iterrows():
                    if 2 * row['speed'] >= v_max:
                        self.current_reward += 1
                    else:
                        self.current_reward -= 1

            for road in self.roads.getAll():
                for data_collection_point in road.data_collection_points.getAll():
                    if data_collection_point.get('type') != 'intersection':
                        continue

                    for data_collection_measurement in data_collection_point.data_collection_measurements.getAll():
                        if data_collection_measurement.get('type') == 'multiple':
                            continue

                    num_vehs_record = data_collection_measurement.get('num_vehs_record')
                    num_vehs_list = num_vehs_record['num_vehs'].tail(self.duration_steps).tolist()
                    self.current_reward += sum(num_vehs_list)

            self.current_reward /= 10  # 報酬のスケールを調整

        elif self.reward_id == 5:
            # 流入道路の入れるスペースの和
            self.current_reward = 0
            for road_order_id in range(1, self.num_roads + 1):
                road = self.roads[road_order_id]
                space = ((road.get('length') - self.road_max_queue_map[road_order_id]) / road.get('length')) * 10 - 5  # -5〜5に正規化
                self.current_reward += space

        # 記録する
        self.reward_record.append(self.current_reward)
        self.total_reward += self.current_reward 
        return
    
    # 学習データを作成するメソッド
    def makeLearningData(self):
        if self.infer_flg == False:
            return
        
        if len(self.state_record) != self.td_steps + 1:
            return
        
        # 状態，行動，報酬，終了フラグを取得
        state = self.state_record[0]
        next_state = self.state_record[-1]
        action = self.action_record[0]
        cumulative_reward = 0
        for reward in list(reversed(self.reward_record)):
            cumulative_reward = reward + self.gamma * cumulative_reward
        done = int(self.done_flg)

        # データを保存
        self.learning_data.append((state, action, cumulative_reward, next_state, done))

        # データ拡張を実施
        if self.data_augmentation_flg:
            self.data_augmentation_target = self.learning_data[-1]
            self._runDataAugmentation()
        return
    
    # データ拡張をするメソッド
    def _runDataAugmentation(self):
        data_augmentation_type = self.data_augmentation_flg
        if data_augmentation_type == 0:
            return
         
        if self.num_roads == 4:
            state_origin, action_origin, cumulative_reward, next_state_origin, done = self.data_augmentation_target

            if data_augmentation_type == 1:
                symmetry_types = range(1, self.num_roads)
            elif data_augmentation_type == 2:
                symmetry_types = [2]
            else:
                raise ValueError('data_augmentation_type must be 0, 1, or 2.') 
            
            for symmetry_type in symmetry_types:
                # stateを変換
                state = self._rotateState(state_origin, symmetry_type)
                next_state = self._rotateState(next_state_origin, symmetry_type)
                
                # actionを変換
                action = self.symmetry_phase_map[action_origin][symmetry_type]
                self.learning_data.append((state, action, cumulative_reward, next_state, done))
        else:
            raise ValueError('Data augmentation is only available for intersections with 4 roads.')
        return
    
    # 状態を回転させるメソッド（symmetry_type = 1: 90度，2:180度，3:270度）
    def _rotateState(self, state_origin, symmetry_type):
        state = {}
        state['roads'] = {}
        for road_order_id in range(1, self.num_roads + 1):
            new_road_order_id = (road_order_id + symmetry_type - 1) % self.num_roads + 1
            state['roads'][new_road_order_id] = state_origin['roads'][road_order_id]

        phase_id = self._reshapePhaseState(state_origin['phase'])
        symmetry_phase_id = self.symmetry_phase_map[phase_id][symmetry_type]
        phase_state = [0] * (self.intersection.get('num_phases'))
        phase_state[symmetry_phase_id - 1] = 1
        state['phase'] = torch.tensor(phase_state).float()
        return state

    # one-hotベクトルのフェーズ（テンソル）をフェーズIDに変換するメソッド
    def _reshapePhaseState(self, phase_state):
        phase_state = phase_state.tolist()
        for idx, val in enumerate(phase_state):
            if val == 1: 
                return idx + 1
        raise ValueError('Current phase state is invalid.')
    
    @property
    def infer_flg(self):
        future_phase_ids = self.signal_controller.get('future_phase_ids')
        return len(future_phase_ids) <= 1

    @property
    def evaluate_flg(self):
        return self.infer_flg

    @property
    def current_time(self):
        return self.network.simulation.get('current_time')
    

