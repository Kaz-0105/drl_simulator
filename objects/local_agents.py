from libs.container import Container
from libs.object import Object
from objects.links import Lanes
from objects.neural_networks.apex.proto_q_net import ProtoQNet

import torch
import random
from collections import deque
import pandas as pd
import time
import copy

class LocalAgents(Container):
    def __init__(self, upper_object, device=None):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor
        self.shared_resources = upper_object.shared_resources

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object 
            self.device = device
            self._initProps()
            self._initElements()

        elif upper_object.__class__.__name__ == 'MasterAgent':
            self.master_agent = upper_object 
        else:
            raise ValueError(f"Not supported upper_object: {upper_object.__class__.__name__}")

        return
    
    def _initProps(self):
        # set simulation_count
        self.simulation_count = self.network.get('simulation_count')

        # set epsilons_df
        self.epsilons_df = self.config.get('epsilons_df').copy()
        if (self.simulation_count - 1) % len(self.epsilons_df) != 0:
            self.epsilons_df = pd.concat([
                self.epsilons_df.iloc[(self.simulation_count - 1) % len(self.epsilons_df):],
                self.epsilons_df.iloc[:(self.simulation_count - 1) % len(self.epsilons_df)]
             ], ignore_index=True)  
        
        return
    
    def _initElements(self):
        for intersection in self.network.intersections.getAll(sorted_flg=True):

            self.add(LocalAgent(
                local_agents=self, 
                intersection=intersection, 
                epsilon=float(self.epsilons_df.iloc[intersection.get('id') - 1]['epsilon'])
            ))
        return
    
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
    
    def syncDataFrame(self):
        for agent in self.getAll():
            self.executor.submit(agent.syncDataFrame)
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

    def __init__(self, local_agents, intersection, epsilon):
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

        self._initProps(epsilon)

        # set road_lanes_map
        self._makeRoadLanesMap()

        # set data_augmentation_type
        if self.data_augmentation_flg:
            self._makeDataAugmentationType()

        # initialize model
        self._makeModel()
        self._syncModel()
        return
    
    @property
    def current_state(self):
        return self.state_record[-1] if len(self.state_record) > 0 else None

    @property
    def num_learning_data(self):
        return len(self.learning_data_list)
    
    def _initProps(self, epsilon):
        # set num_roads and num_phases
        self.num_roads = self.master_agent.get('num_roads')
        self.num_phases = self.master_agent.get('num_phases')

        self.num_lanes_map = self.master_agent.get('num_lanes_map') 
        self.symmetry_phase_map = self.master_agent.get('symmetry_phase_map')
        self.random_phase_prob_map = self.master_agent.get('random_phase_prob_map')
        self.phase_map = self.signal_controller.get('phase_map')

        # set epsilon
        self.epsilon = epsilon

        # set other drl information
        drl_info = self.config.get('drl_info')
        self.duration_steps = drl_info['duration_steps']
        self.td_steps = drl_info['framework']['apex']['td_steps']
        self.architecture = drl_info['architecture']['type']

        self.num_vehicles = drl_info['state']['vehicle']['number']
        self.state_info = drl_info['state']
        self.random_action_type = drl_info['action']['random_type']
        self.reward_id = drl_info['reward']['id']
        self.gamma = float(drl_info['reward']['gamma'])
        self.data_augmentation_flg = drl_info['data_augmentation']['flg']

        # initialize other properties
        self.state_record = deque(maxlen=self.td_steps + 1)
        self.action_record = deque(maxlen=self.td_steps)
        self.reward_record = deque(maxlen=self.td_steps)
        self.calc_time_record_list = []
        self.done_flg = False
        self.total_reward = 0
        self.learning_data_list = []
        return
    
    def _makeRoadLanesMap(self):
        self.road_lanes_map = {}
        for road_order_id in range(1, self.num_roads + 1):
            road = self.roads[road_order_id]
            lanes = Lanes(self)

            # right branching lane
            for link in road.links.getAll():
                if link.get('type') != 'right':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            # main lane
            for link in road.links.getAll():
                if link.get('type') != 'main':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            # left branching lane
            for link in road.links.getAll():
                if link.get('type') != 'left':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            self.road_lanes_map[road_order_id] = lanes

        return
    
    def _makeDataAugmentationType(self):
        if self.num_roads == 4:
            if len(self.num_lanes_map.values()) == 1:
                self.data_augmentation_type = 1
            elif self.num_lanes_map[1] == self.num_lanes_map[3] and self.num_lanes_map[2] == self.num_lanes_map[4]:
                self.data_augmentation_type = 2
            else:
                self.data_augmentation_type = 0
        else:
            raise NotImplementedError(f"Not supported number of roads: {self.num_roads}")
    
    # DNNを初期化するメソッド
    def _makeModel(self):
        if self.architecture == 'proto':
            self.model = ProtoQNet(self)
        else:
            raise NotImplementedError(f"Not supported architecture: {self.architecture}")

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
        self.vehicles_df_map = {}
        for road_id in range(1, self.num_roads + 1):
            road = self.roads[road_id]
            lanes = self.road_lanes_map[road_id]    

            # get needed information for making vehicle_data
            if self.reward_id in [1, 2]:
                route_signal_color_map = road.get('route_signal_color_map')
                v_max = road.get('max_speed')
                max_queue_length = self.intersection.get('max_queue_length')
                near_length = max_queue_length if max_queue_length > v_max else v_max

            for lane_order_id in lanes.getKeys(container_flg=True, sorted_flg=True):
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
                        signal_color = route_signal_color_map[vehicle_row['direction_id']] if vehicle_row['direction_id'] != 0 else 'red'
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
                
                self.vehicles_df_map[road_id, lane_order_id] = vehicles_df
        return

    def getState(self):
        if not self.infer_flg:
            return
        
        # 自動車に関する情報を更新
        self._updateVehiclesDf()

        if self.architecture == 'proto':
            state = {}

            # phase
            phase_feature_list = [0] * self.num_phases
            
            phase_id = self.intersection.get('current_phase_id')
            if phase_id == 0:
                phase_feature_list[0] = 1
            else:
                phase_feature_list[phase_id - 1] = 1
            
            state['phase'] = torch.tensor(phase_feature_list, dtype=torch.float32)
            
            # roads
            state['roads'] = {f"road_{road_id}": {} for road_id in range(1, self.num_roads + 1)}
            for road_id in range(1, self.num_roads + 1):
                road = self.roads[road_id]
                state['roads'][f"road_{road_id}"]['road'] = torch.tensor([
                    road.get('max_queue_length'),
                    road.get('average_delay'),
                ], dtype=torch.float32)

            # lanes
            for road_id in range(1, self.num_roads + 1):
                lanes = self.road_lanes_map[road_id]
                state['roads'][f"road_{road_id}"]['lanes'] = {f"lane_{lane_id}": {} for lane_id in range(1, self.num_lanes_map[road_id] + 1 )}
                
                for lane_id in range(1, lanes.count() + 1):
                    lane = lanes[lane_id]
                    lane_features = [lane.get('num_vehicles'), lane.get('length_info')['length']]

                    if lane.link.get('type') == 'left':
                        lane_features.extend([1, 0, 0])       
                    elif lane.link.get('type') == 'main':
                        lane_features.extend([0, 1, 0])
                    elif lane.link.get('type') == 'right':   
                        lane_features.extend([0, 0, 1])
                    else:
                        raise NotImplementedError(f"Not supported lane type: {lane.link.get('type')}")
                    
                    state['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['lane'] = torch.tensor(lane_features, dtype=torch.float32)
                    
            # vehicles 
            for road_id in range(1, self.num_roads + 1):
                lanes = self.road_lanes_map[road_id]
                for lane_id in range(1, lanes.count() + 1):
                    vehicles_df = self.vehicles_df_map[road_id, lane_id]

                    vehicle_features_list = []
                    for vehicle_id, vehicle_row in vehicles_df.iterrows():
                        if vehicle_id >= self.num_vehicles:
                            break

                        vehicle_features = [0] * self.num_roads
                        vehicle_features[int(vehicle_row['direction_id'])] = 1
                        vehicle_features.extend([
                            vehicle_row['position'],
                            vehicle_row['speed'],
                            1,
                        ])
                        vehicle_features_list.append(vehicle_features)
                    
                    if len(vehicle_features_list) < self.num_vehicles:
                        vehicle_features_list.extend([[0] * (self.num_roads + 3)] * (self.num_vehicles - len(vehicle_features_list)))
                    
                    state['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['vehicles'] = torch.tensor(vehicle_features_list, dtype=torch.float32)
     
        state = self._toDevice(state)
        state = self._unsqueeze(state)
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
                action_values = self.model(self.current_state)
                action = torch.argmax(action_values).item() + 1
            
            end_time = time.time()
            calc_time = end_time - start_time
            self.calc_time_record_list.append({'time': self.current_time, 'calc_time': calc_time})

        # 記録
        self.current_action = action
        self.action_record.append(action)
        self.signal_controller.setPhases([self.current_action] * self.duration_steps)
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
                    vehicles_df = lane.get('vehicles_df')
                    for _, vehicle_row in vehicles_df.iterrows():
                        direction_id = vehicle_row['direction_id']
                        if direction_id == 0:
                            continue
                        signal_num_vehs_map[(road_order_id - 1) * (self.num_roads - 1) + direction_id] += 1

            phase_num_vehs_map = {}
            for phase_id, phase_list in self.phase_map.items():
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
            for _, vehicles_df in self.vehicles_df_map.items():
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
            for lane_str, vehicles_df in self.lane_str_vehicles_df_map.items():
                if vehicles_df.shape[0] == 0:
                    continue

                road_order_id, _ = map(int, lane_str.split('-'))
                road = self.roads[road_order_id]
                v_max = road.get('max_speed')

                for _, row in vehicles_df.iterrows():
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
            for lane_str, vehicles_df in self.lane_str_vehicles_df_map.items():
                if vehicles_df.shape[0] == 0:
                    continue

                road_order_id, _ = map(int, lane_str.split('-'))
                road = self.roads[road_order_id]
                v_max = road.get('max_speed')

                for _, row in vehicles_df.iterrows():
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

        # calculate cumulative reward
        cumulative_reward = 0
        for reward in list(reversed(self.reward_record)):
            cumulative_reward = reward + self.gamma * cumulative_reward
        
        # get learning_data
        learning_data = {
            'state': self.state_record[0],
            'action': self.action_record[0],
            'cumulative_reward': cumulative_reward,
            'next_state': self.state_record[-1],
            'done': int(self.done_flg),
        }

        # update learning_data_list
        self.learning_data_list.append(learning_data)

        # data augmentation
        if self.data_augmentation_flg:
            self._runDataAugmentation(learning_data)
        return
    
    # データ拡張をするメソッド
    def _runDataAugmentation(self, learning_data):
        if self.data_augmentation_type == 0:
            return
         
        if self.num_roads == 4:
            if self.data_augmentation_type == 1:
                symmetry_types = range(1, self.num_roads)
            elif self.data_augmentation_type == 2:
                symmetry_types = [2]
            else:
                raise NotImplementedError(f"Not supported data_augmentation_type: {self.data_augmentation_type}")
            
            for symmetry_type in symmetry_types:
                rotated_learning_data = {
                    'state': self._rotateState(learning_data['state'], symmetry_type),
                    'action': learning_data['action'],
                    'cumulative_reward': learning_data['cumulative_reward'],
                    'next_state': self._rotateState(learning_data['next_state'], symmetry_type),
                    'done': learning_data['done'],
                }

                self.learning_data_list.append(rotated_learning_data)
        else:
            raise NotImplementedError(f"Not supported number of roads: {self.num_roads}")
        return
    
    # 状態を回転させるメソッド（symmetry_type = 1: 90度，2:180度，3:270度）
    def _rotateState(self, state, symmetry_type):
        rotated_state = {}

        # set rotated road features
        rotated_state['roads'] = {}
        for road_order_id in range(1, self.num_roads + 1):
            new_road_order_id = (road_order_id + symmetry_type - 1) % self.num_roads + 1
            rotated_state['roads'][f"road_{new_road_order_id}"] = state['roads'][f"road_{road_order_id}"]        

        # get phase_id
        for id, flg in enumerate(state['phase'].squeeze(0).tolist()):
            if flg == 0:
                continue
            phase_id = id + 1    
            break

        # get symmetry_phase_id
        symmetry_phase_id = self.symmetry_phase_map[phase_id][symmetry_type]

        # set rotated phase features
        phase_state = [0] * (self.intersection.get('num_phases'))
        phase_state[symmetry_phase_id - 1] = 1
        rotated_state['phase'] = torch.tensor(phase_state, dtype=torch.float32)

        return rotated_state
    
    def syncDataFrame(self):
        self.calc_time_record_df = pd.DataFrame(self.calc_time_record_list)
        return
    
    def _toDevice(self, data):
        if isinstance(data, dict):
            return {key: self._toDevice(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._toDevice(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data
    
    def _unsqueeze(self, data):
        if isinstance(data, dict):
            return {key: self._unsqueeze(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._unsqueeze(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.unsqueeze(0)
        else:
            return data

    
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
    

