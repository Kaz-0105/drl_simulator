from libs.container import Container
from libs.object import Object
from libs.torch_module import ExtendedModule
from objects.neural_networks.apex.proto_q_net import ProtoQNet


import torch
import random
from collections import deque
import pandas as pd
import time
import copy
import numpy as np

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
    
    @property
    def simulation_id(self):
        if hasattr(self, 'network'):
            return self.network.simulation.get('id')
        elif hasattr(self, 'master_agent'):
            return self.master_agent.get('simulation_id')
        else:
            raise NotImplementedError("Not supported case.")
    
    def _initProps(self):
        # set simulation_type
        drl_info = self.config.get('drl_info')
        self.simulation_type = drl_info['simulation_type']

        # set epsilons_df
        if self.simulation_type == 'train':
            self.epsilons_df = self.config.get('epsilons_df').copy()
            if (self.simulation_id - 1) % len(self.epsilons_df) != 0:
                self.epsilons_df = pd.concat([
                    self.epsilons_df.iloc[(self.simulation_id - 1) % len(self.epsilons_df):],
                    self.epsilons_df.iloc[:(self.simulation_id - 1) % len(self.epsilons_df)]
                ], ignore_index=True)  
        return
    
    def _initElements(self):
        for intersection in self.network.intersections.getAll(sorted_flg=True):
            if self.simulation_type == 'train':
                epsilon = float(self.epsilons_df.iloc[intersection.get('id') - 1]['epsilon'])
            elif self.simulation_type == 'test':
                epsilon = 0.0
            else:
                raise NotImplementedError(f"Not supported simulation_type: {self.simulation_type}")
            
            self.add(LocalAgent(
                local_agents=self, 
                intersection=intersection, 
                epsilon=epsilon
            ))
        return
    
    def update(self, type):
        for agent in self.getAll():
            self.executor.submit(agent.update, type)
        self.executor.wait()
        return
    
    def sync(self, type):
        for agent in self.getAll():
            self.executor.submit(agent.sync, type)
        self.executor.wait()

        for agent in self.getAll():
            if not agent.get('sync_flg'):
                continue
            agent.showInfo('sync')
            agent.set('sync_flg', False)    

        return
    
    def showInfo(self, type):
        if type == 'action_result':
            if not any(agent.get('infer_flg') for agent in self.getAll()):
                return

            print('==============================================')
            print(f"status: action and reward results")
            for agent in self.getAll():
                if not agent.get('infer_flg'):
                    continue
                
                if agent.get('current_random_action_flg'):
                    print(f"local_agent {agent.get('id')}: action = {agent.get('current_action')} (r), reward = {agent.get('current_reward'):.1f}")
                else:
                    print(f"local_agent {agent.get('id')}: action = {agent.get('current_action')}, reward = {agent.get('current_reward'):.1f}")
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return
    
    @property
    def done_flg(self):
        for agent in self.getAll():
            if agent.get('done_flg'):
                return True
        return False
    
class LocalAgent(Object):
    # signal color
    RED = 1
    GREEN = 3

    # average spacing and max delay
    INITIAL_SPACING = 6
    VEHICLE_DISTANCE = 1
    MAX_DELAY = 120


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

        # set properties
        self._initProps(epsilon)

        # initialize model
        self._makeModel()
        self._syncModel()
        return
    
    @property
    def current_time(self):
        return self.network.simulation.get('current_time')
    
    @property
    def current_state(self):
        return self._toDevice(self.state_record[-1]) if len(self.state_record) > 0 else None

    @property
    def current_action(self):
        return self.action_record[-1] if len(self.action_record) > 0 else None
    
    @property
    def current_random_action_flg(self):
        return self.random_action_flg_record[-1] if len(self.random_action_flg_record) > 0 else None
    
    @property
    def current_reward(self):
        return self.reward_record[-1] if len(self.reward_record) > 0 else None

    @property
    def num_learning_data(self):
        return len(self.learning_data_list)
    
    @property
    def infer_flg(self):
        future_phase_ids = self.signal_controller.get('future_phase_ids')
        return len(future_phase_ids) <= 1
    
    def _initProps(self, epsilon):
        # set num_roads and num_max_phases
        self.num_roads = self.master_agent.get('num_roads')
        self.num_max_phases = self.master_agent.get('num_max_phases')

        self.num_lanes_map = self.master_agent.get('num_lanes_map') 
        self.symmetry_phase_map = self.master_agent.get('symmetry_phase_map')
        self.active_phase_list = self.master_agent.get('active_phase_list')
        self.phase_map = self.signal_controller.get('phase_map')

        # set epsilon
        self.epsilon = epsilon

        # set other drl information
        drl_info = self.config.get('drl_info')
        self.duration_steps = drl_info['duration_steps']
        self.td_steps = drl_info['framework']['apex']['td_steps']
        self.update_interval = drl_info['framework']['apex']['local_agent']['update_interval']
        self.architecture = drl_info['architecture']['type']

        # state information
        self.num_vehicles = drl_info['state']['vehicle']['number']
        self.state_info = copy.deepcopy(drl_info['state'])
        del self.state_info['vehicle']['number']

        # action information
        self.random_action_type = drl_info['action']['random']['type']
        if self.random_action_type == 'top_k':
            self.num_top_k_actions = drl_info['action']['random']['top_k']['num_actions']

        # reward information
        self.reward_type = drl_info['reward']['type']
        self.gamma = float(drl_info['reward']['common']['gamma'])
        
        if self.reward_type == 'waiting_vehicles':
            waiting_vehicles_info = drl_info['reward']['waiting_vehicles']
            self.bonus = waiting_vehicles_info['bonus']
            self.weight_flg = (waiting_vehicles_info['weight']['type'] is not None)
            if self.weight_flg:
                self.weight_type = waiting_vehicles_info['weight']['type']
                if self.weight_type == 'exponential':
                    self.half_life = waiting_vehicles_info['weight'][self.weight_type]['half']
                elif self.weight_type == 'queue_exponential':
                    pass
                else:
                    raise NotImplementedError(f"Not supported weight_type: {self.weight_type}")

            self.reward_scaling_flg = (waiting_vehicles_info['scaler']['type'] is not None)
            if self.reward_scaling_flg:
                self.reward_scaling_function = WaitingVehiclesRewardScaler(self)
        
        elif self.reward_type == 'speedy_vehicles':
            speedy_vehicles_info = drl_info['reward']['speedy_vehicles']
            self.speed_threshold = speedy_vehicles_info['threshold']
        
        elif self.reward_type == 'throughput':
            throughput_info = drl_info['reward']['throughput']
            self.baseline = (throughput_info['baseline'] * (1 / 3600) * self.network.simulation.get('time_step')) / 2 # the unit is veh/step
        else: 
            raise NotImplementedError(f"Not supported reward_type: {self.reward_type}")
        
        # set data augmentation information
        self.data_augmentation_flg = drl_info['data_augmentation']['flg']
        if self.data_augmentation_flg:
            if self.num_roads == 4:
                if len(set(self.num_lanes_map.values())) == 1:
                    self.data_augmentation_type = 1
                elif self.num_lanes_map[1] == self.num_lanes_map[3] and self.num_lanes_map[2] == self.num_lanes_map[4]:
                    self.data_augmentation_type = 2
                else:
                    self.data_augmentation_type = 0
            else:
                raise NotImplementedError(f"Not supported number of roads: {self.num_roads}")

        # set length_info_map
        self.length_info_map = {}
        self.branch_pos_map = {}
        for road_id, road in self.roads.items():
            for link in road.links.getAll(): 
                if link.get('type') == 'main':
                    self.length_info_map[road.get('id'), link.get('id')] = {
                        'length': link.get('length'),
                        'start_pos': 0,
                        'signal_pos': link.get('length')
                    }
                elif link.get('type') == 'connector':
                    next_link = link.to_links.getAll()[0]
                    self.length_info_map[road.get('id'), link.get('id')] = {
                        'length': link.get('length'),
                        'start_pos': link.get('from_pos'),
                        'signal_pos': link.get('from_pos') + link.get('length') - link.get('to_pos') + next_link.get('length') 
                    }
                    self.branch_pos_map[road_id] = link.get('from_pos')
                elif link.get('type') in ['right', 'left']:
                    from_connector = link.from_links.getAll()[0]
                    self.length_info_map[road.get('id'), link.get('id')] = {
                        'length': link.get('length'),
                        'start_pos': from_connector.get('from_pos') + from_connector.get('length') - from_connector.get('to_pos'),
                        'signal_pos': from_connector.get('from_pos') + from_connector.get('length') - from_connector.get('to_pos') + link.get('length'),
                    }
                else:
                    raise NotImplementedError(f"Not supported link type: {link.get('type')}")
                
        # set lanes_map
        self.lanes_map = {road_id: {} for road_id in range(1, self.num_roads + 1)}
        for road_id, road in self.roads.items():
            counter = 1

            # right branching lane
            for link in road.links.getAll():
                if link.get('type') != 'right':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    self.lanes_map[road_id][counter] = lane
                    counter += 1
            
            # main lane
            for link in road.links.getAll():
                if link.get('type') != 'main':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    self.lanes_map[road_id][counter] = lane
                    counter += 1
            
            # left branching lane
            for link in road.links.getAll():
                if link.get('type') != 'left':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    self.lanes_map[road_id][counter] = lane
                    counter += 1

        # initialize other properties
        self.state_record = deque(maxlen=self.td_steps + 1)
        self.action_record = deque(maxlen=self.td_steps)
        self.random_action_flg_record = deque(maxlen=self.td_steps)
        self.reward_record = deque(maxlen=self.td_steps)
        self.calc_time_record_list = []

        self.spacing = LocalAgent.INITIAL_SPACING
        self.done_flg = False
        self.learning_data_list = []
        self.total_reward = 0
        self.num_model_runs = 0
        self.sync_flg = False
        return
        
    def _makeModel(self):
        if self.architecture == 'proto':
            self.model = ProtoQNet(self)
        else:
            raise NotImplementedError(f"Not supported architecture: {self.architecture}")

        self.model.eval()
        self.model.to(self.device)
        return
    
    def _syncModel(self):
        master_agent_model = self.master_agent.get('model')
        self.model.load_state_dict(master_agent_model.state_dict())
        return
    
    def _getVehiclesDf(self):
        self.vehicles_df_map = {}
        for road_id in range(1, self.num_roads + 1):
            road = self.roads[road_id]
            vehicles_df = road.get('vehicles_df').copy()

            if vehicles_df.shape[0] == 0:
                column_list = ['id', 'position_1', 'position_2', 'length', 'speed', 'lane_id', 'link_id', 'route_id', 'wait_link_id', 'wait_lane_id']
                if self.reward_type == 'waiting_vehicles':
                    column_list.append('wait_flg')

                for lane_id in self.lanes_map[road_id].keys():
                    self.vehicles_df_map[road_id, lane_id] = pd.DataFrame(columns=column_list)

                continue

            # remove in_queue and road_id columns
            vehicles_df = vehicles_df.drop(columns=['in_queue', 'road_id'])
            
            # add position_1 column (def: the distance from the start of the link)
            position_1_list = []
            for _, vehicle_row in vehicles_df.iterrows():
                length_info = self.length_info_map[road.get('id'), int(vehicle_row['link_id'])]
                position_1_list.append(vehicle_row['position'] + length_info['start_pos'])
            
            vehicles_df['position_1'] = position_1_list
            vehicles_df = vehicles_df.sort_values(by='position_1', ascending=False)
            vehicles_df = vehicles_df.reset_index(drop=True)

            # remove position column
            vehicles_df = vehicles_df.drop(columns=['position'])

            # remove vehicle which has no route information (def: route_id = 0 and next_link_id = np.nan)
            vehicles_df = vehicles_df[~(vehicles_df['route_id'].astype(int) == 0)].copy().reset_index(drop=True) 

            # add wait_link_id and wait_lane_id columns
            wait_link_id_list = []
            wait_lane_id_list = []
            for _, vehicle_row in vehicles_df.iterrows():
                if int(vehicle_row['next_link_id']) not in road.links.getKeys():
                    wait_link_id_list.append(int(vehicle_row['link_id']))
                    wait_lane_id_list.append(int(vehicle_row['lane_id']))
                    continue

                next_link = road.links[int(vehicle_row['next_link_id'])]
                if next_link.get('type') == 'connector':
                    wait_link = next_link.to_links.getAll()[0]
                    wait_lane = next_link.to_lane
                elif next_link.get('type') in ['right', 'left']:
                    wait_link = next_link
                    wait_lane = road.links[int(vehicle_row['link_id'])].to_lane
                else:
                    raise NotImplementedError(f"Not supported link type: {next_link.get('type')}")
                
                wait_link_id_list.append(int(wait_link.get('id')))
                wait_lane_id_list.append(int(wait_lane.get('id')))
                
            vehicles_df['wait_link_id'] = wait_link_id_list
            vehicles_df['wait_lane_id'] = wait_lane_id_list

            # remove next_link_id column
            vehicles_df = vehicles_df.drop(columns=['next_link_id'])

            # add position_2 column (def: the distance from the traffic signal)
            position_2_list = []
            for _, vehicle_row in vehicles_df.iterrows():
                length_info = self.length_info_map[road.get('id'), int(vehicle_row['wait_link_id'])]
                position_2_list.append(length_info['signal_pos'] - vehicle_row['position_1'])
        
            vehicles_df['position_2'] = position_2_list

            if self.reward_type == 'waiting_vehicles':
                # get near_flg_list
                near_flg_list = []
                close_threshold = road.get('close_threshold')
                for _, vehicle_row in vehicles_df.iterrows():
                    near_flg_list.append(vehicle_row['position_2'] <= close_threshold)

                # get red_flg_list
                red_flg_list = []
                route_signal_color_map = road.get('route_signal_color_map')
                for _, vehicle_row in vehicles_df.iterrows():
                    if int(vehicle_row['route_id']) == 0:
                        red_flg_list.append(True)
                        continue

                    signal_color = route_signal_color_map[int(vehicle_row['route_id'])]
                    red_flg_list.append(signal_color == self.RED)

                if road_id in self.branch_pos_map:
                    # get branch_flg_list
                    branch_flg_list = []
                    for _, vehicle_row in vehicles_df.iterrows():
                        branch_flg_list.append(vehicle_row['position_1'] >= self.branch_pos_map[road_id])

                    # initialize space_map and last_vehs_map
                    last_vehs_map = {}
                    space_map = {}
                    for lane in self.lanes_map[road_id].values():
                        link = lane.link
                        space_map[link.get('id'), lane.get('id')] = self.length_info_map[road.get('id'), link.get('id')]['signal_pos'] - self.branch_pos_map[road_id]
                    
                    
                    # get wait_flg_list
                    wait_flg_list = []
                    for vehicle_id, vehicle_row in vehicles_df.iterrows():
                        # update space_map and last_vehs_map
                        space_map[int(vehicle_row['wait_link_id']), int(vehicle_row['wait_lane_id'])] -= (vehicle_row['length'] + LocalAgent.VEHICLE_DISTANCE)
                        last_vehs_map[int(vehicle_row['wait_link_id']), int(vehicle_row['wait_lane_id']), int(vehicle_row['route_id'])] = {
                            'id': vehicle_id,
                            'position_2': vehicle_row['position_2']
                        }

                        if not near_flg_list[vehicle_id]:
                            wait_flg_list.append(False)
                            continue

                        if red_flg_list[vehicle_id]:
                            wait_flg_list.append(True)
                            continue
                        
                        target_veh_info = None
                        for (wait_link_id, wait_lane_id, route_id), vehicle_info in last_vehs_map.items():
                            if not (wait_link_id == int(vehicle_row['wait_link_id']) and wait_lane_id == int(vehicle_row['wait_lane_id'])):
                                if branch_flg_list[vehicle_id]:
                                    continue

                                if space_map[int(vehicle_row['wait_link_id']), int(vehicle_row['wait_lane_id'])] >= 0:
                                    continue

                            if route_id == int(vehicle_row['route_id']):
                                continue

                            if target_veh_info is None:
                                target_veh_info = vehicle_info
                                continue

                            if vehicle_info['position_2'] > target_veh_info['position_2']:
                                target_veh_info = vehicle_info
                    
                        if target_veh_info is not None:
                            wait_flg_list.append(wait_flg_list[target_veh_info['id']])
                        else:
                            wait_flg_list.append(False)

                    vehicles_df['wait_flg'] = wait_flg_list
                else:
                    # get wait_flg_list
                    wait_flg_list = []
                    last_vehs_map = {}
                    for vehicle_id, vehicle_row in vehicles_df.iterrows():
                        last_vehs_map[int(vehicle_row['route_id'])] = {
                            'id': vehicle_id,
                            'position_2': vehicle_row['position_2']
                        }

                        if not near_flg_list[vehicle_id]:
                            wait_flg_list.append(False)
                            continue

                        if red_flg_list[vehicle_id]:
                            wait_flg_list.append(True)
                            continue

                        target_veh_info = None
                        for route_id, vehicle_info in last_vehs_map.items():
                            if route_id == int(vehicle_row['route_id']):
                                continue

                            if target_veh_info is None:
                                target_veh_info = vehicle_info
                                continue

                            if vehicle_info['position_2'] > target_veh_info['position_2']:
                                target_veh_info = vehicle_info
                    
                        if target_veh_info is not None:
                            wait_flg_list.append(wait_flg_list[target_veh_info['id']])
                        else:
                            wait_flg_list.append(False)
                        
                    vehicles_df['wait_flg'] = wait_flg_list
            else:
                raise NotImplementedError(f"Not supported reward_type: {self.reward_type}")

            for lane_id, lane in self.lanes_map[road_id].items():
                self.vehicles_df_map[road_id, lane_id] = vehicles_df[
                    (vehicles_df['link_id'] == lane.link.get('id')) & 
                    (vehicles_df['lane_id'] == lane.get('id'))
                ].copy().reset_index(drop=True)
        return

    def _updateSpacing(self):
        veh_length_list = []
        for _, vehicles_df in self.vehicles_df_map.items():
            if vehicles_df.shape[0] == 0:
                continue

            veh_length_list.extend(vehicles_df['length'].tolist())
        
        if len(veh_length_list) == 0:
            return
        
        avg_veh_length = sum(veh_length_list) / len(veh_length_list)
        self.spacing = avg_veh_length + LocalAgent.VEHICLE_DISTANCE
        return 
    
    def _getState(self):
        state = {}

        # intersection
        intersection_feature_list = [0] * self.num_max_phases   
        
        phase_id = self.intersection.get('current_phase_id')
        if phase_id == 0:
            intersection_feature_list[0] = 1
        else:
            intersection_feature_list[phase_id - 1] = 1
        
        state['intersection'] = torch.tensor(intersection_feature_list, dtype=torch.float32)
        
        # roads
        state['roads'] = {f"road_{road_id}": {} for road_id in range(1, self.num_roads + 1)}
        for road_id, road in self.roads.items(sorted_flg=True):
            road_features = [
                road.get('max_queue_length') / road.get('length'),
                road.get('average_delay') / LocalAgent.MAX_DELAY,
            ]
            turn_ratio_list = list(road.get('turn_ratios').values())
            road_features.extend([turn_ratio / sum(turn_ratio_list) for turn_ratio in turn_ratio_list])
            state['roads'][f"road_{road_id}"]['road'] = torch.tensor(road_features, dtype=torch.float32)

        # lanes
        for road_id, road in self.roads.items(sorted_flg=True):
            state['roads'][f"road_{road_id}"]['lanes'] = {f"lane_{lane_id}": {} for lane_id in range(1, self.num_lanes_map[road_id] + 1 )}
            
            for lane_id, lane in self.lanes_map[road_id].items():
                lane_features = [
                    lane.get('num_vehicles') / (lane.get('length') / self.spacing),
                    lane.get('length') / self.roads.get('max_length'),
                ]

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
        for road_id, road in self.roads.items(sorted_flg=True):
            for lane_id in self.lanes_map[road_id].keys():
                vehicles_df = self.vehicles_df_map[road_id, lane_id]

                vehicle_features_list = []
                for vehicle_id, vehicle_row in vehicles_df.iterrows():
                    if vehicle_id >= self.num_vehicles:
                        break

                    vehicle_features = [0] * self.num_roads
                    vehicle_features[int(vehicle_row['route_id'])] = 1
                    vehicle_features.extend([
                        vehicle_row['position_2'] / self.length_info_map[road.get('id'), int(vehicle_row['wait_link_id'])]['signal_pos'],
                        vehicle_row['speed'] / road.get('max_speed'),
                        1,
                    ])
                    vehicle_features_list.append(vehicle_features)
                
                if len(vehicle_features_list) < self.num_vehicles:
                    vehicle_features_list.extend([[0] * (self.num_roads + 3)] * (self.num_vehicles - len(vehicle_features_list)))
                
                state['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['vehicles'] = torch.tensor(vehicle_features_list, dtype=torch.float32)
     
        state = self._unsqueeze(state)
        self.state_record.append(state)
        return

    def _getAction(self):
        if random.random() < self.epsilon:
            action = self._getRandomAction()
            random_action_flg = True
        else:
            # get action and calc_time
            start_time = time.time()

            with torch.no_grad():
                action_values = self.model(self.current_state)
            action = torch.argmax(action_values).item() + 1

            end_time = time.time()
            self.calc_time_record_list.append({'time': self.current_time, 'calc_time': end_time - start_time})

            self.num_model_runs += 1

            # get random_action_flg
            random_action_flg = False

        # save action and set signal phase
        self.action_record.append(action)
        self.random_action_flg_record.append(random_action_flg)
        self.signal_controller.setPhases([self.current_action] * self.duration_steps)
        return
    
    def _getRandomAction(self):
        if self.random_action_type == 'normal':
            return random.choice(self.active_phase_list)

        elif self.random_action_type == 'top_k':
            with torch.no_grad():
                action_values = self.model(self.current_state)
            top_k_phase_list = (torch.topk(action_values, self.num_top_k_actions).indices.flatten() + 1).tolist()
            valid_top_k_phase_list = list(set(top_k_phase_list) & set(self.active_phase_list))

            return random.choice(valid_top_k_phase_list) if len(valid_top_k_phase_list) > 0 else random.choice(self.active_phase_list)
        
        elif self.random_action_type == 'num_vehs':
            signal_num_vehs_map = {signal_id: 0 for signal_id in range(1, self.num_roads * (self.num_roads - 1) + 1)}
            for (road_id, _), vehicles_df in self.vehicles_df_map.items():
                for _, vehicle_row in vehicles_df.iterrows():
                    if int(vehicle_row['route_id']) == 0:
                        continue

                    signal_num_vehs_map[(road_id - 1) * (self.num_roads - 1) + int(vehicle_row['route_id'])] += 1

            phase_num_vehs_map = {}
            for phase_id, phase_list in self.phase_map.items():
                if phase_id not in self.active_phase_list:
                    continue

                tmp_num_vehs = 0
                for signal_id in phase_list:
                    tmp_num_vehs += signal_num_vehs_map[signal_id]
                phase_num_vehs_map[phase_id] = tmp_num_vehs
            
            if sum(phase_num_vehs_map.values()) == 0:
                return random.choice(self.active_phase_list)
            else:
                return random.choices(
                    list(phase_num_vehs_map),
                    weights=phase_num_vehs_map.values(),
                    k=1
                )[0]
        
        else: 
            raise NotImplementedError(f"Not supported random_action_type: {self.random_action_type}")

    def _getReward(self):
        if self.reward_type == 'waiting_vehicles':
            reward = self._getWaitingVehiclesReward()
                                                
        elif self.reward_type == 'speedy_vehicles':
            reward = self._getSpeedyVehiclesReward()

        elif self.reward_type == 'space':
            reward = self._getSpaceReward()
        
        elif self.reward_type == 'throughput':
            reward = self._getThroughputReward()

        self.reward_record.append(reward)
        self.total_reward += reward
        return
    
    def _getWaitingVehiclesReward(self):
        # (the number of not-waiting vehicles) - (the number of waiting vehicles) + (the number of passing vehicles)
        reward = 0
        num_vehs = 0

        # the number of waiting and not-waiting vehicles
        if self.weight_flg:
            for (road_id, _), vehicles_df in self.vehicles_df_map.items():
                if vehicles_df.shape[0] == 0:
                    continue

                weight_array = self._getWaitingVehicleWeights(vehicles_df, road_id)

                num_vehs += weight_array.sum()
                reward += np.dot(
                    np.where(vehicles_df['wait_flg'].to_numpy(), -1.0, 1.0),
                    weight_array
                )

        else:
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
                    reward += num_pass_vehs * (1 + self.bonus)
                    num_vehs += num_pass_vehs
        
        # normalize the reward
        reward = reward / num_vehs if num_vehs > 0 else 0

        if not self.reward_scaling_flg:
            return reward
        
        reward = self.reward_scaling_function(reward).item()
        return reward
    
    def _getWaitingVehicleWeights(self, vehicles_df, road_id):
        if self.weight_type == 'exponential':
            decay_constant = np.log(2) / self.half_life
            return np.exp(-decay_constant * vehicles_df['position_2'].to_numpy())
        elif self.weight_type == 'queue_exponential':
            decay_constant = np.log(2) / self.roads[road_id].get('close_threshold')
            return np.exp(-decay_constant * vehicles_df['position_2'].to_numpy())
        else:
            raise NotImplementedError(f"Not supported weight_type: {self.weight_type}")
    
    def _getSpeedyVehiclesReward(self):
        reward = 0
        num_vehs = 0
        for (road_id, _), vehicles_df in self.vehicles_df_map.items():
            if vehicles_df.shape[0] == 0:
                continue

            road = self.roads[road_id]
            v_max = road.get('max_speed')

            for _, vehicle_row in vehicles_df.iterrows():
                if vehicle_row['speed'] <= self.speed_threshold * v_max:
                    reward -= 1
                else:
                    reward += 1
            
            num_vehs += vehicles_df.shape[0]

        for road in self.roads.getAll():
            for data_collection_point in road.data_collection_points.getAll():
                if data_collection_point.get('type') != 'intersection':
                    continue

                for data_collection_measurement in data_collection_point.data_collection_measurements.getAll():
                    if data_collection_measurement.get('type') == 'multiple':
                        continue

                num_vehs_record = data_collection_measurement.get('num_vehs_record')
                num_vehs_list = num_vehs_record['num_vehs'].tail(self.duration_steps).tolist()
                reward += sum(num_vehs_list)
                num_vehs += sum(num_vehs_list)
        
        reward = reward / num_vehs if num_vehs > 0 else 0

        return reward
    
    def _getSpaceReward(self):
        reward = 0
        for road_order_id in range(1, self.num_roads + 1):
            road = self.roads[road_order_id]
            space = ((road.get('length') - self.road_max_queue_map[road_order_id]) / road.get('length')) * 10 - 5  # -5〜5に正規化
            reward += space

        return reward
    
    def _getThroughputReward(self):
        reward = 0
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

        # change the unit to veh/step
        reward = reward / self.duration_steps

        # # normalize the reward by baseline and shift to around -1 to 1
        # reward = reward / self.baseline - 1

        return reward
    
    def _updateLearningDataList(self):
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
            'done_flg': int(self.done_flg),
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
                symmetry_types = list(range(1, self.num_roads))
            elif self.data_augmentation_type == 2:
                symmetry_types = [2]
            else:
                raise NotImplementedError(f"Not supported data_augmentation_type: {self.data_augmentation_type}")
            
            for symmetry_type in symmetry_types:
                rotated_learning_data = {
                    'state': self._rotateState(learning_data['state'], symmetry_type),
                    'action': self.symmetry_phase_map[learning_data['action']][symmetry_type],
                    'cumulative_reward': learning_data['cumulative_reward'],
                    'next_state': self._rotateState(learning_data['next_state'], symmetry_type),
                    'done_flg': learning_data['done_flg'],
                }

                self.learning_data_list.append(rotated_learning_data)
        else:
            raise NotImplementedError(f"Not supported number of roads: {self.num_roads}")
        return
    
    def _rotateState(self, state, symmetry_type):
        rotated_state = {}

        # set rotated road features
        rotated_state['roads'] = {}
        for road_order_id in range(1, self.num_roads + 1):
            new_road_order_id = (road_order_id + symmetry_type - 1) % self.num_roads + 1
            rotated_state['roads'][f"road_{new_road_order_id}"] = state['roads'][f"road_{road_order_id}"]        

        # get phase_id
        for id, flg in enumerate(state['intersection'].squeeze(0).tolist()):
            if flg == 0:
                continue
            phase_id = id + 1    
            break

        # get symmetry_phase_id
        symmetry_phase_id = self.symmetry_phase_map[phase_id][symmetry_type]

        # set rotated intersection features
        intersection_state = [0] * (self.num_max_phases)
        intersection_state[symmetry_phase_id - 1] = 1
        rotated_state['intersection'] = torch.tensor(intersection_state, dtype=torch.float32)

        return rotated_state
    
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
        
    def _showInfo(self, type):
        print('==============================================')
        if type == 'sync':
            print(f"status: syncronized local agent to master agent")
            print(f"local agent id: {self.id}")
            print(f"master agent id: {self.master_agent.get('id')}")
            print(f"update interval: {self.update_interval}")
        else:
            raise NotImplementedError(f"Not supported type: {type}")


    def update(self, type):
        if not self.infer_flg:
            return
        
        if type == 'initial_state':
            self._getVehiclesDf()
            self._updateSpacing()
            self._getState()

        elif type == 'action':
            self._getAction()

        elif type == 'state':
            self._getVehiclesDf()
            self._getState()
            self._getReward()
            self._updateLearningDataList()
            
        else:
            raise NotImplementedError(f"Not supported type: {type}")
        
    def sync(self, type):
        if type == 'model':
            if self.num_model_runs < self.update_interval:
                return
            
            self._syncModel()
            self.sync_flg = True
            self.num_model_runs = 0 

        elif type == 'dataframe':
            self.calc_time_record_df = pd.DataFrame(self.calc_time_record_list)

        else:
            raise NotImplementedError(f"Not supported type: {type}")
        
        return
    
    def showInfo(self, type):
        self._showInfo(type)
        return
    
class WaitingVehiclesRewardScaler(ExtendedModule):
    def __init__(self, local_agent):
        super().__init__()

        self.local_agent = local_agent
        self.config = local_agent.config

        self._initProps()
        self.eval()
        return
    
    def _initProps(self):
        # set num_roads
        self.num_roads = self.local_agent.get('num_roads')

        # set scaler information
        drl_info = self.config.get('drl_info')
        waiting_vehicles_info = drl_info['reward']['waiting_vehicles']

        self.scale_type = waiting_vehicles_info['scaler']['type']
        if self.scale_type == 'tanh':
            self.alpha = waiting_vehicles_info['scaler']['tanh']['alpha']
            self.center = (3 - self.num_roads) / (self.num_roads - 1)

        elif self.scale_type == 'tanh_linear':
            self.alpha = waiting_vehicles_info['scaler']['tanh_linear']['alpha']
            self.center = (3 - self.num_roads) / (self.num_roads - 1)
    
        else:
            raise NotImplementedError(f"Not supported reward_scaling_type: {self.scale_type}")
        
        return

    def forward(self, reward):
        if self.scale_type == 'tanh':
            reward = torch.tanh(torch.tensor(self.alpha * (reward - self.center), dtype=torch.float32))
        elif self.scale_type == 'tanh_linear':
            if reward < self.center:
                reward = torch.tanh(torch.tensor(self.alpha * (reward - self.center), dtype=torch.float32))
            else:
                reward = self.alpha * (reward - self.center)
                reward = torch.tensor(reward, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Not supported reward_scaling_type: {self.scale_type}")
        
        return  reward
    

