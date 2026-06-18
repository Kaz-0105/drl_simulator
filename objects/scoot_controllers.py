from libs.container import Container
from libs.object import Object

from collections import deque
import copy
import pandas as pd

class ScootControllers(Container):
    def __init__(self, network):
        super().__init__()

        self.config = network.config
        self.executor = network.executor
        self.shared_resources = network.shared_resources
        self.network = network

        self._initProps()
        self._initElements()
        return
    
    def _initProps(self):
        self.max_saturation_map = None
        return
    
    def _initElements(self):
        # set each scoot_controller
        for intersection_id, intersection in self.network.intersections.items(sorted_flg=True):
            scoot_controller = ScootController(
                scoot_controllers=self, 
                intersection=intersection,
                id=intersection_id
            )
            self.add(scoot_controller)

        # set max_outflow_rate_map
        self.max_outflow_rate_map = {}

        for scoot_controller in self.getAll():
            num_roads = scoot_controller.get('num_roads')

            for road in scoot_controller.roads.getAll():
                for route_id in range(1, num_roads):
                    self.max_outflow_rate_map[(num_roads, road.get('type'), route_id, road.get('max_speed'))] = {}
        
        # set max_outflow_rate_map to each scoot_controller
        for scoot_controller in self.getAll():
            scoot_controller.set('max_outflow_rate_map', self.max_outflow_rate_map)

        return
    
    def update(self):
        for scoot_controller in self.getAll():
            scoot_controller.update()
        return
    
class ScootController(Object):
    PHASE_ORDER_LIST = [1, 3, 2, 4]
    STRAIGHT_PHASE_LIST = [1, 2]
    RIGHT_PHASE_LIST = [3, 4]

    MIN_DISTANCE = 1.0
    INITIAL_VEHICLE_SIZE = 5.0
    INITIAL_FLOW_RATE = 0.2

    TURN_LEFT_ID = 1
    GO_STRAIGHT_ID = 2
    TURN_RIGHT_ID = 3

    SPILLBACK_QUEUE_THRESHOLD = 0.5

    def __init__(self, scoot_controllers, intersection, id):
        super().__init__()

        self.config = scoot_controllers.config
        self.executor = scoot_controllers.executor
        self.shared_resources = scoot_controllers.shared_resources
        self.scoot_controllers = scoot_controllers
        self.network = scoot_controllers.network

        self._initProps(id)
        self._connectObjects(intersection)
        self._showInfo('initial')
        return
    
    @property
    def current_time(self):
        return self.network.simulation.get('current_time')

    @property
    def first_partition(self):
        return self.remain_steps_info['split'][0]

    @property
    def second_partition(self):
        return self.remain_steps_info['split'][1]

    @property
    def third_partition(self):
        return self.remain_steps_info['split'][2]
    
    @property
    def fourth_partition(self):
        return self.remain_steps_info['split'][3]
    
    @property
    def previous_partition(self):
        return self.fourth_partition
    
    @property
    def previous_phase(self):
        return self.previous_partition['phase']['from']

    @property
    def current_phase(self):
        return self.first_partition['phase']['from']
    
    @property
    def next_phase(self):
        return self.second_partition['phase']['from']

    @property
    def cycle_update_flg(self):
        return self.remain_steps_info['cycle'] == 0
        
    @property
    def split_update_flg(self):
        if self.first_partition['fixed']:
            return False
        
        return self.first_partition['steps'] <= self.change_steps['split']['normal']
    
    @property
    def saturated_flg(self):
        if not self.emergency_flg_map['queue']:
            return False

        if self.current_split_changes_map['queue'] >= self.emergency_max_changes_map['queue']:
            return False
        
        if self.first_partition['steps'] > self.change_steps['split']['normal'] + 1:
            return False
        
        if self.first_partition['fixed']:
            return False
        
        return self.current_max_queue_map[self.current_phase] > self.emergency_threshold_map['queue']['up']
    
    @property
    def sparse_flg(self):
        if not self.emergency_flg_map['queue']:
            return False
        
        if self.current_split_changes_map['queue'] >= self.emergency_max_changes_map['queue']:
            return False
        
        if self.first_partition['fixed']:
            return False
        
        return self.current_max_queue_map[self.current_phase]  < self.emergency_threshold_map['queue']['down'] and any(queue > self.emergency_threshold_map['queue']['up'] for queue in self.current_max_queue_map.values())
    
    @property
    def clear_flg(self):
        if not self.emergency_flg_map['pass']:
            return False
        
        if self.current_split_changes_map['pass'] >= self.emergency_max_changes_map['pass']:
            return False
        
        if self.first_partition['fixed']:
            return False
        
        for pass_list in self.road_pass_map.values():
            if len(pass_list) < self.emergency_threshold_map['pass']:
                return False
            
            if self.first_partition['steps'] + len(pass_list) == self.params['split'][self.current_phase]:
                return False
            
            if any(pass_list):
                return False
               
        return True
    
    @property
    def blocked_flg(self):
        if not self.emergency_flg_map['blocked']:
            return False
        
        if self.current_split_changes_map['blocked'] >= self.emergency_max_changes_map['blocked']:
            return False
        
        if self.first_partition['fixed']:
            return False
        
        return self.blocked_map[self.current_phase]

    def _initProps(self, id):
        # set id and num_phases
        self.id = id
        self.num_phases = len(self.PHASE_ORDER_LIST)

        # set params, remain_steps_info, change_steps, and thresholds
        scoot_info = self.config.get('scoot_info')
        self.params = {
            'cycle': scoot_info['initial_parameters']['cycle'],
            'split': {
                phase_id: scoot_info['initial_parameters']['split'][phase_id - 1] for phase_id in range(1, self.num_phases + 1)
            }
        }
        self.previous_params = copy.deepcopy(self.params)

        self.remain_steps_info = {
            'cycle': self.params['cycle'],
            'split': deque(maxlen=self.num_phases)
        }
        sum_steps = 0
        for id in range(len(self.PHASE_ORDER_LIST)):
            sum_steps += self.params['split'][self.PHASE_ORDER_LIST[id]]
            self.remain_steps_info['split'].append({
                'phase': {
                    'from': self.PHASE_ORDER_LIST[id],
                    'to': self.PHASE_ORDER_LIST[(id + 1) % self.num_phases]
                },
                'steps': sum_steps,
                'fixed': False
            })

        self.change_steps = scoot_info['change_steps']
        self.min_cycle = scoot_info['range']['cycle']['min']
        self.max_cycle = scoot_info['range']['cycle']['max']
        self.min_split = scoot_info['range']['split']['min']

        self.normal_threshold_map = {}
        self.normal_threshold_map['saturation'] = {
            'up': scoot_info['normal']['saturation']['threshold']['up'],
            'down': scoot_info['normal']['saturation']['threshold']['down'],
        }

        self.emergency_flg_map = {
            'queue': scoot_info['emergency']['queue']['flg'],
            'pass': scoot_info['emergency']['pass']['flg'],
            'blocked': scoot_info['emergency']['blocked']['flg'],
        }
        
        self.emergency_threshold_map = {}
        if self.emergency_flg_map['queue']:
            self.emergency_threshold_map['queue'] = {
                'down': scoot_info['emergency']['queue']['threshold']['down'],
                'up': scoot_info['emergency']['queue']['threshold']['up'],
            }
        if self.emergency_flg_map['pass']:
            self.emergency_threshold_map['pass'] = scoot_info['emergency']['pass']['threshold']

        self.emergency_max_changes_map = {}
        for emergency_type in ['queue', 'pass', 'blocked']:
            if self.emergency_flg_map[emergency_type]:
                self.emergency_max_changes_map[emergency_type] = scoot_info['emergency'][emergency_type]['max_changes']

        # set time_step
        self.time_step = self.network.simulation.get('time_step')

        # set phases_map
        self.phases_map = None

        # set num_roads
        self.num_roads = None

        # set branch_info_map, from_pos_map, and sig_pos_map
        self.branch_info_map = None
        self.from_pos_map = None
        self.sig_pos_map = None

        # set properties for signal parameter adjustment
        self.inflow_rate_record_map = None
        self.outflow_rate_record_map = None
        self.max_outflow_rate_map = None
        self.saturation_map = None
        self.phase_max_saturation_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        self.phase_avg_saturation_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        
        # set properties for emergency signal parameter adjustment
        self.current_split_changes_map = {type: 0 for type, flg in self.emergency_flg_map.items() if flg}
        self.current_queue_map = None
        self.current_max_queue_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        self.max_queue_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        self.blocked_map = {phase_id: False for phase_id in range(1, self.num_phases + 1)}
        self.road_pass_map = None

        return
    
    def _connectObjects(self, intersection):
        # set intersection
        self.intersection = intersection
        self.intersection.scoot_controller = self

        # set signal_controller and roads
        self.signal_controller = self.intersection.signal_controller
        self.roads = self.intersection.input_roads
        
        # set initial phases in signal_controller
        self.signal_controller.setPhases([self.first_partition['phase']['from']] * self.first_partition['steps'])

        # set branch_info_map, from_pos_map, and sig_pos_map
        self.branch_info_map, self.from_pos_map, self.sig_pos_map = {}, {}, {}
        for road_id, road in self.roads.items():
            if road.get('type') == 1:
                self.branch_info_map[road_id] = {
                    'pos': road.right_connector.get('from_pos'),
                    'length': {
                        'straight': road.get('length') - road.right_connector.get('from_pos'),
                        'right': road.right_connector.get('length') - road.right_connector.get('to_pos') + road.right_link.get('length'),
                    }
                }
    
                self.from_pos_map[road_id] = {
                    'straight': 0.0,
                    'right_connector': road.right_connector.get('from_pos'),
                    'right_link': road.right_connector.get('from_pos') + road.right_connector.get('length') - road.right_connector.get('to_pos'),
                }

                self.sig_pos_map[road_id] = {
                    'straight': road.get('length'),
                    'right': road.right_connector.get('from_pos') + road.right_connector.get('length') - road.right_connector.get('to_pos') + road.right_link.get('length'),
                }
            else:
                raise NotImplementedError(f"Not supported road type: {road.get('type')}")
            
        # set num_roads
        self.num_roads = self.roads.count()

        # set phases_map
        self.phases_map = {phase_id: [] for phase_id in range(1, self.num_phases + 1)}
        phases_df = self.config.get('phases_df_map')[self.num_roads]
        for _, phase_row in phases_df.iterrows():
            if int(phase_row['id']) not in self.PHASE_ORDER_LIST:
                continue
            
            for signal_group_order_id in range(1, self.num_roads + 1):
                signal_group_id = int(phase_row[f"signal_group{signal_group_order_id}"])
                road_id = (signal_group_id - 1) // (self.num_roads - 1) + 1
                route_id = (signal_group_id - 1) % (self.num_roads - 1) + 1
                self.phases_map[int(phase_row['id'])].append((road_id, route_id))
        
        self.inflow_rate_record_map = {(road_id, route_id): pd.DataFrame(columns=['time', 'inflow_rate']) for road_id in range(1, self.num_roads + 1) for route_id in range(1, self.num_roads)}
        self.outflow_rate_record_map = {(road_id, route_id): pd.DataFrame(columns=['time', 'outflow_rate']) for road_id in range(1, self.num_roads + 1) for route_id in range(1, self.num_roads)}
        self.saturation_map = {(road_id, route_id): 0.0 for road_id in range(1, self.num_roads + 1) for route_id in range(1, self.num_roads)}
        
        self.current_queue_map = {}
        for road_id, road in self.roads.items():
            if road.get('type') == 1:
                self.current_queue_map[road_id] = {'straight': 0.0, 'right': 0.0}
            else:
                raise NotImplementedError(f"Not supported road type: {road.get('type')}")

        self.road_pass_map = {}
        for road_id, _ in self.phases_map[self.current_phase]:
            if road_id not in self.road_pass_map.keys():
                self.road_pass_map[road_id] = deque(maxlen=self.emergency_threshold_map['pass'])
        return
    
    def _updateQueueInfo(self):
        for road_id, road in self.roads.items():
            if road.get('type') == 1:
                vehicles_df = road.get('vehicles_df')
                vehicles_df = vehicles_df[
                    (vehicles_df['link_id'] == road.main_link.get('id')) &
                    (vehicles_df['position'] >= self.branch_info_map[road_id]['pos'])
                ].sort_values(by='position', ascending=True).reset_index(drop=True)
                queue_length = 0.0 if vehicles_df.empty else self.sig_pos_map[road_id]['straight'] - vehicles_df.iloc[0]['position']

                if queue_length / self.branch_info_map[road_id]['length']['straight'] >= self.SPILLBACK_QUEUE_THRESHOLD:
                    queue_length = max(queue_length, road.get('max_queue_length'))

                self.current_queue_map[road_id]['straight'] = queue_length

                vehicles_df = road.get('vehicles_df')
                if any(vehicles_df['link_id'] == road.right_connector.get('id')):
                    vehicles_df = vehicles_df[vehicles_df['link_id'] == road.right_connector.get('id')].sort_values(by='position', ascending=True).reset_index(drop=True)
                    queue_length = self.sig_pos_map[road_id]['right'] - (self.from_pos_map[road_id]['right_connector'] + vehicles_df.iloc[0]['position'])
                elif any(vehicles_df['link_id'] == road.right_link.get('id')):
                    vehicles_df = vehicles_df[vehicles_df['link_id'] == road.right_link.get('id')].sort_values(by='position', ascending=True).reset_index(drop=True)
                    queue_length = self.sig_pos_map[road_id]['right'] - (self.from_pos_map[road_id]['right_link'] + vehicles_df.iloc[0]['position'])
                else:
                    queue_length = 0.0
                
                if queue_length / self.branch_info_map[road_id]['length']['right'] >= self.SPILLBACK_QUEUE_THRESHOLD:
                    queue_length = max(queue_length, road.get('max_queue_length'))
                
                self.current_queue_map[road_id]['right'] = queue_length

            else:
                raise NotImplementedError(f"Not supported road type: {road.get('type')}")

        for phase_id in range(1, self.num_phases + 1):
            self.current_max_queue_map[phase_id] = 0.0

            if phase_id in self.STRAIGHT_PHASE_LIST:
                road_set = set([road_id for road_id, _ in self.phases_map[phase_id]])
            elif phase_id in self.RIGHT_PHASE_LIST:
                road_set = set([road_id for road_id, route_id in self.phases_map[phase_id] if route_id == self.TURN_RIGHT_ID])
            else:
                raise NotImplementedError(f"Not supported phase id: {phase_id}")

            for road_id in road_set:
                road = self.roads[road_id]
                if road.get('type') == 1:
                    if phase_id in self.STRAIGHT_PHASE_LIST:
                        self.current_max_queue_map[phase_id] = max(
                            self.current_max_queue_map[phase_id],
                            self.current_queue_map[road_id]['straight'] / road.get('length')
                        )  
                    elif phase_id in self.RIGHT_PHASE_LIST:
                        self.current_max_queue_map[phase_id] = max(
                            self.current_max_queue_map[phase_id],
                            self.current_queue_map[road_id]['right'] / road.get('length')
                        )
                    else:
                        raise NotImplementedError(f"Not supported phase id: {phase_id}")
                else:
                    raise NotImplementedError(f"Not supported road type: {road.get('type')}")
                
            self.max_queue_map[phase_id] = max(self.max_queue_map[phase_id], self.current_max_queue_map[phase_id])
        return
    
    def _updateBlockedInfo(self):
        for phase_id in range(1, self.num_phases + 1):
            self.blocked_map[phase_id] = False  
            if phase_id in self.STRAIGHT_PHASE_LIST:
                road_set = set([road_id for road_id, _ in self.phases_map[phase_id]])
            elif phase_id in self.RIGHT_PHASE_LIST:
                road_set = set([road_id for road_id, route_id in self.phases_map[phase_id] if route_id == self.TURN_RIGHT_ID])
            else:
                raise NotImplementedError(f"Not supported phase id: {phase_id}")
            
            for road_id in road_set:
                road = self.roads[road_id]
                if road.get('type') == 1:
                    if phase_id in self.STRAIGHT_PHASE_LIST:
                        tmp_blocked_flg = self.current_queue_map[road_id]['right'] >= self.branch_info_map[road_id]['length']['right']
                        tmp_blocked_flg = tmp_blocked_flg and self.current_queue_map[road_id]['straight'] <= self.branch_info_map[road_id]['length']['straight'] / 2
                        self.blocked_map[phase_id] = self.blocked_map[phase_id] or tmp_blocked_flg
                    elif phase_id in self.RIGHT_PHASE_LIST:
                        tmp_blocked_flg = self.current_queue_map[road_id]['straight'] >= self.branch_info_map[road_id]['length']['straight']
                        tmp_blocked_flg = tmp_blocked_flg and self.current_queue_map[road_id]['right'] <= self.branch_info_map[road_id]['length']['right'] / 2
                        self.blocked_map[phase_id] = self.blocked_map[phase_id] or tmp_blocked_flg
                    else:
                        raise NotImplementedError(f"Not supported phase id: {phase_id}")
                else:
                    raise NotImplementedError(f"Not supported road type: {road.get('type')}")                
        return
    
    def _updatePassInfo(self):
        if self.current_phase in self.STRAIGHT_PHASE_LIST:
            for road_id in set([tmp_road_id for tmp_road_id, _ in self.phases_map[self.current_phase]]):
                road = self.roads[road_id]
                pass_flg = False
                for data_collection_point in road.data_collection_points.getAll():
                    if data_collection_point.get('type') != 'intersection':
                        continue

                    if data_collection_point.get('route_id') == self.TURN_RIGHT_ID:
                        continue

                    if data_collection_point.get('current_num_vehs') != 0:
                        pass_flg = True
                        break
                
                self.road_pass_map[road_id].append(pass_flg)
        elif self.current_phase in self.RIGHT_PHASE_LIST:
            for road_id in set([tmp_road_id for tmp_road_id, route_id in self.phases_map[self.current_phase] if route_id == self.TURN_RIGHT_ID]):
                road = self.roads[road_id]
                pass_flg = False
                for data_collection_point in road.data_collection_points.getAll():
                    if data_collection_point.get('type') != 'intersection':
                        continue

                    if data_collection_point.get('route_id') != self.TURN_RIGHT_ID:
                        continue

                    if data_collection_point.get('current_num_vehs') != 0:
                        pass_flg = True
                        break
                
                self.road_pass_map[road_id].append(pass_flg)
        else:
            raise NotImplementedError(f"Not supported phase id: {self.current_phase}")
        
        return
    
    def _updateFlowRateInfo(self):
        # update inflow_rate_record_map
        for road_id, route_id in self.phases_map[self.current_phase]:
            # get inflow_rate_record
            inflow_rate_record = self.inflow_rate_record_map[(road_id, route_id)]

            # get inflow_rate
            road = self.roads[road_id]
            for data_collection_point in road.data_collection_points.getAll():
                if data_collection_point.get('type') != 'input':
                    continue

                inflow_rate = data_collection_point.getFlowRate(duration_step=self.params['cycle'] - self.change_steps['split']['normal'])
                break
            
            # get turn_ratios
            turn_ratios = road.get('turn_ratios')

            # add new record to inflow_rate_record
            inflow_rate_record.loc[len(inflow_rate_record)] = {
                'time': int(self.network.get('current_time')),
                'inflow_rate': inflow_rate * turn_ratios[route_id] / sum(turn_ratios.values())
            }

        # update outflow_rate_record_map and max_outflow_rate_map
        for road_id, route_id in self.phases_map[self.current_phase]:
            # get outflow_rate_record
            outflow_rate_record = self.outflow_rate_record_map[(road_id, route_id)]

            # get outflow_rate
            road = self.roads[road_id]
            outflow_rate = 0.0
            for data_collection_point in road.data_collection_points.getAll():
                if data_collection_point.get('type') != 'intersection':
                    continue

                if data_collection_point.get('route_id') != route_id:
                    continue

                outflow_rate += data_collection_point.getFlowRate(duration_step=self.params['split'][self.current_phase] - self.change_steps['split']['normal'])

            # add new record to outflow_rate_record
            outflow_rate_record.loc[len(outflow_rate_record)] = {
                'time': int(self.network.get('current_time')),
                'outflow_rate': outflow_rate
            }

            # set new max_outflow_rate
            tmp_key = (self.num_roads, road.get('type'), route_id, road.get('max_speed'))
            if self.params['split'][self.current_phase] in self.max_outflow_rate_map[tmp_key]:
                self.max_outflow_rate_map[tmp_key][self.params['split'][self.current_phase]] = max(self.max_outflow_rate_map[tmp_key][self.params['split'][self.current_phase]], outflow_rate)

                for num_steps in self.max_outflow_rate_map[tmp_key].keys():
                    if num_steps <= self.params['split'][self.current_phase]:
                        continue

                    self.max_outflow_rate_map[tmp_key][num_steps] = max(
                        self.max_outflow_rate_map[tmp_key][num_steps], 
                        self.max_outflow_rate_map[tmp_key][self.params['split'][self.current_phase]]
                    )
            else:
                self.max_outflow_rate_map[tmp_key][self.params['split'][self.current_phase]] = max(self.INITIAL_FLOW_RATE, outflow_rate)

        return
    
    def _updateSaturationInfo(self):
        # update saturation_map
        for road_id, route_id in self.phases_map[self.current_phase]:
            # get road
            road = self.roads[road_id]

            # get inflow_rate and max_outflow_rate
            inflow_rate = self.inflow_rate_record_map[(road_id, route_id)]['inflow_rate'].iloc[-1]
            max_outflow_rate = self.max_outflow_rate_map[(self.num_roads, road.get('type'), route_id, road.get('max_speed'))][self.params['split'][self.current_phase]]

            # update saturation_map 
            self.saturation_map[(road_id, route_id)] = inflow_rate * self.params['cycle'] / (max_outflow_rate * self.params['split'][self.current_phase])    
        
        # update phase_max_saturation_map and phase_avg_saturation_map
        self.phase_max_saturation_map = {}
        for phase_id in range(1, self.num_phases + 1):
            if phase_id in self.STRAIGHT_PHASE_LIST:
                self.phase_max_saturation_map[phase_id] = max([self.saturation_map[(road_id, route_id)] for road_id, route_id in self.phases_map[phase_id]])
            elif phase_id in self.RIGHT_PHASE_LIST:
                max_saturation = 0.0
                for road_id, route_id in self.phases_map[phase_id]:
                    if route_id == self.TURN_RIGHT_ID:
                        max_saturation = max(max_saturation, self.saturation_map[(road_id, route_id)])
                self.phase_max_saturation_map[phase_id] = max_saturation
            else:
                raise NotImplementedError(f"Not supported phase id: {phase_id}")
            
        self.phase_avg_saturation_map = {}
        for phase_id in range(1, self.num_phases + 1):
            if phase_id in self.STRAIGHT_PHASE_LIST:
                self.phase_avg_saturation_map[phase_id] = sum([self.saturation_map[(road_id, route_id)] for road_id, route_id in self.phases_map[phase_id]]) / len(self.phases_map[phase_id])
            elif phase_id in self.RIGHT_PHASE_LIST:
                saturation_list = []
                for road_id, route_id in self.phases_map[phase_id]:
                    if route_id == self.TURN_RIGHT_ID:
                        saturation_list.append(self.saturation_map[(road_id, route_id)])
                self.phase_avg_saturation_map[phase_id] = sum(saturation_list) / len(saturation_list) if len(saturation_list) > 0 else 0.0
            else:
                raise NotImplementedError(f"Not supported phase id: {phase_id}")
        return

    def _updateTrafficInfo(self):
        # update queue information
        self._updateQueueInfo()

        # update blocked information
        self._updateBlockedInfo()
        
        # update vehicle passing information
        self._updatePassInfo()
        
        if not self.split_update_flg:
            return
        
        # update flow rate information
        self._updateFlowRateInfo()
        
        # update saturation information
        self._updateSaturationInfo()
        return 
    
    def _updateSplit(self):
        if self.split_update_flg:
            if self.phase_avg_saturation_map[self.current_phase] > self.phase_avg_saturation_map[self.next_phase]:
                self._incrementSplit()
            else:
                self._decrementSplit()

            return

        # emergency split update
        
        if self.saturated_flg:
            self._incrementSplit()
            self.current_split_changes_map['queue'] += 1

        elif self.sparse_flg:
            self._decrementSplit()
            self.current_split_changes_map['queue'] += 1
        
        elif self.clear_flg:
            self._decrementSplit()
            self.current_split_changes_map['pass'] += 1

        elif self.blocked_flg:
            self._decrementSplit()
            self.current_split_changes_map['blocked'] += 1

        else:
            raise NotImplementedError('Not supported split update type')
        
        return
    
    def _decrementSplit(self):
        if self.split_update_flg:
            change_steps = min(self.change_steps['split']['normal'], self.params['split'][self.current_phase] - self.min_split, self.first_partition['steps'])
        elif self.sparse_flg or self.clear_flg or self.blocked_flg:
            change_steps = min(self.change_steps['split']['emergency'], self.params['split'][self.current_phase] - self.min_split, self.first_partition['steps'])
        else:
            raise NotImplementedError('Not supported split update type')
        
        if change_steps <= 0: return

        if self.split_update_flg:
            self.first_partition['fixed'] = True
        self.first_partition['steps'] -= change_steps

        self.params['split'][self.current_phase] -= change_steps
        self.params['split'][self.next_phase] += change_steps

        self.signal_controller.deletePhases(type='end', steps=change_steps)
        return
    
    def _incrementSplit(self):
        # get change_steps
        if self.split_update_flg:
            change_steps = self.change_steps['split']['normal']
        elif self.saturated_flg:
            change_steps = self.change_steps['split']['emergency']
        else:
            raise NotImplementedError('Not supported split update type')
        
        # get change_steps_list
        change_steps_list = []
        for partition_id in range(0, self.num_phases - 1):
            partition = self.remain_steps_info['split'][partition_id]
            if len(change_steps_list) == 0 or sum(change_steps_list) < change_steps:
                change_steps_list.append(min(
                    change_steps - sum(change_steps_list), 
                    self.params['split'][partition['phase']['to']] - self.min_split
                ))
            else:
                change_steps_list.append(0)

        # update fix_flg
        if self.split_update_flg:
            self.first_partition['fixed'] = True
        
        # skip update if no change happens
        if sum(change_steps_list) <= 0: return

        for partition_id in range(0, self.num_phases - 1):
            # update partition
            partition = self.remain_steps_info['split'][partition_id]
            partition['steps'] += sum(change_steps_list[partition_id:])

            # update split parameters except for current phase
            self.params['split'][partition['phase']['to']] -= change_steps_list[partition_id]
        
        # update split parameter for current phase
        self.params['split'][self.current_phase] += sum(change_steps_list)
        self.signal_controller.setPhases([self.current_phase] * sum(change_steps_list))
        return
    
    def _updateCycle(self):
        # get phase_change_map
        phase_change_map = self._getPhaseChangeMap()

        # update split information
        cumulative_change_steps = 0
        for partition_id, partition in enumerate(self.remain_steps_info['split']):
            cumulative_change_steps += phase_change_map[partition['phase']['from']]

            partition['steps'] += cumulative_change_steps
            self.params['split'][partition['phase']['from']] += phase_change_map[partition['phase']['from']]

            if partition_id != 0:
                continue

            if partition['steps'] <= self.change_steps['split']['normal']:
                partition['fixed'] = True
            
            if phase_change_map[partition['phase']['from']] > 0:
                self.signal_controller.setPhases([partition['phase']['from']] * phase_change_map[partition['phase']['from']])

            elif phase_change_map[partition['phase']['from']] < 0:
                self.signal_controller.deletePhases(type='end', steps=-phase_change_map[partition['phase']['from']])

        # update cycle information
        self.params['cycle'] += cumulative_change_steps    

        # show update information
        self._showInfo('update')

        # update previous_params
        self.previous_params = copy.deepcopy(self.params)
        return
    
    def _getPhaseChangeMap(self):
        phase_change_map = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
        cumurative_change_steps = 0
        
        saturation_phase_list = sorted(range(1, self.num_phases + 1), key=lambda phase_id: self.phase_max_saturation_map[phase_id], reverse=True)
        for phase_id in reversed(saturation_phase_list):
            if not (self.phase_max_saturation_map[phase_id] < self.normal_threshold_map['saturation']['down']):
                continue

            change_steps = min(self.params['cycle'] + cumurative_change_steps - self.min_cycle, self.change_steps['cycle'], self.params['split'][phase_id] - self.min_split)
            if change_steps <= 0: continue

            if self.first_partition['phase']['from'] == phase_id:
                change_steps = min(change_steps, self.first_partition['steps'])
            
            phase_change_map[phase_id] = - change_steps
            cumurative_change_steps += phase_change_map[phase_id]
        

        for phase_id in saturation_phase_list:
            if not (self.phase_max_saturation_map[phase_id] > self.normal_threshold_map['saturation']['up']):
                continue

            change_steps = min(self.max_cycle - self.params['cycle'] - cumurative_change_steps, self.change_steps['cycle'])
            if change_steps <= 0: continue

            phase_change_map[phase_id] = change_steps
            cumurative_change_steps += phase_change_map[phase_id]
        return phase_change_map

    def _proceedOneStep(self):
        # update split information
        if self.first_partition['steps'] == 0:
            last_partition = self.remain_steps_info['split'].popleft()
            last_partition['steps'] = self.params['cycle']
            last_partition['fixed'] = False
            self.remain_steps_info['split'].append(last_partition)

            self.road_pass_map = {}
            if self.current_phase in self.STRAIGHT_PHASE_LIST:
                for road_id, _ in self.phases_map[self.current_phase]:
                    if road_id not in self.road_pass_map.keys():
                        self.road_pass_map[road_id] = deque(maxlen=self.emergency_threshold_map['pass'])
            
            elif self.current_phase in self.RIGHT_PHASE_LIST:
                for road_id, route_id in self.phases_map[self.current_phase]:
                    if route_id == self.TURN_LEFT_ID:
                        continue

                    if road_id not in self.road_pass_map.keys():
                        self.road_pass_map[road_id] = deque(maxlen=self.emergency_threshold_map['pass'])

            self.current_split_changes_map = {type: 0 for type, flg in self.emergency_flg_map.items() if flg}

            self.signal_controller.setPhases([self.first_partition['phase']['from']] * self.first_partition['steps'])
        
        for partition in self.remain_steps_info['split']:
            partition['steps'] -= 1

        # update cycle information
        if self.remain_steps_info['cycle'] == 0:
            self.remain_steps_info['cycle'] = self.params['cycle']

            self.max_queue_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}

        self.remain_steps_info['cycle'] -= 1
        return

    def _showInfo(self, type):
        print('==============================================')
        if type == 'update':
            print('status: update parameters')
            print(f"intersection id: {self.id}")
            print(f"cycle: {self.previous_params['cycle']} -> {self.params['cycle']} steps")
            for phase_id in self.PHASE_ORDER_LIST:
                print(f"phase {phase_id}: {self.previous_params['split'][phase_id]} -> {self.params['split'][phase_id]} steps")
            
        elif type == 'initial':
            print('status: setup scoot controller')
            print(f"intersection id: {self.id}")
            print(f"cycle: {self.params['cycle']} steps")
            for phase_id in self.PHASE_ORDER_LIST:
                print(f"phase {phase_id}: {self.params['split'][phase_id]} steps")

        else:
            raise NotImplementedError(f"Not supported type: {type}")
    
    def update(self):
        # update traffic information
        self._updateTrafficInfo()

        # update cycle
        if self.cycle_update_flg:
            self._updateCycle()
        
        # if cycle adjustment leads first_partition to be fixed, proceed one step without split adjustment and return
        if self.first_partition['fixed']:
            self._proceedOneStep()
            return
        
        # update split
        if self.split_update_flg or self.saturated_flg or self.sparse_flg or self.clear_flg or self.blocked_flg:
            self._updateSplit()
        
        # proceed one step
        self._proceedOneStep()
        return