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

    NORTH_ROAD_ID = 1
    EAST_ROAD_ID = 2
    SOUTH_ROAD_ID = 3
    WEST_ROAD_ID = 4

    PHASE_ROAD_LIST_MAP = {
        1: [NORTH_ROAD_ID, SOUTH_ROAD_ID],
        2: [EAST_ROAD_ID, WEST_ROAD_ID],
        3: [NORTH_ROAD_ID, SOUTH_ROAD_ID],
        4: [EAST_ROAD_ID, WEST_ROAD_ID],    
    }

    OPPOSITE_PHASE_MAP = {
        1: 2,
        2: 1,
        3: 4,
        4: 3,
    }

    SAME_DIRECTION_PHASE_MAP = {
        1: 3,
        2: 4,
        3: 1,
        4: 2,
    }

    STRAIGHT_PHASE_LIST = [1, 2]
    RIGHT_PHASE_LIST = [3, 4]

    MIN_DISTANCE = 1.0
    INITIAL_VEHICLE_SIZE = 5.0

    TURN_LEFT_ID = 1
    GO_STRAIGHT_ID = 2
    TURN_RIGHT_ID = 3

    INITIAL_FLOW_RATE = 0.2
    SPILLBACK_THRESHOLD = 0.90

    QUEUE_THRESHOLD = 0.5
    SATURATION_THRESHOLD_MAP = {
        'up': 1.0,
        'down': 0.9,
    }

    PASS_DURATION = 2

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
    def previous_phase(self):
        return self.remain_steps_info['split'][-1]['phase']['from']

    @property
    def current_phase(self):
        return self.remain_steps_info['split'][0]['phase']['from']
    
    @property
    def next_phase(self):
        return self.remain_steps_info['split'][1]['phase']['from']

    @property
    def cycle_update_flg(self):
        return self.remain_steps_info['cycle'] == 0
    
    @property
    def split_update_flg(self):
        return self.current_blocked_flg or self.normal_split_update_flg or self.over_queue_flg
        
    @property
    def normal_split_update_flg(self):
        return self.first_partition['steps'] == self.change_steps['split']['normal'] and not self.first_partition['fixed']

    @property
    def current_blocked_flg(self):
        return any(self.blocked_info_map[self.current_phase].values())
    
    @property
    def over_queue_flg(self):
        return self.current_max_queue_map[self.current_phase] > self.QUEUE_THRESHOLD and self.current_max_queue_map[self.current_phase] >= self.current_max_queue_map[self.next_phase]
        
    @property
    def no_pass_flg(self):
        for pass_flg_list in self.pass_map.values():
            if not any(pass_flg_list):
                return True
        
        return False
    
    @property
    def next_blocked_flg(self):
        return self.current_phase in self.STRAIGHT_PHASE_LIST and any(self.blocked_info_map[self.next_phase].values())

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
        self.max_cycle = scoot_info['max_cycle']
        self.min_split = scoot_info['min_split']

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

        # initialize other properties
        self.inflow_rate_record_map = None
        self.outflow_rate_record_map = None
        self.saturation_map = None
        self.max_outflow_rate_map = None
        self.current_blocked_info_map = {phase_id: {} for phase_id in range(1, self.num_phases + 1)}
        self.blocked_info_map = {phase_id: {} for phase_id in range(1, self.num_phases + 1)}
        self.current_max_queue_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        self.max_queue_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        self.pass_map = None
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
        
        self.pass_map = {}
        for road_id, _ in self.phases_map[self.current_phase]:
            if road_id not in self.pass_map.keys():
                self.pass_map[road_id] = deque(maxlen=self.PASS_DURATION)
        return

    def _updateTrafficInfo(self):
        # udpate blocked_info_map, current_blocked_info_map
        for phase_id in range(1, self.num_phases + 1):
            if phase_id in self.STRAIGHT_PHASE_LIST:
                for road_id in set([tmp_road_id for tmp_road_id, _ in self.phases_map[phase_id]]):
                    road = self.roads[road_id]    
                    if road.get('type') == 1:
                        vehicles_df = road.get('vehicles_df')
                        vehicle_size = vehicles_df['length'].mean() if not vehicles_df.empty else self.INITIAL_VEHICLE_SIZE

                        num_vehs = vehicles_df[vehicles_df['link_id'].isin([road.right_link.get('id'), road.right_connector.get('id')])].shape[0]
                        
                        self.current_blocked_info_map[phase_id][road_id] = num_vehs * (vehicle_size + self.MIN_DISTANCE) >= self.branch_info_map[road_id]['length']['right'] * self.SPILLBACK_THRESHOLD
                        
                    else:
                        raise NotImplementedError(f"Not supported road type: {road.get('type')}")
            
            elif phase_id in self.RIGHT_PHASE_LIST:
                for road_id in set([tmp_road_id for tmp_road_id, route_id in self.phases_map[phase_id] if route_id == self.TURN_RIGHT_ID]):
                    road = self.roads[road_id]
                    if road.get('type') == 1:
                        vehicles_df = road.get('vehicles_df')
                        vehicle_size = vehicles_df['length'].mean() if not vehicles_df.empty else self.INITIAL_VEHICLE_SIZE

                        num_vehs = vehicles_df[
                            (vehicles_df['link_id'] == road.main_link.get('id')) &
                            (vehicles_df['position'] >= self.branch_info_map[road_id]['pos'])
                        ].shape[0]
                        self.current_blocked_info_map[phase_id][road_id] = num_vehs * (vehicle_size + self.MIN_DISTANCE) >= self.branch_info_map[road_id]['length']['straight'] * self.SPILLBACK_THRESHOLD
                    else:
                        raise NotImplementedError(f"Not supported road type: {road.get('type')}")

            else:
                raise NotImplementedError(f"Not supported phase id: {phase_id}")
            
        if self.current_phase in self.STRAIGHT_PHASE_LIST:
            self.blocked_info_map[self.current_phase] = copy.deepcopy(self.current_blocked_info_map[self.current_phase])
            
            if len(self.blocked_info_map[self.SAME_DIRECTION_PHASE_MAP[self.current_phase]].keys()) == 0:
                self.blocked_info_map[self.SAME_DIRECTION_PHASE_MAP[self.current_phase]] = copy.deepcopy(self.current_blocked_info_map[self.SAME_DIRECTION_PHASE_MAP[self.current_phase]])
        
        # self.blocked_info_map[self.current_phase] = {}

        # if self.current_phase in self.STRAIGHT_PHASE_LIST:
        #     for road_id, road in self.roads.items():
        #         if road.get('type') == 1:
        #             if road_id not in self.PHASE_ROAD_LIST_MAP[self.current_phase]:
        #                 continue

        #             vehicles_df = road.get('vehicles_df')
        #             vehicle_size = vehicles_df['length'].mean()

        #             num_vehs = vehicles_df[vehicles_df['link_id'].isin([road.right_link.get('id'), road.right_connector.get('id')])].shape[0]
        #             self.blocked_info_map[self.current_phase][road_id] = num_vehs * (vehicle_size + self.MIN_DISTANCE) >= self.branch_info_map[road_id]['length']['right'] * self.SPILLBACK_THRESHOLD
        #         else:
        #             raise NotImplementedError(f"Not supported road type: {road.get('type')}")

        # elif self.current_phase in self.RIGHT_PHASE_LIST:
        #     for road_id, road in self.roads.items():
        #         if road.get('type') == 1:
        #             if road_id in self.PHASE_ROAD_LIST_MAP[self.current_phase]:
        #                 vehicles_df = road.get('vehicles_df')
        #                 vehicle_size = vehicles_df['length'].mean()

        #                 num_vehs = vehicles_df[
        #                     (vehicles_df['link_id'] == road.main_link.get('id')) &
        #                     (vehicles_df['position'] >= self.branch_info_map[road_id]['pos'])
        #                 ].shape[0]

        #                 self.blocked_info_map[self.current_phase][road_id] = num_vehs * (vehicle_size + self.MIN_DISTANCE) >= self.branch_info_map[road_id]['length']['straight'] * self.SPILLBACK_THRESHOLD
                    
        #             elif road_id in self.PHASE_ROAD_LIST_MAP[self.OPPOSITE_PHASE_MAP[self.current_phase]]:
        #                 if road_id in self.blocked_info_map[self.OPPOSITE_PHASE_MAP[self.current_phase]] and self.blocked_info_map[self.OPPOSITE_PHASE_MAP[self.current_phase]][road_id]:
        #                     continue

        #                 vehicles_df = road.get('vehicles_df')
        #                 vehicle_size = vehicles_df['length'].mean()

        #                 num_vehs = vehicles_df[
        #                     (vehicles_df['link_id'] == road.main_link.get('id')) &
        #                     (vehicles_df['position'] >= self.branch_info_map[road_id]['pos'])
        #                 ].shape[0]

        #                 self.blocked_info_map[self.OPPOSITE_PHASE_MAP[self.current_phase]][road_id] = num_vehs * (vehicle_size + self.MIN_DISTANCE) >= self.branch_info_map[road_id]['length']['straight'] * self.SPILLBACK_THRESHOLD
        #             else:
        #                 raise NotImplementedError(f"Road id {road_id} is not included in current phase {self.current_phase} and opposite phase {self.OPPOSITE_PHASE_MAP[self.current_phase]}")
        #         else:
        #             raise NotImplementedError(f"Not supported road type: {road.get('type')}")
        # else:
        #     raise NotImplementedError(f"Not supported phase id: {self.current_phase}")
        
        # update max_queue_map, current_max_queue_map
        for phase_id in range(1, self.num_phases + 1):
            self.current_max_queue_map[phase_id] = 0.0
            if phase_id in self.STRAIGHT_PHASE_LIST:
                for road_id in set([tmp_road_id for tmp_road_id, _ in self.phases_map[phase_id]]):
                    road = self.roads[road_id]
                    if road.get('type') == 1:
                        vehicles_df = road.get('vehicles_df')
                        vehicles_df = vehicles_df[
                            (vehicles_df['link_id'] == road.main_link.get('id')) &
                            (vehicles_df['position'] >= self.branch_info_map[road_id]['pos'])
                        ].sort_values(by='position', ascending=True).reset_index(drop=True)
                        queue_length = 0.0 if vehicles_df.empty else self.sig_pos_map[road_id]['straight'] - vehicles_df.iloc[0]['position']

                        if queue_length / self.branch_info_map[road_id]['length']['straight'] >= 0.5:
                            queue_length = max(queue_length, road.get('main_queue_length'))

                        # if self.current_blocked_info_map[phase_id][road_id]:
                        #     vehicles_df = road.get('vehicles_df')
                        #     vehicles_df = vehicles_df[
                        #         (vehicles_df['link_id'] == road.main_link.get('id')) &
                        #         (vehicles_df['position'] >= self.branch_info_map[road_id]['pos'])
                        #     ].sort_values(by='position', ascending=True).reset_index(drop=True)
                        #     queue_length = 0.0 if vehicles_df.empty else self.sig_pos_map[road_id]['straight'] - vehicles_df.iloc[0]['position']
                        # else:
                        #     queue_length = road.get('main_queue_length')
                        
                        self.current_max_queue_map[phase_id] = max(self.current_max_queue_map[phase_id], queue_length / road.get('length'))
                    else:
                        raise NotImplementedError(f"Not supported road type: {road.get('type')}")

            elif phase_id in self.RIGHT_PHASE_LIST:
                for road_id in set([tmp_road_id for tmp_road_id, route_id in self.phases_map[phase_id] if route_id == self.TURN_RIGHT_ID]):
                    road = self.roads[road_id]
                    if road.get('type') == 1:
                        vehicles_df = road.get('vehicles_df')
                        if any(vehicles_df['link_id'] == road.right_connector.get('id')):
                            vehicles_df = vehicles_df[vehicles_df['link_id'] == road.right_connector.get('id')].sort_values(by='position', ascending=True).reset_index(drop=True)
                            queue_length = self.sig_pos_map[road_id]['right'] - (self.from_pos_map[road_id]['right_connector'] + vehicles_df.iloc[0]['position'])
                        elif any(vehicles_df['link_id'] == road.right_link.get('id')):
                            vehicles_df = vehicles_df[vehicles_df['link_id'] == road.right_link.get('id')].sort_values(by='position', ascending=True).reset_index(drop=True)
                            queue_length = self.sig_pos_map[road_id]['right'] - (self.from_pos_map[road_id]['right_link'] + vehicles_df.iloc[0]['position'])
                        else:
                            queue_length = 0.0

                        if queue_length / self.branch_info_map[road_id]['length']['right'] >= 0.5:
                            queue_length = max(queue_length, road.get('right_queue_length'))

                        # if len(self.blocked_info_map[phase_id].keys()) == 0:
                        #     blocked_flg = self.current_blocked_info_map[phase_id][road_id]
                        # else:
                        #     blocked_flg = self.blocked_info_map[phase_id][road_id]
                
                        # if blocked_flg:
                        #     vehicles_df = road.get('vehicles_df')
                        #     if any(vehicles_df['link_id'] == road.right_connector.get('id')):
                        #         vehicles_df = vehicles_df[vehicles_df['link_id'] == road.right_connector.get('id')]
                        #         vehicles_df = vehicles_df.sort_values(by='position', ascending=True).reset_index(drop=True)

                        #         queue_length = self.sig_pos_map[road_id]['right'] - (self.from_pos_map[road_id]['right_connector'] + vehicles_df.iloc[0]['position'])
                            
                        #     elif any(vehicles_df['link_id'] == road.right_link.get('id')):
                        #         vehicles_df = vehicles_df[vehicles_df['link_id'] == road.right_link.get('id')]
                        #         vehicles_df = vehicles_df.sort_values(by='position', ascending=True).reset_index(drop=True)
                                
                        #         queue_length = self.sig_pos_map[road_id]['right'] - (self.from_pos_map[road_id]['right_link'] + vehicles_df.iloc[0]['position'])
                            
                        #     else:
                        #         queue_length = 0.0

                        # else:
                        #     queue_length = road.get('right_queue_length')
                    
                        self.current_max_queue_map[phase_id] = max(self.current_max_queue_map[phase_id], queue_length / road.get('length'))
                    else:
                        raise NotImplementedError(f"Not supported road type: {road.get('type')}")
            else:
                raise NotImplementedError(f"Not supported phase id: {phase_id}")
            
            self.max_queue_map[phase_id] = max(self.max_queue_map[phase_id], self.current_max_queue_map[phase_id])

        if self.id == 3:
            print(f"p1: {self.current_max_queue_map[1]:.2f}, p2: {self.current_max_queue_map[2]:.2f}, p3: {self.current_max_queue_map[3]:.2f}, p4: {self.current_max_queue_map[4]:.2f}")
        # for phase_id in range(1, self.num_phases + 1):
        #     self.max_queue_map[phase_id] = 0.0
        #     for road_id, route_id in self.phases_map[phase_id]:
        #         road = self.roads[road_id]

        #         if road.get('type') == 1:
        #             if phase_id in self.RIGHT_PHASE_LIST and route_id == self.TURN_LEFT_ID:
        #                 continue

        #             if route_id in [self.TURN_LEFT_ID, self.GO_STRAIGHT_ID]:
        #                 queue_length = road.get('main_queue_length')
        #             elif route_id == self.TURN_RIGHT_ID:
        #                 queue_length = road.get('right_queue_length')
        #             else:
        #                 raise NotImplementedError(f"Not supported route id: {route_id}")
                    
        #             self.max_queue_map[phase_id] = max(self.max_queue_map[phase_id], queue_length / road.get('length'))
        #         else:
        #             raise NotImplementedError(f"Not supported road type: {road.get('type')}")
        
        # update pass_map
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
                
                self.pass_map[road_id].append(pass_flg)
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
                
                self.pass_map[road_id].append(pass_flg)
        else:
            raise NotImplementedError(f"Not supported phase id: {self.current_phase}")
        
        if not self.normal_split_update_flg:
            return
        
        # update inflow_rate_record_map, outflow_rate_record_map, and max_outflow_rate_map
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

        # update saturation_map
        for road_id, route_id in self.phases_map[self.current_phase]:
            # get road
            road = self.roads[road_id]

            # get inflow_rate and max_outflow_rate
            inflow_rate = self.inflow_rate_record_map[(road_id, route_id)]['inflow_rate'].iloc[-1]
            max_outflow_rate = self.max_outflow_rate_map[(self.num_roads, road.get('type'), route_id, road.get('max_speed'))][self.params['split'][self.current_phase]]

            # update saturation_map 
            self.saturation_map[(road_id, route_id)] = inflow_rate * self.params['cycle'] / (max_outflow_rate * self.params['split'][self.current_phase])    
        
        return 

    def _getMaxSaturation(self, phase_id):
        if phase_id in self.STRAIGHT_PHASE_LIST:
            return max([self.saturation_map[(road_id, route_id)] for road_id, route_id in self.phases_map[phase_id]])
        elif phase_id in self.RIGHT_PHASE_LIST:
            max_saturation = 0.0
            for road_id, route_id in self.phases_map[phase_id]:
                if route_id == self.TURN_RIGHT_ID:
                    max_saturation = max(max_saturation, self.saturation_map[(road_id, route_id)])
            return max_saturation
        else:
            raise NotImplementedError(f"Not supported phase id: {phase_id}")
        
    def _getAvgSaturation(self, phase_id):
        if phase_id in self.STRAIGHT_PHASE_LIST:
            return sum([self.saturation_map[(road_id, route_id)] for road_id, route_id in self.phases_map[phase_id]]) / len(self.phases_map[phase_id])
        elif phase_id in self.RIGHT_PHASE_LIST:
            saturation_list = []
            for road_id, route_id in self.phases_map[phase_id]:
                if route_id == self.TURN_RIGHT_ID:
                    saturation_list.append(self.saturation_map[(road_id, route_id)])
            return sum(saturation_list) / len(saturation_list) if len(saturation_list) > 0 else 0.0
        else:
            raise NotImplementedError(f"Not supported phase id: {phase_id}")
    
    def _updateSplit(self):
        # if self.current_blocked_flg:
        #     self._decrementSplit(type='blocked')
        #     self._showInfo('blocked', 'current')

        # elif self.over_queue_flg:
        #     self._decrementSplit(type='over_queue')
        #     self._showInfo('over_queue')

        # elif not any(self.pass_map[self.current_phase]):
        #     self._decrementSplit(type='not_pass')
        #     self._showInfo('not_pass')
            
        # elif self.normal_split_update_flg:
        #     if self.max_queue_map[self.current_phase] > self.QUEUE_THRESHOLD:
        #         self._incrementSplit('normal')
            
        #     elif self.next_blocked_flg:
        #         self._incrementSplit(type='blocked')
        #         self._showInfo('blocked', 'next')

        #     elif self._getAvgSaturation(self.current_phase) > self._getAvgSaturation(self.next_phase):
        #         self._incrementSplit('normal')

        #     else:
        #         self._decrementSplit('normal')

        # else:
        #     raise NotImplementedError(f"Not supported split update type")

        if self.normal_split_update_flg:
            if self.max_queue_map[self.current_phase] > self.QUEUE_THRESHOLD and self.max_queue_map[self.current_phase] > self.max_queue_map[self.next_phase]:
                self._incrementSplit('normal')

            elif self.max_queue_map[self.next_phase] > self.QUEUE_THRESHOLD and self.max_queue_map[self.next_phase] > self.max_queue_map[self.current_phase]:
                self._decrementSplit('normal')

            else:
                if self._getAvgSaturation(self.current_phase) > self._getAvgSaturation(self.next_phase):
                    self._incrementSplit('normal')
                else:
                    self._decrementSplit('normal')

        elif self.over_queue_flg:
            self._incrementSplit(type='over_queue')
        
        elif self.no_pass_flg:
            self._decrementSplit(type='not_pass')

        
        
        return
    
    def _decrementSplit(self, type):
        if type == 'normal':
            change_steps = min(self.change_steps['split']['normal'], self.params['split'][self.current_phase] - self.min_split, self.first_partition['steps'])
        elif type == 'blocked':
            change_steps = min(self.change_steps['split']['blocked'], self.params['split'][self.current_phase] - self.min_split, self.first_partition['steps'])
        elif type == 'over_queue':
            change_steps = min(self.change_steps['split']['over_queue'], self.params['split'][self.current_phase] - self.min_split, self.first_partition['steps'])
        elif type == 'not_pass':
            change_steps = min(self.change_steps['split']['not_pass'], self.params['split'][self.current_phase] - self.min_split, self.first_partition['steps'])
        else:
            raise NotImplementedError(f"Not supported type: {type}")
        
        if change_steps <= 0: return

        # not change fixed property if only blocked_flg is true
        if self.first_partition['steps'] - change_steps <= self.change_steps['split']['normal']:
            self.first_partition['fixed'] = True
        self.first_partition['steps'] -= change_steps

        self.params['split'][self.current_phase] -= change_steps
        self.params['split'][self.next_phase] += change_steps

        self.signal_controller.deletePhases(type='end', steps=change_steps)
        return
    
    def _incrementSplit(self, type):
        if type == 'normal':
            change_steps = min(self.change_steps['split']['normal'], self.params['split'][self.next_phase] - self.min_split)
        elif type == 'blocked':
            change_steps = min(self.change_steps['split']['blocked'], self.params['split'][self.next_phase] - self.min_split)
        elif type == 'over_queue':
            change_steps = min(self.change_steps['split']['over_queue'], self.params['split'][self.next_phase] - self.min_split)
        else:
            raise NotImplementedError(f"Not supported type: {type}")
        
        if change_steps <= 0: return

        self.first_partition['fixed'] = True
        self.first_partition['steps'] += change_steps

        self.params['split'][self.current_phase] += change_steps
        self.params['split'][self.next_phase] -= change_steps

        self.signal_controller.setPhases([self.current_phase] * change_steps)
        return
    
    def _updateCycle(self):
        # get phase_change_map
        phase_change_map = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
        cumurative_change_steps = 0

        max_saturation_map = {phase_id: self._getMaxSaturation(phase_id) for phase_id in range(1, self.num_phases + 1)} 
        for phase_id in range(1, self.num_phases + 1):
            # if not self.max_queue_map[phase_id] < self.QUEUE_THRESHOLD:
            #     continue

            if max_saturation_map[phase_id] > self.SATURATION_THRESHOLD_MAP['down']:
                continue

            change_steps = min(self.change_steps['cycle'], self.params['split'][phase_id] - self.min_split)
            if change_steps <= 0: continue

            if self.first_partition['phase']['from'] == phase_id:
                phase_change_map[phase_id] = - min(change_steps, self.first_partition['steps'])
            else:
                phase_change_map[phase_id] = - change_steps
            cumurative_change_steps += phase_change_map[phase_id]

        # prioritize_phase_list = sorted(range(1, self.num_phases + 1), key=lambda phase_id: self.max_queue_map[phase_id], reverse=True)
        # for phase_id in prioritize_phase_list:
        #     if self.max_queue_map[phase_id] < self.QUEUE_THRESHOLD:
        #         continue

        #     change_steps = min(self.max_cycle - self.params['cycle'] - cumurative_change_steps, self.change_steps['cycle'])
        #     if change_steps <= 0: continue

        #     phase_change_map[phase_id] = change_steps
        #     cumurative_change_steps += phase_change_map[phase_id]
        
        prioritize_phase_list = sorted(range(1, self.num_phases + 1), key=lambda phase_id: max_saturation_map[phase_id], reverse=True)
        for phase_id in prioritize_phase_list:
            # if not self.max_queue_map[phase_id] < self.QUEUE_THRESHOLD:
            #     continue

            if max_saturation_map[phase_id] < self.SATURATION_THRESHOLD_MAP['up']:
                continue

            change_steps = min(self.max_cycle - self.params['cycle'] - cumurative_change_steps, self.change_steps['cycle'])
            if change_steps <= 0: continue

            phase_change_map[phase_id] = change_steps
            cumurative_change_steps += phase_change_map[phase_id]

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
        # self._showInfo('update')

        # update previous_params
        self.previous_params = copy.deepcopy(self.params)
        return

    def _proceedOneStep(self):
        # update split information
        if self.first_partition['steps'] == 0:
            last_partition = self.remain_steps_info['split'].popleft()
            last_partition['steps'] = self.params['cycle']
            last_partition['fixed'] = False
            self.remain_steps_info['split'].append(last_partition)

            self.pass_map = {}
            if self.current_phase in self.STRAIGHT_PHASE_LIST:
                for road_id, _ in self.phases_map[self.current_phase]:
                    if road_id not in self.pass_map.keys():
                        self.pass_map[road_id] = deque(maxlen=self.PASS_DURATION)
            
            elif self.current_phase in self.RIGHT_PHASE_LIST:
                for road_id, route_id in self.phases_map[self.current_phase]:
                    if route_id == self.TURN_LEFT_ID:
                        continue

                    if road_id not in self.pass_map.keys():
                        self.pass_map[road_id] = deque(maxlen=self.PASS_DURATION)

            self.signal_controller.setPhases([self.first_partition['phase']['from']] * self.first_partition['steps'])
        
        for partition in self.remain_steps_info['split']:
            partition['steps'] -= 1

        # update cycle information
        if self.remain_steps_info['cycle'] == 0:
            self.remain_steps_info['cycle'] = self.params['cycle']

            self.max_queue_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}

        self.remain_steps_info['cycle'] -= 1
        return

    def _showInfo(self, type, option=None):
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
        
        elif type == 'blocked' and option == 'current':
            print('status: blocked in current phase')
            print(f"intersection id: {self.id}")
            print(f"current_phase: {self.current_phase}")
        
        elif type == 'blocked' and option == 'next':
            print('status: blocked in next phase')
            print(f"intersection id: {self.id}")
            print(f"current_phase: {self.current_phase}")
        
        elif type == 'over_queue':
            print('status: over queue threshold in next phase')
            print(f"intersection id: {self.id}")
            print(f"current_phase: {self.current_phase}")

        elif type == 'not_pass':
            print('status: not passing in current phase')
            print(f"intersection id: {self.id}")
            print(f"current_phase: {self.current_phase}")

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
        if self.split_update_flg:
            self._updateSplit()
        
        # proceed one step
        self._proceedOneStep()
        return