from libs.container import Container
from libs.object import Object

from collections import deque
import copy

class ScootControllers(Container):
    def __init__(self, network):
        super().__init__()

        self.config = network.config
        self.executor = network.executor
        self.shared_resources = network.shared_resources
        self.network = network

        self._initElements()
        return
    
    def _initElements(self):
        for intersection_id, intersection in self.network.intersections.items(sorted_flg=True):
            scoot_contoller = ScootController(
                scoot_controllers=self, 
                intersection=intersection,
                id=intersection_id
            )
            self.add(scoot_contoller)
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

    TURN_LEFT_ID = 1
    GO_STRAIGHT_ID = 2
    TURN_RIGHT_ID = 3

    MIN_DISTANCE = 1.0
    INITIAL_VEHICLE_SIZE = 5.0
    SPILLBACK_THRESHOLD = 0.9

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
    def first_partition(self):
        return self.remain_steps_info['split'][0]

    @property
    def current_phase(self):
        return self.first_partition['phase']['from']
    
    @property
    def next_phase(self):
        return self.first_partition['phase']['to']

    @property
    def cycle_update_flg(self):
        return self.remain_steps_info['cycle'] == 0
    
    @property
    def split_update_flg(self):
        return (self.first_partition['steps'] == self.change_steps['split'] and not self.first_partition['fixed'])
    
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
        
        self.saturation_threshold = 1 / scoot_info['spacing_threshold']

        # set branch_length_map (ScootController._connectObjects())
        self.branch_info_map = None
        self.link_phase_map = None

        # set effective_storage_length_map (ScootController._connectObjects())
        self.effective_storage_length_map = None 

        # initialize other properties
        self.total_num_vehicles = None
        self.avg_saturation = None
        self.num_vehs_record = []
        self.saturation_record = []
        self.saturation_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        self.num_vehicles_map = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
        self.vehicle_size = self.INITIAL_VEHICLE_SIZE
        self.blocked_info_map = {phase_id: {} for phase_id in range(1, self.num_phases + 1)}
        # self.empty_info_map = {phase_id: {} for phase_id in range(1, self.num_phases + 1)}
        # self.spillback_info_map = {phase_id: {} for phase_id in range(1, self.num_phases + 1)}
        # self.block_map = {phase_id: False for phase_id in range(1, self.num_phases + 1)}
        return
    
    def _connectObjects(self, intersection):
        # set intersection
        self.intersection = intersection
        self.intersection.scoot_controller = self

        # set signal_controller and roads
        self.signal_controller = self.intersection.signal_controller
        self.roads = self.intersection.input_roads

        # set branch_length_map
        self.branch_info_map = {road_id: {} for road_id in self.roads.getKeys(container_flg=True)}
        for road_id, road in self.roads.items():
            if road.get('type') == 1:
                self.branch_info_map[road_id]['pos'] = road.right_connector.get('from_pos')
                self.branch_info_map[road_id]['length'] = {
                    'straight': road.main_link.get('length') - road.right_connector.get('from_pos'),
                    'right': road.right_connector.get('length') - road.right_connector.get('to_pos') + road.right_link.get('length'),
                }
            else:
                raise NotImplementedError(f"Not supported road type: {road.get('type')}")
            
        # set link_phase_map
        self.link_phase_map = {}
        for road_id, road in self.roads.items():
            if road.get('type') == 1:
                self.link_phase_map[road.main_link.get('id')] = 1 if road_id in [self.NORTH_ROAD_ID, self.SOUTH_ROAD_ID] else 2
                self.link_phase_map[road.right_link.get('id')] = 3 if road_id in [self.NORTH_ROAD_ID, self.SOUTH_ROAD_ID] else 4
                self.link_phase_map[road.right_connector.get('id')] = 3 if road_id in [self.NORTH_ROAD_ID, self.SOUTH_ROAD_ID] else 4
            else:
                raise NotImplementedError(f"Not supported road type: {road.get('type')}")

        # set effective_storage_length_map
        self.effective_storage_length_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        for road_id, road in self.roads.items():
            effective_storage_length_map = road.get('effective_storage_length_map')
            if road_id in [self.NORTH_ROAD_ID, self.SOUTH_ROAD_ID]:
                self.effective_storage_length_map[1] += effective_storage_length_map['left'] + effective_storage_length_map['straight']
                self.effective_storage_length_map[3] += effective_storage_length_map['right']
            elif road_id in [self.EAST_ROAD_ID, self.WEST_ROAD_ID]:
                self.effective_storage_length_map[2] += effective_storage_length_map['left'] + effective_storage_length_map['straight']
                self.effective_storage_length_map[4] += effective_storage_length_map['right']
            else:
                raise NotImplementedError(f"Not supported road order ID: {road_id}")
        
        # set initial phases in signal_controller
        self.signal_controller.setPhases([self.first_partition['phase']['from']] * self.first_partition['steps'])
        return

    def _updateTrafficInfo(self):
        # update num_vehs_record
        tmp_num_vehicles = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
        for road_id, road in self.roads.items():
            route_num_vehs_map = road.get('route_num_vehs_map')
            if road_id in [self.NORTH_ROAD_ID, self.SOUTH_ROAD_ID]:
                tmp_num_vehicles[1] += route_num_vehs_map[1] + route_num_vehs_map[2] # add number of left-turning and straight-going vehicles to phase 1
                tmp_num_vehicles[3] += route_num_vehs_map[3] # add number of right-turning vehicles to phase 3
            elif road_id in [self.EAST_ROAD_ID, self.WEST_ROAD_ID]:
                tmp_num_vehicles[2] += route_num_vehs_map[1] + route_num_vehs_map[2] # add number of left-turning and straight-going vehicles to phase 2
                tmp_num_vehicles[4] += route_num_vehs_map[3] # add number of right-turning vehicles to phase 4
            else:
                raise NotImplementedError(f"Not supported road order ID: {road_id}")
        self.num_vehs_record.append(tmp_num_vehicles)

        # update saturation_record
        self.saturation_record.append({
            phase_id: tmp_num_vehicles[phase_id] / self.effective_storage_length_map[phase_id] for phase_id in range(1, self.num_phases + 1)
        })

        # update phase_saturation_map and phase_num_vehicles_map
        if len(self.num_vehs_record) == 1:
            for phase_id in range(1, self.num_phases + 1):
                self.saturation_map[phase_id] = self.saturation_record[-1][phase_id]
                self.num_vehicles_map[phase_id] = self.num_vehs_record[-1][phase_id]

        elif len(self.num_vehs_record) <= self.params['cycle']:
            for phase_id in range(1, self.num_phases + 1):
                self.saturation_map[phase_id] = (self.saturation_map[phase_id] * (len(self.num_vehs_record) - 1) + self.saturation_record[-1][phase_id]) / len(self.num_vehs_record)
                self.num_vehicles_map[phase_id] = (self.num_vehicles_map[phase_id] * (len(self.num_vehs_record) - 1) + self.num_vehs_record[-1][phase_id]) / len(self.num_vehs_record)

        else:
            for phase_id in range(1, self.num_phases + 1):
                self.saturation_map[phase_id] += (self.saturation_record[-1][phase_id] - self.saturation_record[-1 - self.params['cycle']][phase_id]) / self.params['cycle']
                self.num_vehicles_map[phase_id] += (self.num_vehs_record[-1][phase_id] - self.num_vehs_record[-1 - self.params['cycle']][phase_id]) / self.params['cycle']
        
        # update total_num_vehicles and avg_saturation
        self.total_num_vehicles = sum(self.num_vehicles_map.values())
        self.avg_saturation = 0.0
        if self.total_num_vehicles != 0:
            for phase_id in range(1, self.num_phases + 1):
                self.avg_saturation += self.saturation_map[phase_id] * self.num_vehicles_map[phase_id]
            self.avg_saturation /= self.total_num_vehicles

        if not self.split_update_flg:
            return
        
        # update vehicle_size
        num_vehs = 0
        sum_length = 0.0
        for road in self.roads.getAll():
            vehicles_df = road.get('vehicles_df')
            if vehicles_df.shape[0] == 0:
                continue

            sum_length += vehicles_df['length'].sum()
            num_vehs += vehicles_df.shape[0]
        self.vehicle_size = sum_length / num_vehs if num_vehs != 0 else self.INITIAL_VEHICLE_SIZE
        
        # udpate blocked_info_map
        self.blocked_info_map[self.current_phase] = {}
        for road_id, road in self.roads.items():
            if road.get('type') == 1:
                if self.current_phase in [1, 2] and self.link_phase_map[road.main_link.get('id')] == self.current_phase:
                    vehicles_df = road.get('vehicles_df')
                    num_vehs = vehicles_df[vehicles_df['link_id'].isin([road.right_link.get('id'), road.right_connector.get('id')])].shape[0]
                    self.blocked_info_map[self.current_phase][road_id] = num_vehs * (self.vehicle_size + self.MIN_DISTANCE) >= self.branch_info_map[road_id]['length']['right'] * self.SPILLBACK_THRESHOLD
                
                elif self.current_phase in [3, 4] and self.link_phase_map[road.right_link.get('id')] == self.current_phase:
                    vehicles_df = road.get('vehicles_df')
                    num_vehs = vehicles_df[
                        (vehicles_df['link_id'] == road.main_link.get('id')) &
                        (vehicles_df['position'] >= self.branch_info_map[road_id]['pos'])
                    ].shape[0]
                    self.blocked_info_map[self.current_phase][road_id] = num_vehs * (self.vehicle_size + self.MIN_DISTANCE) >= self.branch_info_map[road_id]['length']['straight'] * self.SPILLBACK_THRESHOLD

            else:
                raise NotImplementedError(f"Not supported road type: {road.get('type')}")

        # # update empty_info_map
        # if self.current_phase in [1, 2]:
        #     self.empty_info_map[self.current_phase] = {}
        #     self.empty_info_map[self.next_phase] = {}
        # else:
        #     self.empty_info_map[self.current_phase] = {}

        # for road_id, road in self.roads.items(): 
        #     if road.get('type') == 1:
        #         # set empty information
        #         vehicles_df = road.get('vehicles_df').copy()
        #         if self.link_phase_map[road.main_link.get('id')] == self.current_phase:
        #             self.empty_info_map[self.current_phase][road_id] = vehicles_df[
        #                 (vehicles_df['link_id'] == road.main_link.get('id')) |
        #                 (vehicles_df['position'] >= self.branch_pos_map[road_id])
        #             ].shape[0] == 0
        #             self.empty_info_map[self.next_phase][road_id] = vehicles_df[
        #                 (vehicles_df['link_id'] == road.right_link.get('id')) |
        #                 (vehicles_df['link_id'] == road.right_connector.get('id'))
        #             ].shape[0] == 0
        #         elif self.link_phase_map[road.right_link.get('id')] == self.current_phase:
        #             self.empty_info_map[self.current_phase][road_id] = vehicles_df[
        #                 (vehicles_df['link_id'] == road.right_link.get('id')) | 
        #                 (vehicles_df['link_id'] == road.right_connector.get('id'))
        #             ].shape[0] == 0
        #     else:
        #         raise NotImplementedError(f"Not supported road type: {road.get('type')}")
            
        # # update spillback_info_map
        # if self.current_phase in [1, 2]:
        #     self.spillback_info_map[self.current_phase] = {road_id: False for road_id in self.empty_info_map[self.current_phase].keys()}
        #     self.spillback_info_map[self.next_phase] = {road_id: False for road_id in self.empty_info_map[self.next_phase].keys()}
        # else:  
        #     self.spillback_info_map[self.current_phase] = {road_id: False for road_id in self.empty_info_map[self.current_phase].keys()}

        # for road_id, road in self.roads.items():
        #     if road.get('type') == 1:
        #         if road_id not in self.empty_info_map[self.current_phase].keys():
        #             continue

        #         if self.current_phase in [1, 2]:
        #             # Check spillback for right turn
        #             vehicles_df = road.main_link.get('vehicles_df').copy()  
        #             vehicles_df = vehicles_df[
        #                 (vehicles_df['in_queue'] == True) &
        #                 (vehicles_df['route_id'].isin([self.TURN_RIGHT_ID]))
        #             ].sort_values(by='position', ascending=True)

        #             if vehicles_df.shape[0] != 0 and vehicles_df.iloc[0]['position'] <= self.branch_pos_map[road_id]:
        #                 self.spillback_info_map[self.current_phase][road_id] = True
                
        #             # Check spillback for left turn and straight
        #             vehicles_df = road.main_link.get('vehicles_df').copy()
        #             vehicles_df = vehicles_df[
        #                 (vehicles_df['in_queue'] == True) &
        #                 (vehicles_df['route_id'].isin([self.TURN_LEFT_ID, self.GO_STRAIGHT_ID]))
        #             ].sort_values(by='position', ascending=True)

        #             if vehicles_df.shape[0] != 0 and vehicles_df.iloc[0]['position'] <= self.branch_pos_map[road_id]:
        #                 self.spillback_info_map[self.next_phase][road_id] = True
                
        #         else:
        #             # Check spillback for right turn
        #             vehicles_df = road.right_link.get('vehicles_df').copy()  
        #             vehicles_df = vehicles_df[
        #                 (vehicles_df['in_queue'] == True) &
        #                 (vehicles_df['route_id'].isin([self.TURN_RIGHT_ID]))
        #             ].sort_values(by='position', ascending=True)

        #             if vehicles_df.shape[0] != 0 and vehicles_df.iloc[0]['position'] <= self.branch_pos_map[road_id]:
        #                 self.spillback_info_map[self.current_phase][road_id] = True


        #     else:
        #         raise NotImplementedError(f"Not supported road type: {road.get('type')}")
            
        # if self.current_phase in [1, 2]:
        #     self.block_map[self.current_phase] = any([
        #         self.empty_info_map[self.current_phase][road_id] and self.spillback_info_map[self.current_phase][road_id] for road_id in self.empty_info_map[self.current_phase].keys()
        #     ])
        #     self.block_map[self.next_phase] = any([
        #         self.empty_info_map[self.next_phase][road_id] and self.spillback_info_map[self.next_phase][road_id] for road_id in self.empty_info_map[self.next_phase].keys()
        #     ])

        #     if self.block_map[self.current_phase] or self.block_map[self.next_phase]:
        #         print('test')
        # else:
        #     self.block_map[self.current_phase] = any([
        #         self.empty_info_map[self.current_phase][road_id] and self.spillback_info_map[self.current_phase][road_id] for road_id in self.empty_info_map[self.current_phase].keys()
        #     ])

        #     if self.block_map[self.current_phase]:
        #         print('test')
        
        return 
    
    def _updateSplit(self):

        if any(self.blocked_info_map[self.current_phase].values()):
            self._decrementSplit()
            self._showInfo('blocked')
        elif self.saturation_map[self.current_phase] < self.saturation_map[self.next_phase]:
            self._decrementSplit()
        else:
            self._incrementSplit()
        return
    
    def _decrementSplit(self):
        if self.params['split'][self.current_phase] >= self.change_steps['split'] + self.min_split:
            change_steps = self.change_steps['split']
        else:
            change_steps = self.params['split'][self.current_phase] - self.min_split

        if change_steps <= 0: return

        self.first_partition['steps'] -= change_steps
        self.first_partition['fixed'] = True

        self.params['split'][self.current_phase] -= change_steps
        self.params['split'][self.next_phase] += change_steps

        self.signal_controller.deletePhases(type='end', steps=change_steps)
        return
    
    def _incrementSplit(self):
        if self.params['split'][self.next_phase] >= self.change_steps['split'] + self.min_split:
            change_steps = self.change_steps['split']
        else:
            change_steps = self.params['split'][self.next_phase] - self.min_split
        
        if change_steps <= 0: return

        self.first_partition['steps'] += change_steps
        self.first_partition['fixed'] = True

        self.params['split'][self.current_phase] += change_steps
        self.params['split'][self.next_phase] -= change_steps

        self.signal_controller.setPhases([self.current_phase] * change_steps)
        return
    
    def _updateCycle(self):
        if self.avg_saturation < self.saturation_threshold:
            if self.first_partition['steps'] == 0:
                return
            
            # update split information
            cumulative_change_steps = 0
            for partition_id, partition in enumerate(self.remain_steps_info['split']):
                if self.params['split'][partition['phase']['from']] >= self.change_steps['cycle'] + self.min_split:
                    change_steps = self.change_steps['cycle']
                else:
                    change_steps = self.params['split'][partition['phase']['from']] - self.min_split
                
                cumulative_change_steps += change_steps

                partition['steps'] -= cumulative_change_steps
                self.params['split'][partition['phase']['from']] -= change_steps

                if partition_id == 0:
                    self.signal_controller.deletePhases(type='end', steps=change_steps)

            # update cycle information
            self.params['cycle'] -= cumulative_change_steps

        else:
            # get phase_change_map
            phase_change_map = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
            phase_order_list = sorted(range(1, self.num_phases + 1), key=lambda x: self.saturation_map[x], reverse=True)
            for phase_order_id in range(min(self.max_cycle - self.params['cycle'], self.change_steps['cycle'] * self.num_phases)):
                if any(self.blocked_info_map[phase_order_list[phase_order_id % self.num_phases]]):
                    continue
                phase_change_map[phase_order_list[phase_order_id % self.num_phases]] += 1
            
            # update split information
            cumulative_change_steps = 0
            for partition_id, partition in enumerate(self.remain_steps_info['split']):
                cumulative_change_steps += phase_change_map[partition['phase']['from']]

                partition['steps'] += cumulative_change_steps
                self.params['split'][partition['phase']['from']] += phase_change_map[partition['phase']['from']]

                if partition_id == 0:
                    self.signal_controller.setPhases([partition['phase']['from']] * phase_change_map[partition['phase']['from']])

            # update cycle information
            self.params['cycle'] += cumulative_change_steps    
        return

    def _proceedOneStep(self):
        # update split information
        if self.first_partition['steps'] == 0:
            last_partition = self.remain_steps_info['split'].popleft()
            last_partition['steps'] = self.params['cycle']
            last_partition['fixed'] = False
            self.remain_steps_info['split'].append(last_partition)

            self.signal_controller.setPhases([self.first_partition['phase']['from']] * self.first_partition['steps'])
        
        for partition in self.remain_steps_info['split']:
            partition['steps'] -= 1

        # update cycle information
        if self.remain_steps_info['cycle'] == 0:
            self.remain_steps_info['cycle'] = self.params['cycle']
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
        
        elif type == 'blocked':
            print('status: current phase is empty and blocked')
            print(f"intersection id: {self.id}")
            print(f"current_phase: {self.current_phase}")
            for road_id, blocked_flg in self.blocked_info_map[self.current_phase].items():
                print(f"road {road_id}: blocked_flg = {blocked_flg}")

        else:
            raise NotImplementedError(f"Not supported type: {type}")
    
    def update(self):
        self._updateTrafficInfo()

        if self.cycle_update_flg:
            self._updateCycle()
            self._showInfo('update')
            self.previous_params = copy.deepcopy(self.params)

        if self.split_update_flg:
            self._updateSplit()

        self._proceedOneStep()
        return