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

    PHASE_ROAD_LIST_MAP = {
        1: [NORTH_ROAD_ID, SOUTH_ROAD_ID],
        2: [EAST_ROAD_ID, WEST_ROAD_ID],
        3: [NORTH_ROAD_ID, SOUTH_ROAD_ID],
        4: [EAST_ROAD_ID, WEST_ROAD_ID],    
    }

    STRAIGHT_PHASE_LIST = [1, 2]
    RIGHT_PHASE_LIST = [3, 4]

    MIN_DISTANCE = 1.0

    TURN_LEFT_ID = 1
    GO_STRAIGHT_ID = 2
    TURN_RIGHT_ID = 3

    INITIAL_FLOW_RATE = 0.5
    SPILLBACK_THRESHOLD = 0.8

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
        return (self.first_partition['steps'] == self.change_steps['split']['normal'] and not self.first_partition['fixed'])

    @property
    def current_blocked_flg(self):
        return any(self.blocked_info_map[self.current_phase].values())
    
    @property
    def next_blocked_flg(self):
        return any(self.blocked_info_map[self.next_phase].values())
    
    @property
    def max_saturation(self):
        return max(self.saturation_map.values())

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

        # initialize other properties
        self.max_outflow_rate_map = {phase_id: self.INITIAL_FLOW_RATE for phase_id in range(1, self.num_phases + 1)}
        self.inflow_rate_record_map = {phase_id: pd.DataFrame(columns=['time', 'inflow_rate']) for phase_id in range(1, self.num_phases + 1)}
        self.outflow_rate_record_map = {phase_id: pd.DataFrame(columns=['time', 'outflow_rate']) for phase_id in range(1, self.num_phases + 1)}
        self.saturation_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        self.blocked_info_map = {phase_id: {} for phase_id in range(1, self.num_phases + 1)}
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

        # set branch_length_map and link_phase_map
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
        
        return

    def _updateTrafficInfo(self):
        # udpate blocked_info_map
        self.blocked_info_map[self.current_phase] = {}
        for road_id, road in self.roads.items():
            if road.get('type') == 1:
                if road_id not in self.PHASE_ROAD_LIST_MAP[self.current_phase]:
                    continue

                vehicles_df = road.get('vehicles_df')
                vehicle_size = vehicles_df['length'].mean()

                if self.current_phase in self.STRAIGHT_PHASE_LIST:
                    num_vehs = vehicles_df[vehicles_df['link_id'].isin([road.right_link.get('id'), road.right_connector.get('id')])].shape[0]
                    self.blocked_info_map[self.current_phase][road_id] = num_vehs * (vehicle_size + self.MIN_DISTANCE) >= self.branch_info_map[road_id]['length']['right'] * self.SPILLBACK_THRESHOLD
                
                elif self.current_phase in self.RIGHT_PHASE_LIST:
                    num_vehs = vehicles_df[
                        (vehicles_df['link_id'] == road.main_link.get('id')) &
                        (vehicles_df['position'] >= self.branch_info_map[road_id]['pos'])
                    ].shape[0]
                    self.blocked_info_map[self.current_phase][road_id] = num_vehs * (vehicle_size + self.MIN_DISTANCE) >= self.branch_info_map[road_id]['length']['straight'] * self.SPILLBACK_THRESHOLD

        if self.split_update_flg:
            return
        
        # update inflow_rate_record_map and outflow_rate_record_map
        inflow_rate_record = self.inflow_rate_record_map[self.current_phase]
        inflow_rate = 0.0
        for road_id in self.PHASE_ROAD_LIST_MAP[self.current_phase]:
            road = self.roads[road_id]
            for data_collection_point in road.data_collection_points.getAll():
                if data_collection_point.get('type') != 'input':
                    continue

                tmp_inflow_rate = data_collection_point.getFlowRate(duration_step=self.params['cycle'] - self.change_steps['split']['normal'])
            
            turn_ratios = road.get('turn_ratios')
            if self.current_phase in self.STRAIGHT_PHASE_LIST:
                tmp_inflow_rate *= (turn_ratios[self.TURN_LEFT_ID] + turn_ratios[self.GO_STRAIGHT_ID]) / sum(turn_ratios.values()) 
            elif self.current_phase in self.RIGHT_PHASE_LIST:
                tmp_inflow_rate *= turn_ratios[self.TURN_RIGHT_ID] / sum(turn_ratios.values())
            else:
                raise NotImplementedError(f"Not supported phase id: {self.current_phase}")

            inflow_rate += tmp_inflow_rate

        inflow_rate_record.loc[len(inflow_rate_record)] = {
            'time': int(self.network.get('current_time')),
            'inflow_rate': inflow_rate
        }

        outflow_rate_record = self.outflow_rate_record_map[self.current_phase]
        outflow_rate = 0.0
        for road_id in self.PHASE_ROAD_LIST_MAP[self.current_phase]:
            road = self.roads[road_id]
            tmp_outflow_rate = 0.0
            for data_collection_point in road.data_collection_points.getAll():
                if data_collection_point.get('type') != 'intersection':
                    continue

                tmp_outflow_rate += data_collection_point.getFlowRate(duration_step=self.params['split'][self.current_phase] - self.change_steps['split']['normal'])

            outflow_rate += tmp_outflow_rate

        outflow_rate_record.loc[len(outflow_rate_record)] = {
            'time': int(self.network.get('current_time')),
            'outflow_rate': outflow_rate
        }

        # update max_outflow_rate_map
        self.max_outflow_rate_map[self.current_phase] = max(self.max_outflow_rate_map[self.current_phase], outflow_rate)

        # update saturation_map
        self.saturation_map[self.current_phase] = inflow_rate * self.params['cycle'] / (self.max_outflow_rate_map[self.current_phase] * self.params['split'][self.current_phase])
        return 
    
    def _updateSplit(self):
        if self.first_partition['fixed']:
            return
        
        if self.current_blocked_flg:
            self._decrementSplit(type='blocked')
            self._showInfo('blocked', 'current')
        elif (self.current_phase in self.STRAIGHT_PHASE_LIST) and self.next_blocked_flg:
            self._incrementSplit(type='blocked')
            self._showInfo('blocked', 'next')
        elif self.saturation_map[self.current_phase] < self.saturation_map[self.next_phase]:
            self._decrementSplit(type='normal')
        else:
            self._incrementSplit(type='normal')

        return
    
    def _decrementSplit(self, type):
        if type == 'normal':
            change_steps = min(self.change_steps['split']['normal'], self.params['split'][self.current_phase] - self.min_split)
        elif type == 'blocked':
            change_steps = min(self.change_steps['split']['blocked'], self.params['split'][self.current_phase] - self.min_split)
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
        if self.max_saturation < 0.8:
            phase_change_map = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
            for phase_id in range(1, self.num_phases + 1):
                if self.saturation_map[phase_id] > 0.8:
                    continue

                change_steps = min(self.change_steps['cycle'], self.params['split'][phase_id] - self.min_split)

                if change_steps <= 0: continue
                
                if self.first_partition['phase']['from'] == phase_id:
                    phase_change_map[phase_id] = min(change_steps, self.first_partition['steps'])
                else:
                    phase_change_map[phase_id] = change_steps 

            # update split information
            cumulative_change_steps = 0
            for partition_id, partition in enumerate(self.remain_steps_info['split']):
                cumulative_change_steps += phase_change_map[partition['phase']['from']]

                partition['steps'] -= cumulative_change_steps
                self.params['split'][partition['phase']['from']] -= phase_change_map[partition['phase']['from']]
                
                if partition_id != 0:
                    continue

                if partition['steps'] <= self.change_steps['split']['normal']:
                    partition['fixed'] = True

                if phase_change_map[partition['phase']['from']] > 0:
                    self.signal_controller.deletePhases(type='end', steps=phase_change_map[partition['phase']['from']])

            # update cycle information
            self.params['cycle'] -= cumulative_change_steps

        elif self.max_saturation > 0.9:
            # get phase_change_map
            phase_change_map = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
            prioritize_phase_list = sorted(range(1, self.num_phases + 1), key=lambda x: self.saturation_map[x], reverse=True)
            cumurative_change_steps = 0
            for phase_id in prioritize_phase_list:
                if self.saturation_map[phase_id] < 0.9:
                    continue

                change_steps = min(self.max_cycle - self.params['cycle'] - cumurative_change_steps, self.change_steps['cycle'])

                if change_steps <= 0: continue

                phase_change_map[phase_id] = change_steps
                cumurative_change_steps += change_steps
                
            # update split information
            cumulative_change_steps = 0
            for partition_id, partition in enumerate(self.remain_steps_info['split']):
                cumulative_change_steps += phase_change_map[partition['phase']['from']]

                partition['steps'] += cumulative_change_steps
                self.params['split'][partition['phase']['from']] += phase_change_map[partition['phase']['from']]

                if partition_id == 0 and phase_change_map[partition['phase']['from']] > 0:
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

        else:
            raise NotImplementedError(f"Not supported type: {type}")
    
    def update(self):
        self._updateTrafficInfo()

        if self.cycle_update_flg:
            self._updateCycle()
            self._showInfo('update')
            self.previous_params = copy.deepcopy(self.params)

        if not self.first_partition['fixed']:
            self._proceedOneStep()
            return
        
        if self.split_update_flg or self.current_blocked_flg:
            self._updateSplit()

        self._proceedOneStep()
        return