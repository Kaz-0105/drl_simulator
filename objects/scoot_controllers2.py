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

    TURN_LEFT_ID = 1
    GO_STRAIGHT_ID = 2
    TURN_RIGHT_ID = 3

    INITIAL_FLOW_RATE = 0.2

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
        return (self.first_partition['steps'] == self.change_steps['split'] and not self.first_partition['fixed'])
    
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
        return

    def _updateTrafficInfo(self):
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

                tmp_inflow_rate = data_collection_point.getFlowRate(duration_step=self.params['cycle'] - self.change_steps['split'])
            
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

                if self.current_phase in self.STRAIGHT_PHASE_LIST and data_collection_point.get('route_id') == self.TURN_RIGHT_ID:
                    continue

                if self.current_phase in self.RIGHT_PHASE_LIST and data_collection_point.get('route_id') in [self.TURN_LEFT_ID, self.GO_STRAIGHT_ID]:
                    continue

                tmp_outflow_rate += data_collection_point.getFlowRate(duration_step=self.params['split'][self.current_phase] - self.change_steps['split'])

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
        if self.saturation_map[self.current_phase] < self.saturation_map[self.next_phase]:
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
        if self.max_saturation < 0.8:
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

        elif self.max_saturation > 0.9:
            # get phase_change_map
            phase_change_map = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
            phase_order_list = sorted(range(1, self.num_phases + 1), key=lambda x: self.saturation_map[x], reverse=True)
            for phase_order_id in range(min(self.max_cycle - self.params['cycle'], self.change_steps['cycle'] * self.num_phases)):
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