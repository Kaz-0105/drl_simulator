from libs.container import Container
from libs.object import Object

from collections import deque

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
    NUM_PHASES = 4
    PHASE_ORDER_LIST = [1, 3, 2, 4]

    NORTH_ROAD_ID = 1
    EAST_ROAD_ID = 2
    SOUTH_ROAD_ID = 3
    WEST_ROAD_ID = 4

    def __init__(self, scoot_controllers, intersection, id):
        super().__init__()

        self.config = scoot_controllers.config
        self.executor = scoot_controllers.executor
        self.shared_resources = scoot_controllers.shared_resources
        self.scoot_controllers = scoot_controllers
        self.network = scoot_controllers.network

        self._initProps(id)
        self._connectObjects(intersection)
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
        self.num_phases = len(ScootController.PHASE_ORDER_LIST)

        # set params, remain_steps_info, change_steps, and thresholds
        scoot_info = self.config.get('scoot_info')
        self.params = {
            'cycle': scoot_info['initial_parameters']['cycle'],
            'split': {
                phase_id: scoot_info['initial_parameters']['split'][phase_id - 1] for phase_id in range(1, ScootController.NUM_PHASES + 1)
            }
        }

        self.remain_steps_info = {
            'cycle': self.params['cycle'],
            'split': deque(maxlen=ScootController.NUM_PHASES)
        }
        sum_steps = 0
        for id in range(len(ScootController.PHASE_ORDER_LIST)):
            sum_steps += self.params['split'][ScootController.PHASE_ORDER_LIST[id]]
            self.remain_steps_info['split'].append({
                'phase': {
                    'from': ScootController.PHASE_ORDER_LIST[id],
                    'to': ScootController.PHASE_ORDER_LIST[(id + 1) % ScootController.NUM_PHASES]
                },
                'steps': sum_steps,
                'fixed': False
            })

        self.change_steps = scoot_info['change_steps']
        self.max_cycle = scoot_info['max_cycle']
        self.min_split = scoot_info['min_split']

        self.spacing_threshold = scoot_info['spacing_threshold']
        self.saturation_threshold = 1 / self.spacing_threshold

        # set effective_storage_length_map (ScootController._connectObjects())
        self.effective_storage_length_map = None 

        # initialize other properties
        self.total_num_vehicles = None
        self.avg_saturation = None
        self.num_vehs_record = []
        self.saturation_record = []
        self.phase_saturation_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        self.phase_num_vehicles_map = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
        return
    
    def _connectObjects(self, intersection):
        # set intersection
        self.intersection = intersection
        self.intersection.scoot_controller = self

        # set signal_controller and roads
        self.signal_controller = self.intersection.signal_controller
        self.roads = self.intersection.input_roads

        # set effective_storage_length_map
        self.effective_storage_length_map = {phase_id: 0.0 for phase_id in range(1, self.num_phases + 1)}
        for road_id, road in self.roads.items():
            effective_storage_length_map = road.get('effective_storage_length_map')
            if road_id in [ScootController.NORTH_ROAD_ID, ScootController.SOUTH_ROAD_ID]:
                self.effective_storage_length_map[1] += effective_storage_length_map['left'] + effective_storage_length_map['straight']
                self.effective_storage_length_map[3] += effective_storage_length_map['right']
            elif road_id in [ScootController.EAST_ROAD_ID, ScootController.WEST_ROAD_ID]:
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
            if road_id in [ScootController.NORTH_ROAD_ID, ScootController.SOUTH_ROAD_ID]:
                tmp_num_vehicles[1] += route_num_vehs_map[1] + route_num_vehs_map[2] # add number of left-turning and straight-going vehicles to phase 1
                tmp_num_vehicles[3] += route_num_vehs_map[3] # add number of right-turning vehicles to phase 3
            elif road_id in [ScootController.EAST_ROAD_ID, ScootController.WEST_ROAD_ID]:
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
                self.phase_saturation_map[phase_id] = self.saturation_record[-1][phase_id]
                self.phase_num_vehicles_map[phase_id] = self.num_vehs_record[-1][phase_id]

        elif len(self.num_vehs_record) <= self.params['cycle']:
            for phase_id in range(1, self.num_phases + 1):
                self.phase_saturation_map[phase_id] = (self.phase_saturation_map[phase_id] * (len(self.num_vehs_record) - 1) + self.saturation_record[-1][phase_id]) / len(self.num_vehs_record)
                self.phase_num_vehicles_map[phase_id] = (self.phase_num_vehicles_map[phase_id] * (len(self.num_vehs_record) - 1) + self.num_vehs_record[-1][phase_id]) / len(self.num_vehs_record)

        else:
            for phase_id in range(1, self.num_phases + 1):
                self.phase_saturation_map[phase_id] += (self.saturation_record[-1][phase_id] - self.saturation_record[-1 - self.params['cycle']][phase_id]) / self.params['cycle']
                self.phase_num_vehicles_map[phase_id] += (self.num_vehs_record[-1][phase_id] - self.num_vehs_record[-1 - self.params['cycle']][phase_id]) / self.params['cycle']
        

        # get total_num_vehicles and avg_saturation
        self.total_num_vehicles = sum(self.phase_num_vehicles_map.values())
        self.avg_saturation = 0.0
        for phase_id in range(1, self.num_phases + 1):
            self.avg_saturation += self.phase_saturation_map[phase_id] * self.phase_num_vehicles_map[phase_id]
        self.avg_saturation /= self.total_num_vehicles
        return
    
    def _updateSplit(self):
        if self.phase_saturation_map[self.current_phase] > self.phase_saturation_map[self.next_phase]:
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

        else:
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

        elif self.avg_saturation > self.saturation_threshold and self.params['cycle'] + self.change_steps['cycle'] * self.num_phases <= self.max_cycle:
            # update split information
            cumulative_change_steps = 0
            for partition_id, partition in enumerate(self.remain_steps_info['split']):
                cumulative_change_steps += self.change_steps['cycle']

                partition['steps'] += cumulative_change_steps
                self.params['split'][partition['phase']['from']] += self.change_steps['cycle']

                if partition_id == 0:
                    self.signal_controller.setPhases([partition['phase']['from']] * self.change_steps['cycle'])

            # update cycle information
            self.params['cycle'] += cumulative_change_steps
        else:
            # get phase_change_map
            phase_change_map = {phase_id: 0 for phase_id in range(1, self.num_phases + 1)}
            phase_order_list = sorted(range(1, self.num_phases + 1), key=lambda x: self.phase_saturation_map[x] - self.saturation_threshold, reverse=True)
            for phase_order_id in range(self.max_cycle - self.params['cycle']):
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
    
    def update(self):
        self._updateTrafficInfo()

        if self.cycle_update_flg:
            self._updateCycle()

        if self.split_update_flg:
            self._updateSplit()

        self._proceedOneStep()
        return