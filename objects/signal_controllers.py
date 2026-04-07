from libs.container import Container
from libs.object import Object
from objects.signal_heads import SignalHeads
from collections import deque
import pandas as pd
import copy

class SignalControllers(Container):
    def __init__(self, network):
        super().__init__()
        
        # set objects
        self.config = network.config
        self.network = network
        self.executor = network.executor
        
        # set com object
        self.com = self.network.com.SignalControllers

        # initialize signal_controller objects
        for signal_controller_com in self.com.GetAll():
            self.add(SignalController(signal_controller_com, self))
        
        return
    
    def setNextPhaseToVissim(self):
        for signal_controller in self.getAll():
            signal_controller.setNextPhaseToVissim()
        
    def syncDataFrame(self):
        for signal_controller in self.getAll():
            self.executor.submit(signal_controller.syncDataFrame)
        self.executor.wait()
        return

    def lastUpdate(self):
        for signal_controller in self.getAll():
            self.executor.submit(signal_controller.lastUpdate)
        return
class SignalController(Object):
    def __init__(self, com, signal_controllers):
        super().__init__()

        # set objects
        self.config = signal_controllers.config
        self.signal_controllers = signal_controllers
        self.network = signal_controllers.network
        self.executor = signal_controllers.executor

        # set com object
        self.com = com

        # set properties
        self._initProps()

        # set signal_groups object
        self.signal_groups = SignalGroups(self)
        
        # set intersection object
        self._initIntersection()

        # set phases
        self._initPhases()
        return

    def _initProps(self):
        # set id
        self.id = int(self.com.AttValue('No'))

        simulation_info = self.config.get('simulator_info')

        # set num_red_steps
        self.num_red_steps = simulation_info['num_red_steps']

        # initialize future_phase_ids
        if simulation_info['control_method'] in ['drl', 'bc']:
            drl_info = self.config.get('drl_info')
            self.future_phase_ids = deque(maxlen=drl_info['duration_steps'] + 1)
        elif simulation_info['control_method'] == 'mpc':
            mpc_info = self.config.get('mpc_info')
            self.future_phase_ids = deque(maxlen=mpc_info['remained_steps'] + mpc_info['utilize_steps'])
        elif simulation_info['control_method'] == 'scoot':
            scoot_info = self.config.get('scoot_info')
            self.future_phase_ids = deque(maxlen=scoot_info['max_cycle'] - 3 * scoot_info['min_split'])
        else:
            raise NotImplementedError(f"Not supported control method: {simulation_info['control_method']}")

        # initialize phase_record_df
        self.record_list = []
        self.record_df = None
        return

    def _initIntersection(self):
        # set intersection object
        input_road_list = []
        for signal_group in self.signal_groups.getAll():
            input_road_list.append(signal_group.road.get('id'))
        input_road_list = sorted(list(set(input_road_list)))

        found_flg = False
        for self.intersection in self.network.intersections.getAll():
            tmp_input_roads = self.intersection.input_roads
            if input_road_list == sorted(tmp_input_roads.getMultiAttValues('id')):
                found_flg = True
                break
        
        if not found_flg:
            raise Exception(f"SignalController {self.get('id')} could not find a matching intersection for input roads {input_road_list}.")
        
        # set signal_controller object to intersection object
        self.intersection.set('signal_controller', self)
        return

    def _initPhases(self):
        # get phases_df
        num_roads_phases_map = self.config.get('num_roads_phases_map')
        num_roads = self.intersection.get('num_roads')
        phases_df = num_roads_phases_map[num_roads]

        # initialize phases
        self.phases = {}
        for _, phase_row in phases_df.iterrows():
            tmp_signal_group_ids = []
            for signal_group_order in range(1, num_roads + 1):
                tmp_signal_group_ids.append(int(phase_row[f"signal_group{signal_group_order}"]))
            
            self.phases[int(phase_row['id'])] = tmp_signal_group_ids
        
        self.num_phases = len(self.phases)
        return
        
    def setNextPhases(self, phase_ids):
        # add to future_phase_ids
        self.future_phase_ids.extend(phase_ids)

        # add to signal_groups
        self.signal_groups.setNextPhases(phase_ids)
        return
    
    def deletePhases(self, type, steps):
        # typeによって分岐
        if type == 'end':
            for _ in range(steps):
                self.future_phase_ids.pop()

        elif type == 'start':
            for _ in range(steps):  
                self.future_phase_ids.popleft()

        self.signal_groups.deletePhases(type, steps)
        return
    
    def setNextPhaseToVissim(self):
        # set next phase to vissim
        self.signal_groups.setNextPhaseToVissim()

        # update record_list
        self.record_list.append({
            'time': int(self.network.get('current_time')),
            'value': int(self.next_phase_id),
        })

        # remove the first phase from future_phase_ids
        self.future_phase_ids.popleft()
        return

    def lastUpdate(self):
        self.record_list.append({
            'time': int(self.network.get('current_time')),
            'value': int(self.next_phase_id) if self.next_phase_id is not None else self.record_list[-1]['value'],
        })
        return

    def syncDataFrame(self):
        self.record_df = pd.DataFrame(self.record_list)
        return
    
    @property
    def next_phase_id(self):
        return self.future_phase_ids[0] if self.future_phase_ids else None

    @property
    def current_phase_id(self):
        if len(self.record_list) > 0:
            return self.record_list[-1]['value']
        else:
            return 0

    @property
    def signal_change_flg(self):
        if self.current_phase_id is None:
            return False
        
        if self.current_phase_id == self.future_phase_ids[0]:
            return False
    
        return True
    
    @property
    def remaining_steps(self):
        return len(self.future_phase_ids)

class SignalGroups(Container):
    def __init__(self, upper_object):
        super().__init__()

        # set objects
        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'SignalController':
            # set signal_controller and network
            self.signal_controller = upper_object
            self.network = self.signal_controller.network

            # set com object
            self.com = self.signal_controller.com.SGs

            # initialize signal_group
            self._initElements()

            # connect to signal_head and road objects
            self._makeSignalHeadConnections()
            self._makeRoadConnections()
        
        elif upper_object.__class__.__name__ == 'Road':
            # set road and network
            self.road = upper_object
            self.network = self.road.network

        else:
            raise NotImplementedError(f"Not supported upper object type: {upper_object.__class__.__name__}")
        
        return
    
    def _initElements(self):
        for signal_group_com in self.com.GetAll():
            self.add(SignalGroup(signal_group_com, self))
        return
    
    def _makeSignalHeadConnections(self):
        for signal_group in self.getAll():
            signal_heads = signal_group.signal_heads
            
            for signal_head_com in signal_heads.com.GetAll():
                signal_head_id = int(signal_head_com.AttValue('No'))
                signal_heads.add(self.network.signal_heads[signal_head_id])
        return
    
    def _makeRoadConnections(self):
        for signal_group in self.getAll():
            signal_heads = signal_group.signal_heads

            possible_road_ids = []
            for signal_head in signal_heads.getAll():
                connector = signal_head.connector
                from_link = connector.from_links.getAll()[0]
                road = from_link.road
                possible_road_ids.append(road.get('id'))

            if len(set(possible_road_ids)) == 1:
                signal_group.set('road', road)
                road.signal_groups.add(signal_group)
                
                # direction_signal_groups_mapに保存
                direction_signal_group_map = road.get('direction_signal_group_map')
                direction_signal_group_map[signal_group.direction_id] = signal_group.get('id')
            
            else:
                raise Exception(f"SignalGroup {signal_group.get('id')} has multiple possible roads: {possible_road_ids}. Please check the signal head connections.")       

    def setNextPhases(self, phase_ids):
        # フェーズのリストを取得
        phases = self.signal_controller.get('phases')
        
        # 各フェーズに対応するSignalGroupの値を計算
        sig_color_list = []
        for phase_id in phase_ids:
            signal_group_ids = phases[phase_id]
            tmp_sig_color_list = [1] * self.count()  # 1は赤信号を示す
            for signal_group_id in signal_group_ids:
                tmp_sig_color_list[signal_group_id - 1] = 3
            sig_color_list.append(tmp_sig_color_list)

        # 将来の信号現示を保存する（赤➡青の変化時は全赤の時間があるため，赤を１ステップ追加する）
        for signal_group in self.getAll():
            tmp_sig_color_list = [tmp_row[signal_group.get('id') - 1] for tmp_row in sig_color_list]
            signal_group.setNextPhases(tmp_sig_color_list)
    
    def deletePhases(self, type, steps):
        for signal_group in self.getAll():
            self.executor.submit(signal_group.deletePhases, type, steps)
        self.executor.wait()
        return

    def setNextPhaseToVissim(self):
        for signal_group in self.getAll():
            signal_group.setNextPhaseToVissim()
            
            
class SignalGroup(Object):
    # signal color to value mapping
    RED = 1
    GREEN = 3
    def __init__(self, com, signal_groups):
        super().__init__()

        # set objects
        self.config = signal_groups.config
        self.executor = signal_groups.executor
        self.signal_groups = signal_groups
        self.signal_controller = signal_groups.signal_controller
        self.network = signal_groups.network

        # set com object
        self.com = com
        
        # set properties
        self._initProps()

        # set signal_heads object
        self.signal_heads = SignalHeads(self)
        return

    def _initProps(self):
        # set id
        self.id = int(self.com.AttValue('No'))

        # set signal_colors_df
        self.record_list = []
        self.record_df = None

        # set future_signal_colors
        simulatior_info = self.config.get('simulator_info')
        if simulatior_info['control_method'] in ['drl', 'bc']:
            drl_info = self.config.get('drl_info')
            self.future_signal_colors = deque(maxlen=drl_info['duration_steps'] + 1) # +1は現在のフェーズを含むため
        elif simulatior_info['control_method'] == 'mpc':
            mpc_info = self.config.get('mpc_info')
            self.future_signal_colors = deque(maxlen=mpc_info['remained_steps'] + mpc_info['utilize_steps'])
        elif simulatior_info['control_method'] == 'scoot':
            scoot_info = self.config.get('scoot_info')
            self.future_signal_colors = deque(maxlen=scoot_info['max_cycle'] - 3 * scoot_info['min_split'])
        else:
            raise NotImplementedError(f"Not supported control method: {simulatior_info['control_method']}")
        
        return

    def setNextPhases(self, sig_color_list):
        # get previous signal color
        if self.future_signal_colors:
            previous_signal_color = self.future_signal_colors[-1]
        elif len(self.record_list) > 0:
            if self.record_list[-1]['value'] == 'R':
                previous_signal_color = self.RED
            elif self.record_list[-1]['value'] == 'G':
                previous_signal_color = self.GREEN
            else:
                raise NotImplementedError(f"Not supported signal color string: {self.record_list[-1]['value']}")
        else:
            previous_signal_color = None

        for step, signal_color in enumerate(copy.deepcopy(sig_color_list)):
            # if no records, no need to change signal color
            if previous_signal_color is None:
                continue
            
            # if red to green transition, insert a red phase before green
            if previous_signal_color == self.RED and signal_color == self.GREEN:
                sig_color_list[step] = self.RED
            
            # update previous signal color
            previous_signal_color = signal_color

        # add to future_signal_colors
        self.future_signal_colors.extend(sig_color_list)
        return

    def deletePhases(self, type, steps):
        if type == 'end':
            for _ in range(steps):
                self.future_signal_colors.pop()
        elif type == 'start':
            for _ in range(steps):
                self.future_signal_colors.popleft()
        return

    def setNextPhaseToVissim(self):
        # update record_list
        signal_color = self.future_signal_colors.popleft()
        if signal_color == self.RED:
            signal_color_str = 'R'
        elif signal_color == self.GREEN:
            signal_color_str = 'G'
        else:
            raise NotImplementedError(f"Not supported signal color: {signal_color}")
        self.record_list.append({
            'time': int(self.network.get('current_time')),
            'value': signal_color_str,
        })

        # if the signal color does not change, no need to set the signal color to Vissim
        if len(self.record_list) > 1 and self.record_list[-2]['value'] == self.record_list[-1]['value']:
            return
        
        self.com.SetAttValue('SigState', self.current_signal_color)
        return
    
    @property
    def current_signal_color(self):
        if len(self.record_list) > 0:
            current_signal_color_str = self.record_list[-1]['value']
            if current_signal_color_str == 'R':
                return self.RED
            elif current_signal_color_str == 'G':
                return self.GREEN
            else:
                raise NotImplementedError(f"Not supported signal color string: {current_signal_color_str}")
        else:
            return self.RED

    @property
    def direction_id(self):
        direction_ids = []
        for signal_head in self.signal_heads.getAll():
            direction_ids.append(signal_head.get('direction_id'))
        
        if len(set(direction_ids)) != 1:
            raise Exception(f"SignalGroup {self.get('id')} has multiple direction IDs: {direction_ids}. Please check the signal head connections.")
        
        return direction_ids[0]
        
