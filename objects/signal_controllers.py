from libs.container import Container
from libs.object import Object
from objects.signal_heads import SignalHeads
from collections import deque

class SignalControllers(Container):
    def __init__(self, network):
        # 継承
        super().__init__()
        
        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = network.config
        self.network = network
        
        # comオブジェクトを取得
        self.com = self.network.com.SignalControllers

        # 下位の紐づくオブジェクトを初期化
        self.makeElements()

        # intersectionオブジェクトと紐づける
        self.makeIntersectionConnections()

        # phasesを作成する
        self.makePhases()

    def makeElements(self):
        for signal_controller_com in self.com.GetAll():
            self.add(SignalController(signal_controller_com, self))
    
    def makeIntersectionConnections(self):
        for signal_controller in self.getAll():
            input_road_ids = []
            for signal_group in signal_controller.signal_groups.getAll():
                input_road_ids.append(signal_group.road.get('id'))
            input_road_ids = list(set(input_road_ids))
            
            success_flg = False
            for intersection in self.network.intersections.getAll():
                if sorted(input_road_ids) == sorted(intersection.input_roads.getMultiAttValues('id')):
                    signal_controller.set('intersection', intersection)
                    intersection.set('signal_controller', signal_controller)
                    success_flg = True
                    break

            if success_flg == False:
                raise Exception(f"SignalController {signal_controller.get('id')} has input roads {input_road_ids}, but no matching intersection found.")

    def makePhases(self):
        num_roads_phases_map = self.config.get('num_roads_phases_map')

        for signal_controller in self.getAll():
            num_roads = signal_controller.intersection.get('num_roads')
            phases = num_roads_phases_map[num_roads]

            formatted_phases = {}
            for _, phase in phases.iterrows():
                formatted_phases[int(phase['id'])] = [int(phase['signal_group' + str(i)]) for i in range(1, num_roads + 1)]
        
            signal_controller.set('phases', formatted_phases)
    
    def setNextPhaseToVissim(self):
        for signal_controller in self.getAll():
            signal_controller.setNextPhaseToVissim()

class SignalController(Object):
    def __init__(self, com, signal_controllers):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = signal_controllers.config
        self.signal_controllers = signal_controllers

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = int(self.com.AttValue('No'))

        # 下位の紐づくオブジェクトを初期化
        self.signal_groups = SignalGroups(self)

        # phase_recordとfuture_phase_idsを初期化
        self.initPhaseRecord()
        self.initFuturePhaseIds()

        # red_stepsを初期化
        simulation_info = self.config.get('simulator_info')
        self.red_steps = simulation_info['red_steps']

    @property
    def num_phases(self):
        return len(self.phases)

    @property
    def current_phase_id(self):
        if self.phase_record:
            return self.phase_record[-1]
        else:
            return None # レコードがない場合はNoneを返す

    def initPhaseRecord(self):
        records_info = self.config.get('records_info')
        if records_info['metric']['phase'] == True:
            self.phase_record = []
        else:
            self.phase_record = deque(maxlen=records_info['max_len'])
    
    def initFuturePhaseIds(self):
        simulator_info = self.config.get('simulator_info')
        if simulator_info['control_method'] == 'drl':
            drl_info = self.config.get('drl_info')
            if drl_info['method'] == 'apex':
                apex_info = self.config.get('apex_info')
                self.future_phase_ids = deque(maxlen=apex_info['duration_steps'] + 1) # +1は現在のフェーズを含むため
        
    def setNextPhase(self, phase_ids):
        # フェーズをセット
        self.future_phase_ids.extend(phase_ids)

        # signal_groupにフェーズをセット
        self.signal_groups.setNextPhase(phase_ids)
    
    def setNextPhaseToVissim(self):
        # Vissimにフェーズをセット
        self.signal_groups.setNextPhaseToVissim()

        # phase_recordに追加して、future_phase_idsから削除
        self.phase_record.append(self.future_phase_ids.popleft())

class SignalGroups(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = upper_object.config

        if upper_object.__class__.__name__ == 'SignalController':
            self.signal_controller = upper_object

            # comオブジェクトを取得
            self.com = self.signal_controller.com.SGs

            # 下位の紐づくオブジェクトを初期化
            self.makeElements()

            # signal_groupとsignal_headを紐づける
            self.makeSignalHeadConnections()

            # signal_groupとroadを紐づける
            self.makeRoadConnections()
        
        elif upper_object.__class__.__name__ == 'Road':
            # 上位の紐づくオブジェクトを取得
            self.road = upper_object
    
    def makeElements(self):
        for signal_group_com in self.com.GetAll():
            self.add(SignalGroup(signal_group_com, self))
    
    def makeSignalHeadConnections(self):
        for signal_group in self.getAll():
            signal_heads = signal_group.signal_heads
            
            for signal_head_com in signal_heads.com.GetAll():
                signal_head_id = int(signal_head_com.AttValue('No'))
                signal_heads.add(self.network.signal_heads[signal_head_id])

    @property
    def network(self):
        if self.has('signal_controller'):
            return self.signal_controller.signal_controllers.network
        elif self.has('road'):
            return self.road.roads.network
    
    def makeRoadConnections(self):
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

    def setNextPhase(self, phase_ids):
        # フェーズのリストを取得
        phases = self.signal_controller.get('phases')
        
        # 各フェーズに対応するSignalGroupの値を計算
        sig_value_list = []
        for phase_id in phase_ids:
            signal_group_ids = phases[phase_id]
            tmp_sig_value_list = [1] * self.count()  # 1は赤信号を示す
            for signal_group_id in signal_group_ids:
                tmp_sig_value_list[signal_group_id - 1] = 3
            sig_value_list.append(tmp_sig_value_list)

        # 将来の信号現示を保存する（赤➡青の変化時は全赤の時間があるため，赤を１ステップ追加する）
        for signal_group in self.getAll():
            tmp_sig_value_list = [tmp_row[signal_group.get('id') - 1] for tmp_row in sig_value_list]
            signal_group.setNextPhase(tmp_sig_value_list)

    def setNextPhaseToVissim(self):
        for signal_group in self.getAll():
            signal_group.setNextPhaseToVissim()
            
            
class SignalGroup(Object):
    def __init__(self, com, signal_groups):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = signal_groups.config
        self.signal_groups = signal_groups

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = int(self.com.AttValue('No'))

        # signal_headを格納するコンテナを初期化
        self.signal_heads = SignalHeads(self)

        # future_valuesとvalue_recordを初期化
        self.initFutureValues()
        self.initValueRecord()

        # 現在の値を初期化
        self.current_value = None
    
    def initValueRecord(self):
        records_info = self.config.get('records_info')
        if records_info['metric']['phase'] == True:
            self.value_record = []
        else:
            self.value_record = deque(maxlen=records_info['max_len'])
    
    def initFutureValues(self):
        simulator_info = self.config.get('simulator_info')
        if simulator_info['control_method'] == 'drl':
            drl_info = self.config.get('drl_info')
            if drl_info['method'] == 'apex':
                apex_info = self.config.get('apex_info')
                self.future_values = deque(maxlen=apex_info['duration_steps'] + 1) # +1は現在のフェーズを含むため

    @property
    def direction_id(self):
        possible_direction_ids = []
        for signal_head in self.signal_heads.getAll():
            possible_direction_ids.append(signal_head.get('direction_id'))
        
        if len(set(possible_direction_ids)) == 1:
            return possible_direction_ids[0]
        else:
            raise Exception(f"SignalGroup {self.get('id')} has multiple possible direction IDs: {possible_direction_ids}. Please check the signal head connections.")
            
    def setNextPhase(self, sig_value_list):
        if self.future_values:
            if self.future_values[-1] == 1 and sig_value_list[0] == 3:
                sig_value_list[0] = 1
        elif self.value_record:
            if self.value_record[-1] == 1 and sig_value_list[0] == 3:
                sig_value_list[0] = 1
        
        self.future_values.extend(sig_value_list)

    def setNextPhaseToVissim(self):
        # 現在の値と同じ場合は何もしない
        if (self.current_value is not None) and (self.current_value == self.future_values[0]):
            self.value_record.append(self.future_values.popleft())
            return
        
        # Vissimに信号現示をセット（最初はうまく行かないのでtryで囲む）
        try:
            self.com.SetAttValue('SigState', self.future_values[0])
        except:
            print('Vissim is not running yet, so setting signal state is skipped.')
        
        # future_valuesから1つ削除
        self.value_record.append(self.future_values.popleft())
