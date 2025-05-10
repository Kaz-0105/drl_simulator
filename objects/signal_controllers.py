from libs.container import Container
from libs.object import Object
from objects.signal_heads import SignalHeads

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

    @property
    def direction_id(self):
        possible_direction_ids = []
        for signal_head in self.signal_heads.getAll():
            possible_direction_ids.append(signal_head.get('direction_id'))
        
        if len(set(possible_direction_ids)) == 1:
            return possible_direction_ids[0]
        else:
            raise Exception(f"SignalGroup {self.get('id')} has multiple possible direction IDs: {possible_direction_ids}. Please check the signal head connections.")
            
            
