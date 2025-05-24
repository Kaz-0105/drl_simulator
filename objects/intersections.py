from libs.container import Container
from libs.object import Object
from objects.roads import Roads

class Intersections(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            # 上位クラスのオブジェクトを取得
            self.network = upper_object

            # 要素オブジェクトを初期化
            self.makeElements()
        elif upper_object.__class__.__name__ == 'MasterAgent':
            # 上位クラスのオブジェクトを取得
            self.master_agent = upper_object

    def makeElements(self):
        intersections = self.config.get('intersections')
        for _, intersection in intersections.iterrows():
            self.add(Intersection(intersection, self))

class Intersection(Object):
    def __init__(self, intersection, intersections):
        super().__init__()
        self.config = intersections.config
        self.executor = intersections.executor
        self.intersections = intersections
        
        self.id = int(intersection['id'])
        self.num_roads = int(intersection['num_roads'])

        self.connectRoads()
    
    def connectRoads(self):
        self.input_roads = Roads(self, {'type': 'input'})
        self.output_roads = Roads(self, {'type': 'output'})
    
    def getNetwork(self):
        return self.intersections.network
    
    def getRoadOrderMap(self):
        road_order_map = {}
        for order_id, road in self.input_roads.elements.items():
            road_order_map[road.get('id')] = order_id
        
        for order_id, road in self.output_roads.elements.items():
            road_order_map[road.get('id')] = order_id

        return road_order_map

    @property
    def current_phase_id(self):
        return self.signal_controller.get('current_phase_id')
    
    @property
    def num_phases(self):
        return self.signal_controller.get('num_phases')

    def getNumLanesTurple(self):
        # 車線数のリストを初期化
        num_lanes_list = []

        # 道路を走査
        for road_order_id in self.input_roads.getKeys(container_flg=True, sorted_flg=True):
            road = self.input_roads[road_order_id]

            # 車線数を計算
            num_lanes = 0
            for link in road.links.getAll():
                if link.get('type') == 'connector':
                    continue

                num_lanes += link.lanes.count()
            
            num_lanes_list.append(num_lanes)

        return tuple(num_lanes_list)
