from libs.container import Container
from libs.object import Object

class SignalHeads(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = upper_object.config

        if upper_object.__class__.__name__ == 'Network':
            # 上位の紐づくオブジェクトを取得
            self.network = upper_object

            # comオブジェクトを取得
            self.com = self.network.com.SignalHeads

            # 下位の紐づくオブジェクトを初期化
            self.makeElements()

            # laneと紐づける
            self.makeLaneConnections()

            # vehicle_routeと紐づける
            self.makeVehRouteConnections()

        elif upper_object.__class__.__name__ == 'VehicleRoute':
            # 上位の紐づくオブジェクトを取得
            self.vehicle_route = upper_object
        
        elif upper_object.__class__.__name__ == 'SignalGroup':
            # 上位の紐づくオブジェクトを取得
            self.signal_group = upper_object

            # comオブジェクトを取得
            self.com = self.signal_group.com.SigHeads
    
    def makeElements(self):
        for signal_head_com in self.com.GetAll():
            self.add(SignalHead(signal_head_com, self))

    def makeLaneConnections(self):
        for signal_head in self.getAll():
            lane_com = signal_head.com.Lane
            link_com = lane_com.Link
            lane = self.network.links[link_com.AttValue('No')].lanes[lane_com.AttValue('Index')]

            # laneとsignal_headを紐づける
            signal_head.set('lane', lane)
            lane.set('signal_head', signal_head)
    
    def makeVehRouteConnections(self):
        for signal_head in self.getAll():
            connector = signal_head.connector
            vehicle_route = connector.vehicle_route

            # それぞれに対して紐づける
            signal_head.set('vehicle_route', vehicle_route)
            vehicle_route.signal_heads.add(signal_head)


class SignalHead(Object):
    def __init__(self, com, signal_heads):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = signal_heads.config
        self.signal_heads = signal_heads

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = int(self.com.AttValue('No'))
    
    @property
    def connector(self):
        return self.lane.lanes.link
    
    @property
    def direction_id(self):
        return self.vehicle_route.get('direction_id')
