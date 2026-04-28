from libs.container import Container
from libs.object import Object

class SignalHeads(Container):
    def __init__(self, upper_object):
        super().__init__()

        self.config = upper_object.config

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.com = self.network.com.SignalHeads
            
            self._initElements()

        elif upper_object.__class__.__name__ == 'VehicleRoute':
            self.vehicle_route = upper_object
        
        elif upper_object.__class__.__name__ == 'SignalGroup':
            self.signal_group = upper_object
            self.com = self.signal_group.com.SigHeads
            
            network = self.signal_group.signal_controller.network
            for signal_head_com in self.com.GetAll():
                self.add(network.signal_heads[int(signal_head_com.AttValue('No'))])
        
        return
    
    def _initElements(self):
        for signal_head_com in self.com.GetAll():
            self.add(SignalHead(signal_head_com, self))

        return

class SignalHead(Object):
    def __init__(self, com, signal_heads):
        super().__init__()

        self.config = signal_heads.config
        self.signal_heads = signal_heads
        self.network = signal_heads.network
        self.com = com

        # set id
        self.id = int(self.com.AttValue('No'))

        # set lane and connector
        lane_com = self.com.Lane
        link_com = lane_com.Link
        lane = self.network.links[link_com.AttValue('No')].lanes[lane_com.AttValue('Index')]

        self.lane = lane
        lane.set('signal_head', self)

        self.connector = lane.lanes.link

        # set vehicle route
        self.vehicle_route = self.connector.vehicle_route
        self.vehicle_route.signal_heads.add(self)

        # set route_id 
        self.route_id = self.vehicle_route.get('route_id')
        return
