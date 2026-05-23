from libs.container import Container
from libs.object import Object

class SignalHeads(Container):
    def __init__(self, upper_object):
        super().__init__()

        self.config = upper_object.config

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.com = self.network.com.SignalHeads
            
            self._initElements(upper_object)

        elif upper_object.__class__.__name__ == 'VehicleRoute':
            self.vehicle_route = upper_object
        
        elif upper_object.__class__.__name__ == 'SignalGroup':
            self.signal_group = upper_object
            self.network = self.signal_group.network

            self.com = self.signal_group.com.SigHeads

            self._initElements(upper_object)
        
        else:
            raise NotImplementedError(f"Not supported upper_object class: {upper_object.__class__.__name__}")
        
        return
    
    def _initElements(self, upper_object):
        if upper_object.__class__.__name__ == 'Network':
            for signal_head_com in self.com.GetAll():
                self.add(SignalHead(signal_head_com, self))
        
        elif upper_object.__class__.__name__ == 'SignalGroup':
            for signal_head_com in self.com.GetAll():
                self.add(self.network.signal_heads[int(signal_head_com.AttValue('No'))])
        
        else:
            raise NotImplementedError(f"Not supported upper_object class for element initialization: {upper_object.__class__.__name__}")

        return

class SignalHead(Object):
    def __init__(self, com, signal_heads):
        super().__init__()

        self.config = signal_heads.config
        self.signal_heads = signal_heads
        self.network = signal_heads.network
        self.com = com

        self._initProps()
        self._connectObjects()
        return
    
    def _initProps(self):
        self.id = int(self.com.AttValue('No'))
        self.route_id = None
        return
    
    def _connectObjects(self):
        # set connector
        self.connector = self.network.links[int(self.com.Lane.Link.AttValue('No'))]
        self.connector.signal_head = self

        # set lane
        self.lane = self.connector.lanes[int(self.com.Lane.AttValue('Index'))]
        self.lane.signal_head = self

        # set vehicle_route
        self.vehicle_route = self.connector.vehicle_route
        self.vehicle_route.signal_head = self
        self.route_id = self.vehicle_route.get('route_id')
        return
