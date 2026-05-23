from libs.container import Container
from libs.object import Object
from objects.links import Links
from objects.roads import Roads

class Intersections(Container):
    def __init__(self, upper_object):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self._initElements(upper_object)
        elif upper_object.__class__.__name__ == 'MasterAgent':
            self.master_agent = upper_object
            self.network = upper_object.network
            self._initElements(upper_object)
        else:
            raise NotImplementedError(f"Not supported upper_object: {upper_object.__class__.__name__}")
        
        return

    def _initElements(self, upper_object):
        if upper_object.__class__.__name__ == 'Network':
            intersections_df = self.config.get('intersections_df')
            for _, intersection_row in intersections_df.iterrows():
                self.add(Intersection(intersection_row, self))
                
        elif upper_object.__class__.__name__ == 'MasterAgent':
            num_lanes_tuple = tuple((upper_object.get('num_lanes_map')[road_id] for road_id in range(1, upper_object.get('num_roads') + 1)))
            for intersection in self.network.intersections.getAll():
                if intersection.get('num_lanes_tuple') == num_lanes_tuple:
                    self.add(intersection)
                    intersection.master_agent = self.master_agent

        else:
            raise NotImplementedError(f"Not supported upper_object: {upper_object.__class__.__name__}")
        return
    
class Intersection(Object):
    def __init__(self, intersection_row, intersections):
        super().__init__()
        
        self.config = intersections.config
        self.executor = intersections.executor
        self.intersections = intersections
        self.network = intersections.network
        
        self._initProps(intersection_row)
        self._connectObjects()
        return
    
    def _initProps(self, intersection_row):
        self.id = int(intersection_row['id'])
        self.num_roads = int(intersection_row['num_roads'])
        return
    
    def _connectObjects(self):
        # set roads
        self.input_roads = Roads(self, 'input')
        self.output_roads = Roads(self, 'output')

        # set connectors (Links._connectObjects())
        self.connectors = Links(self)

        # set signal_controller (SignalController._connectObjects())
        self.signal_controller = None

        # set controller
        if self.network.get('control_method') == 'drl':
            if self.network.get('drl_framework') == 'apex':
                self.master_agent = None
                self.local_agent = None
            else:
                raise NotImplementedError(f"Not supported drl_framework: {self.network.get('drl_framework')}")
        
        elif self.network.get('control_method') == 'scoot':
            self.scoot_controller = None

        elif self.network.get('control_method') == 'mpc':
            self.mpc_controller = None
        
        else:
            raise NotImplementedError(f"Not supported control_method: {self.network.get('control_method')}")
        return

    @property
    def road_order_map(self):
        road_order_map = {}
        for order_id, road in self.input_roads.elements.items():
            road_order_map[road.get('id')] = order_id
        for order_id, road in self.output_roads.elements.items():
            road_order_map[road.get('id')] = order_id
        return road_order_map
    
    @property
    def num_lanes_tuple(self):
        num_lanes_list = []
        for road_order_id in self.input_roads.getKeys(container_flg=True, sorted_flg=True):
            road = self.input_roads[road_order_id]
            num_lanes = 0
            for link in road.links.getAll():
                if link.get('type') == 'connector':
                    continue
                num_lanes += link.lanes.count()
            num_lanes_list.append(num_lanes)
        return tuple(num_lanes_list)

    @property
    def current_time(self):
        return self.network.simulation.get('current_time')
    
    @property
    def current_phase_id(self):
        return self.signal_controller.get('current_phase_id')
    
    @property
    def num_phases(self):
        return self.signal_controller.get('num_phases')
    
    @property
    def max_queue_length(self):
        max_queue_length = 0
        for road in self.input_roads.getAll():
            if road.get('max_queue_length') > max_queue_length:
                max_queue_length = road.get('max_queue_length')
        return max_queue_length