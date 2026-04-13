from libs.container import Container
from libs.object import Object
from objects.roads import Roads

class Intersections(Container):
    def __init__(self, upper_object):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self._initElements()
        elif upper_object.__class__.__name__ == 'MasterAgent':
            self.master_agent = upper_object
        else:
            raise NotImplementedError(f"Not supported upper_object: {upper_object.__class__.__name__}")
        
        return

    def _initElements(self):
        intersections_df = self.config.get('intersections_df')
        for _, intersection_row in intersections_df.iterrows():
            self.add(Intersection(intersection_row, self))
        return
    
class Intersection(Object):
    def __init__(self, intersection_row, intersections):
        super().__init__()
        
        self.config = intersections.config
        self.executor = intersections.executor
        self.intersections = intersections
        self.network = intersections.network
        
        self._initProps(intersection_row)
        return
    
    def _initProps(self, intersection_row):
        self.id = int(intersection_row['id'])
        self.num_roads = int(intersection_row['num_roads'])

        self.input_roads = Roads(self, type='input')
        self.output_roads = Roads(self, type='output')

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
            if road.queue_counters.get('max_queue_length') > max_queue_length:
                max_queue_length = road.queue_counters.get('max_queue_length')
        return max_queue_length