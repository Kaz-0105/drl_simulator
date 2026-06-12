from libs.container import Container
from libs.object import Object
import math
from functools import reduce
from objects.travel_time_measurements import TravelTimeMeasurements
from objects.delay_measurements import DelayMeasurements

class VehicleRoutingDecisions(Container):
    def __init__(self, network):
        super().__init__()

        self.config = network.config
        self.executor = network.executor
        self.network = network

        # set com object
        self.com = self.network.com.VehicleRoutingDecisionsStatic

        # set elements
        self._initElements()
            
        return
    
    def _initElements(self):
        for vehicle_routing_decision_com in self.com.GetAll():
            self.add(VehicleRoutingDecision(vehicle_routing_decision_com, self))
        
        return
        
class VehicleRoutingDecision(Object):
    ROAD_TYPE_MAP = {
        (1, 2): 1, # 1: 分岐前1車線，分岐後2車線
        (2, 3): 2, # 2: 分岐前2車線，分岐後3車線
        (1, 1): 3, # 3: 分岐前1車線，分岐後1車線
    }
    def __init__(self, com, vehicle_routing_decisions):
        super().__init__()

        self.config = vehicle_routing_decisions.config
        self.executor = vehicle_routing_decisions.executor
        self.vehicle_routing_decisions = vehicle_routing_decisions
        self.network = vehicle_routing_decisions.network

        self.com = com

        self._initProps()
        self._connectObjects()
        return
    
    def _initProps(self):
        # set id
        self.id = int(self.com.AttValue('No'))

        # set turn_ratios (VehicleRoutingDecision._connectObjects())
        self.turn_ratios = None

        # set num_routes_map (VehicleRoutingDecision._connectObjects())
        self.num_routes_map = None
        return
    
    def _connectObjects(self):
        # set link
        self.link = self.network.links[int(self.com.Link.AttValue('No'))]
        self.link.vehicle_routing_decision = self

        # set road
        self.road = self.link.road
        self.road.vehicle_routing_decision = self

        # set vehicle_routes
        self.vehicle_routes = VehicleRoutes(self)

        # set turn_ratios and num_routes_map
        self.turn_ratios = self.road.get('turn_ratios')
        self.num_routes_map = {route_id: 0 for route_id in range(self.road.output_intersection.get('num_roads'))}
        for vehicle_route in self.vehicle_routes.getAll():
            route_id = vehicle_route.get('route_id')
            self.num_routes_map[route_id] += 1

        # set turn_ratio for each vehicle_route
        lcm = reduce(math.lcm, list(num_routes for num_routes in self.num_routes_map.values() if num_routes > 0))
        for vehicle_route in self.vehicle_routes.getAll():
            route_id = vehicle_route.get('route_id')
            vehicle_route.set('turn_ratio', self.turn_ratios[route_id] / self.num_routes_map[route_id] * lcm)

        # set travel_time_measurements (TravelTimeMeasurement._connectObjects())
        self.travel_time_measurements = TravelTimeMeasurements(self)

        # set delay_measurements (DelayMeasurement._connectObjects())
        self.delay_measurements = DelayMeasurements(self)
        return

class VehicleRoutes(Container):
    def __init__(self, vehicle_routing_decision):
        super().__init__()

        self.config = vehicle_routing_decision.config
        self.executor = vehicle_routing_decision.executor
        self.vehicle_routing_decision = vehicle_routing_decision
        self.com = self.vehicle_routing_decision.com.VehRoutSta

        self._initElements()
        return

    def _initElements(self):
        # get road and intersection
        road = self.vehicle_routing_decision.road
        intersection = road.output_intersection
        
        # make road_direction_map
        road_order_map = intersection.get('road_order_map')
        target_order_id = road_order_map[road.get('id')]
        num_roads = intersection.get('num_roads')

        road_route_map = {}
        for road_id, order_id in road_order_map.items():
            route_id = (order_id - target_order_id) % num_roads
            road_route_map[road_id] = route_id

        for vehicle_route_com in self.com.GetAll():
            self.add(VehicleRoute(
                com=vehicle_route_com,
                vehicle_routes=self,
                road_route_map=road_route_map
            ))

        return           

class VehicleRoute(Object):
    def __init__(self, com, vehicle_routes, road_route_map):
        super().__init__()

        self.config = vehicle_routes.config
        self.executor = vehicle_routes.executor
        self.vehicle_routes = vehicle_routes
        self.vehicle_routing_decision = vehicle_routes.vehicle_routing_decision
        self.network = self.vehicle_routing_decision.network

        self.com = com

        self._initProps()
        self._connectObjects(road_route_map)
        return
    
    def _initProps(self):
        # set id
        self.id = int(self.com.AttValue('No'))

        # set route_id (VehicleRoute._connectObjects())
        self.route_id = None 

        # set turn_ratio
        self.turn_ratio = None
        return
    
    def _connectObjects(self, road_route_map):
        # set signal_head (SignalHead._connectObjects())
        self.signal_head = None

        # set connector
        self.connector = self.network.links[int(self.com.DestLink.AttValue('No'))]
        self.connector.vehicle_route = self
        self.route_id = road_route_map[self.connector.to_link.road.get('id')]

        # set lane
        self.lane = self.connector.lanes.getAll()[0]
        self.lane.vehicle_route = self

        # set data_collection_point (DataCollectionPoint._connectObjects())
        self.data_collection_point = None
        return