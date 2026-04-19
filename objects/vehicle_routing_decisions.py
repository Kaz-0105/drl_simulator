from libs.container import Container
from libs.object import Object
import math
from functools import reduce
from objects.signal_heads import SignalHeads
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
    def __init__(self, com, vehicle_routing_decisions):
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = vehicle_routing_decisions.config
        self.executor = vehicle_routing_decisions.executor
        self.vehicle_routing_decisions = vehicle_routing_decisions
        self.network = vehicle_routing_decisions.network

        # set com object
        self.com = com

        # connect to link and road
        self.link = self.network.links[int(self.com.Link.AttValue('No'))]
        self.link.set('vehicle_routing_decision', self)
        self.road = self.link.road
        self.road.set('vehicle_routing_decision', self) 

        # set vehicle_routes, travel_time_measurements, and delay_measurements objects
        self.vehicle_routes = VehicleRoutes(self)
        self.travel_time_measurements = TravelTimeMeasurements(self)
        self.delay_measurements = DelayMeasurements(self)

        self._initProps()

        if self.network.get('control_method') == 'scoot' and self.road.has('output_intersection'): 
            self._initEffectiveStorageLengthMap()
        return
    
    def _initProps(self):
        # set id and turn_ratios
        self.id = int(self.com.AttValue('No'))
        self.turn_ratios = self.road.get('turn_ratios')

        # set num_routes_map
        num_roads = self.road.output_intersection.get('num_roads')

        self.num_routes_map = {route_id: 0 for route_id in range(1, num_roads + 1)}
        for vehicle_route in self.vehicle_routes.getAll():
            route_id = vehicle_route.get('route_id')
            self.num_routes_map[route_id] += 1

        # set turn_ratio for each vehicle_route
        lcm = reduce(math.lcm, list(num_routes for num_routes in self.num_routes_map.values() if num_routes > 0))
        for vehicle_route in self.vehicle_routes.getAll():
            route_id = vehicle_route.get('route_id')
            vehicle_route.set('turn_ratio', self.turn_ratios[route_id] / self.num_routes_map[route_id] * lcm)
        
        return
    
    def _initEffectiveStorageLengthMap(self):
        effective_storage_length_map = {'left': 0.0, 'straight': 0.0, 'right': 0.0}

        # 道路タイプで分岐
        if self.road.get('type') == 1:
            # 車線数：分岐前1車線，分岐後2車線
            # 進路：左車線は左折と直進，右車線は右折
            
            #　左車線について
            before_branch_length = self.road.right_connector.get('from_pos')
            after_branch_length = self.road.main_link.get('length') - before_branch_length

            effective_storage_length_map['left'] += self.turn_ratios['left'] / (sum(self.turn_ratios.values())) * before_branch_length + self.turn_ratios['left'] / (self.turn_ratios['left'] + self.turn_ratios['straight']) * after_branch_length
            effective_storage_length_map['straight'] += self.turn_ratios['straight'] / (sum(self.turn_ratios.values())) * before_branch_length + self.turn_ratios['straight'] / (self.turn_ratios['left'] + self.turn_ratios['straight']) * after_branch_length
            effective_storage_length_map['right'] += self.turn_ratios['right'] / (sum(self.turn_ratios.values())) * before_branch_length

            # 右車線について
            branch_length = 0.0
            branch_length += self.road.right_link.get('length')
            branch_length += self.road.right_connector.get('length') 
            branch_length -= self.road.right_connector.get('to_pos')
            effective_storage_length_map['right'] += branch_length

        elif self.road.get('type') == 2:
            # 車線数：分岐前2車線，分岐後3車線
            # 進路：左車線は左折と直進，真ん中の車線は直進，右車線は右折

            # 左車線について
            main_link_length = self.road.main_link.get('length')
            effective_storage_length_map['left'] += (self.turn_ratios[1] / (self.turn_ratios[1] + (self.turn_ratios[2] / 2))) *  main_link_length
            effective_storage_length_map['straight'] += ((self.turn_ratios[2] / 2) / (self.turn_ratios[1] + (self.turn_ratios[2] / 2))) *  main_link_length

            # 真ん中の車線について
            before_branch_length = self.road.right_connector.get('from_pos')
            after_branch_length = self.road.main_link.get('length') - before_branch_length
            effective_storage_length_map['straight'] += ((self.turn_ratios[2] / 2) / (self.turn_ratios[3] + (self.turn_ratios[2] / 2))) * before_branch_length + after_branch_length
            effective_storage_length_map['right'] += (self.turn_ratios[3] / (self.turn_ratios[3] + (self.turn_ratios[2] / 2))) * before_branch_length

            # 右車線について
            branch_length = 0.0
            branch_length += self.road.right_link.get('length')
            branch_length += self.road.right_connector.get('length') 
            branch_length -= self.road.right_connector.get('to_pos')
            effective_storage_length_map['right'] += branch_length
        else:
            raise NotImplementedError(f"Not supported road type: {self.road.get('type')}")  

        self.road.set('effective_storage_length_map', effective_storage_length_map)
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

        self.com = com

        # set signal_heads object and connector
        self.signal_heads = SignalHeads(self)
        self.connector = self.vehicle_routing_decision.network.links[int(self.com.DestLink.AttValue('No'))]
        self.connector.set('vehicle_route', self)   

        self._initProps(road_route_map)
        return
    
    def _initProps(self, road_route_map):
        # set id
        self.id = int(self.com.AttValue('No'))

        # set route_id
        target_road = self.connector.to_link.road
        self.route_id = road_route_map[target_road.get('id')]

        # set turn_ratio
        self.turn_ratio = None
        return