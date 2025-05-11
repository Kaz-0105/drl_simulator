from libs.container import Container
from libs.object import Object
import math
from functools import reduce
from objects.signal_heads import SignalHeads
from objects.travel_time_measurements import TravelTimeMeasurements
from objects.delay_measurements import DelayMeasurements

class VehicleRoutingDecisions(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = network.config
        self.network = network

        # 対応するComオブジェクトを取得
        self.com = self.network.com.VehicleRoutingDecisionsStatic

        # 要素オブジェクトを初期化
        self.makeElements()

        # linkクラスに紐づける
        self.makeLinkConnections()

        # vehicle_routeオブジェクトに方向（左折，直進，右折など）を設定
        self.setDirectionsForVehicleRoutes()

        # パラメータを設定
        self.setTurnRatio()
    
    def makeElements(self):
        for vehicle_routing_decision_com in self.com.GetAll():
            self.add(VehicleRoutingDecision(vehicle_routing_decision_com, self))
    
    def makeLinkConnections(self):
        for vehicle_routing_decision in self.getAll():
            # linkオブジェクトを取得
            link_com = vehicle_routing_decision.com.Link
            link = self.network.links[int(link_com.AttValue('No'))]

            # それぞれに対して紐づける
            vehicle_routing_decision.set('link', link)
            link.set('vehicle_routing_decision', vehicle_routing_decision)

            for vehicle_route in vehicle_routing_decision.vehicle_routes.getAll():
                # linkオブジェクト(connector）を取得
                connector_com = vehicle_route.com.DestLink
                connector = self.network.links[int(connector_com.AttValue('No'))]

                # それぞれに対して紐づける
                vehicle_route.set('connector', connector)
                connector.set('vehicle_route', vehicle_route)
    
    def setDirectionsForVehicleRoutes(self):
        for vehicle_routing_decision in self.getAll():
            # 紐づくroadオブジェクトを取得
            road = vehicle_routing_decision.getRoad()
            
            # road_order_mapを取得
            intersection = road.output_intersection
            road_order_map = intersection.getRoadOrderMap()
            num_roads = intersection.get('num_roads')

            # road_direction_mapを作成
            road_direction_map = {}
            current_road_id = road.get('id')
            current_order_id = road_order_map[current_road_id]

            for road_id, order_id in road_order_map.items():
                direction_id = (order_id - current_order_id) % num_roads
                road_direction_map[road_id] = direction_id                    

            # vehicle_routeオブジェクトに方向を設定
            for vehicle_route in vehicle_routing_decision.vehicle_routes.getAll():
                connector = vehicle_route.get('connector')
                to_link = connector.to_links.getAll()[0]
                vehicle_route.set('direction_id', road_direction_map[to_link.road.get('id')])       

    def setTurnRatio(self):
        # 設定ファイルから旋回率に関する情報を取得
        tags = self.config.get('intersection_turn_ratio_tags')
        templates_map = self.config.get('num_road_turn_ratio_map')

        for vehicle_routing_decision in self.getAll():
            # 紐づくroadオブジェクトとintersectionオブジェクトを取得
            road = vehicle_routing_decision.getRoad()
            intersection = vehicle_routing_decision.getIntersection()

            # 紐づく交差点の道路数を取得（十字路，三差路，五差路など）
            num_roads = intersection.get('num_roads')
            
            # 交差点のIDと道路の順番を示すIDを取得
            intersection_id = intersection.get('id')
            road_order_map = intersection.getRoadOrderMap()
            road_order_id = road_order_map[road.get('id')]

            # 該当する設定ファイル内のレコードを取得
            target_tag = tags[
                (tags['intersection_id'] == intersection_id) &
                (tags['road_order_id'] == road_order_id)
            ]
            target_tag = target_tag.iloc[0]

            # テンプレートIDに対応するレコードを取得
            templates = templates_map[num_roads]
            target_template = templates[templates['id'] == target_tag['turn_ratio_template_id']]
            target_template = target_template.iloc[0]

            # 進路方向ごとのvehicle_routeオブジェクトの数を取得
            direction_num_veh_routes_map = vehicle_routing_decision.getDirectionNumVehRoutesMap()

            # 最小公倍数を計算
            lcm_number = reduce(math.lcm, list(direction_num_veh_routes_map.values()))

            # 各vehicle_routeオブジェクトに旋回率をセット
            for vehicle_route in vehicle_routing_decision.vehicle_routes.getAll():
                direction_id = vehicle_route.get('direction_id')
                vehicle_route.set('turn_ratio', int(target_template['ratio' + str(direction_id)] / direction_num_veh_routes_map[direction_id] * lcm_number))

class VehicleRoutingDecision(Object):
    def __init__(self, com, vehicle_routing_decisions):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = vehicle_routing_decisions.config
        self.vehicle_routing_decisions = vehicle_routing_decisions

        # 対応するComオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # 下位の紐づくオブジェクトを初期化
        self.vehicle_routes = VehicleRoutes(self)

        # travel_time_measurementオブジェクトを紐づけるためのコンテナを初期化
        self.travel_time_measurements = TravelTimeMeasurements(self)

        # delay_measurementオブジェクトを紐づけるためのコンテナを初期化
        self.delay_measurements = DelayMeasurements(self)
    
    def getRoad(self):
        return self.link.road
    
    def getIntersection(self):
        return self.link.road.output_intersection
    
    def getDirectionNumVehRoutesMap(self):
        direction_num_veh_routes = {}
        for vehicle_route in self.vehicle_routes.getAll():
            direction_id = vehicle_route.get('direction_id')

            if direction_id not in direction_num_veh_routes:
                direction_num_veh_routes[direction_id] = 1
            else:
                direction_num_veh_routes[direction_id] += 1

        return direction_num_veh_routes

class VehicleRoutes(Container):
    def __init__(self, vehicle_routing_decision):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = vehicle_routing_decision.config
        self.vehicle_routing_decision = vehicle_routing_decision
    
        # 対応するComオブジェクトを取得
        self.com = self.vehicle_routing_decision.com.VehRoutSta

        # 要素オブジェクトを初期化
        self.makeElements()

    def makeElements(self):
        for vehicle_route_com in self.com.GetAll():
            self.add(VehicleRoute(vehicle_route_com, self))


class VehicleRoute(Object):
    def __init__(self, com, vehicle_routes):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = vehicle_routes.config
        self.vehicle_routes = vehicle_routes

        # 対応するComオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # signal_headオブジェクトを格納するコンテナを初期化
        self.signal_heads = SignalHeads(self)