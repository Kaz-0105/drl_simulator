from libs.container import Container
from libs.object import Object


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