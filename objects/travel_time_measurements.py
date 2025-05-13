from libs.container import Container
from libs.object import Object
from objects.links import Links

class TravelTimeMeasurements(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object

            # 対応するComオブジェクトを取得
            self.com = self.network.com.VehicleTravelTimeMeasurements

            # 要素オブジェクトの初期化
            self.makeElements()

            # linkと紐づける
            self.makeLinkConnections()

            # vehicle_routesオブジェクトとvehicle_routing_decisionオブジェクトと紐づける
            self.makeVehicleRouteConnections()

        elif upper_object.__class__.__name__ == 'VehicleRoutingDecision':
            # 上位の紐づくオブジェクトを取得
            self.vehicle_routing_decision = upper_object
        
        elif upper_object.__class__.__name__ == 'Link':
            # 上位の紐づくオブジェクトを取得
            self.link = upper_object
    
    def makeElements(self):
        for travel_time_measurement_com in self.com.GetAll():
            self.add(TravelTimeMeasurement(travel_time_measurement_com, self))
    
    def makeLinkConnections(self):
        for travel_time_measurement in self.getAll():
            # linksオブジェクトtype_link_mapを取得
            links = travel_time_measurement.links
            type_link_map = travel_time_measurement.type_link_map

            # travel_time_measurementの始まりと終りのlinkのcomオブジェクトを取得
            start_link_com = travel_time_measurement.com.StartLink
            end_link_com = travel_time_measurement.com.EndLink

            # 対応するlinkオブジェクトを取得
            start_link = self.network.links[int(start_link_com.AttValue('No'))]
            end_link = self.network.links[int(end_link_com.AttValue('No'))]

            # validation
            if start_link.get('type') != 'main':
                raise ValueError('Every start link must be a main link.')
            if end_link.get('type') != 'connector':
                raise ValueError('Every end link must be a connector link.')

            # travel_time_measurementオブジェクトにlinkオブジェクトを紐づける
            links.add(start_link)
            links.add(end_link)

            # travel_time_measurementのコンテナが存在しない場合は作成
            if start_link.has('travel_time_measurements') == False:
                start_link.set('travel_time_measurements', TravelTimeMeasurements(start_link))
            if end_link.has('travel_time_measurements') == False:
                end_link.set('travel_time_measurements', TravelTimeMeasurements(end_link))

            # linkオブジェクトにtravel_time_measurementオブジェクトを紐づける  
            start_link.travel_time_measurements.add(travel_time_measurement)
            end_link.travel_time_measurements.add(travel_time_measurement)    

            # travel_time_measurement側にはさらにマップに情報を保存
            type_link_map['start'] = start_link.get('id')
            type_link_map['end'] = end_link.get('id')
    
    def makeVehicleRouteConnections(self):
        for travel_time_measurement in self.getAll():
            # vehicle_routeオブジェクトと紐づける
            end_link = travel_time_measurement.end_link
            vehicle_route = end_link.vehicle_route

            vehicle_route.set('travel_time_measurement', travel_time_measurement)
            travel_time_measurement.set('vehicle_route', vehicle_route)

            # vehicle_routing_decisionオブジェクトと紐づける
            start_link = travel_time_measurement.start_link
            vehicle_routing_decision = start_link.vehicle_routing_decision

            vehicle_routing_decision.travel_time_measurements.add(travel_time_measurement)
            travel_time_measurement.set('vehicle_routing_decision', vehicle_routing_decision)

class TravelTimeMeasurement(Object):
    def __init__(self, com, travel_time_measurements):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = travel_time_measurements.config
        self.executor = travel_time_measurements.executor
        self.travel_time_measurements = travel_time_measurements

        # 対応するComオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # 紐づくlinkオブジェクトを格納するためのコンテナを初期化
        self.links = Links(self)
        self.type_link_map = {}
    
    @property
    def start_link(self):
        return self.links[self.type_link_map['start']]
    
    @property
    def end_link(self):
        return self.links[self.type_link_map['end']]

    @property
    def direction_id(self):
        return self.vehicle_route.get('direction_id')