from libs.container import Container
from libs.object import Object
from objects.links import Links
import pandas as pd

class DelayMeasurements(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            # 上位の紐づくオブジェクトを取得
            self.network = upper_object

            # 対応するComオブジェクトを取得
            self.com = self.network.com.DelayMeasurements

            # 要素オブジェクトの初期化
            self.makeElements()

            # travel_time_measurementと紐づける
            self.makeTravelTimeConnections()

            # linkと紐づける
            self.makeLinkConnections()

            # vehicle_routeとvehicle_routing_decisionオブジェクトと紐づける
            self.makeVehicleRouteConnections()
        
        elif upper_object.__class__.__name__ == 'VehicleRoutingDecision':
            # 上位の紐づくオブジェクトを取得
            self.vehicle_routing_decision = upper_object
        
        elif upper_object.__class__.__name__ == 'Link':
            # 上位の紐づくオブジェクトを取得
            self.link = upper_object
    
    def makeElements(self):
        for delay_measurement_com in self.com.GetAll():
            self.add(DelayMeasurement(delay_measurement_com, self))
        
    def makeTravelTimeConnections(self):
        for delay_measurement in self.getAll():
            travel_time_measurements_com = delay_measurement.com.VehTravTmMeas

            # 複数のVehicleTravelTimeMeasurementに接続することも可能であるが，それは認めない
            if travel_time_measurements_com.Count > 1:
                raise ValueError('DelayMeasurement is connected to multiple VehicleTravelTimeMeasurements.')
            
            measurement_id = travel_time_measurements_com.GetAll()[0].AttValue('No')
            travel_time_measurement = self.network.travel_time_measurements[measurement_id]
            
            delay_measurement.set('travel_time_measurement', travel_time_measurement)
            travel_time_measurement.set('delay_measurement', delay_measurement)

    def makeLinkConnections(self):
        for delay_measurement in self.getAll():
            # linksオブジェクトtype_link_mapを取得
            links = delay_measurement.links
            type_link_map = delay_measurement.type_link_map

            # travel_time_measurementの始まりと終りのlinkのcomオブジェクトを取得
            travel_time_measurement = delay_measurement.travel_time_measurement
            start_link = travel_time_measurement.start_link
            end_link = travel_time_measurement.end_link

            # delay_measurementオブジェクトにlinkオブジェクトを紐づける
            links.add(start_link)
            links.add(end_link)
            type_link_map['start'] = start_link.get('id')
            type_link_map['end'] = end_link.get('id')

            # linkオブジェクトにdelay_measurementsオブジェクトが存在しない場合は作成
            if start_link.has('delay_measurements') == False:
                start_link.set('delay_measurements', DelayMeasurements(start_link))
            if end_link.has('delay_measurements') == False:
                end_link.set('delay_measurements', DelayMeasurements(end_link))
            
            # linkオブジェクトにdelay_measurementオブジェクトを紐づける
            start_link.delay_measurements.add(delay_measurement)
            end_link.delay_measurements.add(delay_measurement)
    
    def makeVehicleRouteConnections(self):
        for delay_measurement in self.getAll():
            # vehicle_routeオブジェクトと紐づける
            end_link = delay_measurement.end_link
            vehicle_route = end_link.vehicle_route
            vehicle_route.set('delay_measurement', delay_measurement)
            delay_measurement.set('vehicle_route', vehicle_route)

            # vehicle_routing_decisionオブジェクトと紐づける
            start_link = delay_measurement.start_link
            vehicle_routing_decision = start_link.vehicle_routing_decision
            vehicle_routing_decision.delay_measurements.add(delay_measurement)
            delay_measurement.set('vehicle_routing_decision', vehicle_routing_decision)
    
    def updateData(self):
        for delay_measurement in self.getAll():
            # Comオブジェクトからデータを取得
            delay_measurement_ids = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('No')]
            delays = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('VehDelay(Current, Last, All)')]

            # データを要素オブジェクトにセット（非同期処理）
            for index, delay_measurement_id in enumerate(delay_measurement_ids):
                delay_measurement = self[delay_measurement_id]
                self.executor.submit(delay_measurement.updateData, delays[index])
    

class DelayMeasurement(Object):
    def __init__(self, com, delay_measurements):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = delay_measurements.config
        self.executor = delay_measurements.executor
        self.delay_measurements = delay_measurements

        # 対応するComオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # linkオブジェクトを格納するためのコンテナを初期化
        self.links = Links(self)
        self.type_link_map = {}

        # 現在の遅れの値を初期化
        self.current_delay = 0

        # delays（時系列データ）を初期化
        self.delays = None
    
    @property
    def start_link(self):
        return self.links[self.type_link_map['start']]
    
    @property
    def end_link(self):
        return self.links[self.type_link_map['end']]
    
    @property
    def direction_id(self):
        return self.travel_time_measurement.get('direction_id')

    @property
    def network(self):
        return self.delay_measurements.network
    
    @property
    def current_time(self):
        return self.network.simulation.get('current_time')
    
    def updateData(self, delay):
        # current_delayを更新
        if delay is not None:
            self.current_delay = round(delay, 1)

        # delaysを更新
        new_delay = pd.DataFrame({'time': [self.current_time], 'delay': [self.current_delay]})

        if self.delays is None:
            self.delays = new_delay
        else:
            self.delays = pd.concat([self.delays, new_delay], ignore_index=True)

    
