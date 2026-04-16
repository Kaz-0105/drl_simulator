from libs.container import Container
from libs.object import Object
from objects.links import Links
import pandas as pd
import numpy as np

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
            self._initElements()

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
        
        elif upper_object.__class__.__name__ == 'Road':
            # 上位の紐づくオブジェクトを取得
            self.road = upper_object

        return
    
    def _initElements(self):
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

            # start_link側のみ紐づくroadオブジェクトとも紐づける（end_linkはconnectorなのでroadは存在しない）
            road = start_link.road
            road.delay_measurements.add(delay_measurement)
            delay_measurement.set('road', road)
        return
    
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
    
    def update(self):
        # get data from com object
        delay_measurement_ids = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('No')]
        delays = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('VehDelay(Current, Last, All)')]

        # set current_delay for each delay measurement
        for index, delay_measurement_id in enumerate(delay_measurement_ids):
            delay_measurement = self[delay_measurement_id]
            self.executor.submit(delay_measurement.update, delays[index])
        
        return

    def syncDataFrame(self):
        for delay_measurement in self.getAll():
            self.executor.submit(delay_measurement.syncDataFrame)
        
        self.executor.wait()
        return
    

class DelayMeasurement(Object):
    def __init__(self, com, delay_measurements):
        super().__init__()

        # set objects
        self.config = delay_measurements.config
        self.executor = delay_measurements.executor
        self.delay_measurements = delay_measurements
        self.network = delay_measurements.network

        # set com object
        self.com = com
        
        # initialize links
        self.links = Links(self)
        
        # initialize properties
        self._initProps()
        return
    
    @property
    def current_delay(self):
        for record in reversed(self.record_list):
            if not np.isnan(record['value']):
                return record['value']
        
        return 0.0

    @property
    def start_link(self):
        return self.links[self.type_link_map['start']]
    
    @property
    def end_link(self):
        return self.links[self.type_link_map['end']]
    
    @property
    def route_id(self):
        return self.travel_time_measurement.get('route_id')

    def _initProps(self):
        self.id = self.com.AttValue('No')
        self.record_list = []
        self.record_df = None
        self.type_link_map = {}
        self.time_step = self.network.simulation.get('time_step')
        return

    def update(self, value): 
        self.record_list.append({
            'time': int(self.network.get('current_time')),
            'value': value if value is not None else np.nan,
        })

        if value is None:
            return

        for record in reversed(self.record_list[:-1]):
            if not np.isnan(record['value']):
                break

            record['value'] = max(value - self.record_list[-1]['time'] + record['time'], 0)
            
        return

    def syncDataFrame(self):
        self.record_df = pd.DataFrame(self.record_list)
        return