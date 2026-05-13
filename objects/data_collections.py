from libs.container import Container
from libs.object import Object

import pandas as pd

class DataCollectionPoints(Container):
    def __init__(self, upper_object):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.com = self.network.com.DataCollectionPoints

            self._initElements(upper_object)
        
        elif upper_object.__class__.__name__ == 'Link':
            self.link = upper_object
        
        elif upper_object.__class__.__name__ == 'DataCollectionMeasurement':
            self.data_collection_measurement = upper_object
            self.network = self.data_collection_measurement.network

            self._initElements(upper_object)
        
        return
    
    def _initElements(self, upper_object):
        if upper_object.__class__.__name__ == 'Network':
            for data_collection_point_com in self.com.GetAll():
                self.add(DataCollectionPoint(data_collection_point_com, self))

        elif upper_object.__class__.__name__ == 'DataCollectionMeasurement':
            for data_collection_point_com in self.data_collection_measurement.com.DataCollectionPoints.GetAll():
                data_collection_point = self.network.data_collection_points[data_collection_point_com.AttValue('No')]
                self.add(data_collection_point)
                data_collection_point.data_collection_measurements.add(self.data_collection_measurement)

        else:
            raise NotImplementedError(f"Not supported upper_object class: {upper_object.__class__.__name__}")
        
        return

class DataCollectionPoint(Object):
    def __init__(self, com, data_collection_points):
        super().__init__()

        self.config = data_collection_points.config
        self.executor = data_collection_points.executor
        self.data_collection_points = data_collection_points
        self.network = data_collection_points.network

        self.com = com

        self._initProps()
        self._connectObjects()
        return
    
    def _initProps(self):
        self.id = self.com.AttValue('No')
        self.type = None # initialized after connecting to link object
        return
    
    def _connectObjects(self):
        # set link and lane
        self.link = self.network.links[self.com.Lane.Link.AttValue('No')]
        self.link.data_collection_points.add(self)
        self.lane = self.link.lanes[self.com.Lane.AttValue('Index')]
        self.lane.data_collection_point = self

        # set data_collection_measurements (DataCollectionMeasurement._connectObjects())
        self.data_collection_measurements = DataCollectionMeasurements(self)

        # set type
        if self.link.get('type') == 'connector':
            self.type = 'intersection'
        else:
            num_from_links = self.link.from_links.count()
            num_to_links = self.link.to_links.count()
            if num_from_links == 0 or num_from_links < num_to_links:
                self.type = 'input'
            elif num_to_links == 0 or num_to_links < num_from_links:
                self.type = 'output'
            else:
                raise NotImplementedError(f"Unsupported link connection for DataCollectionPoint {self.id} on Link {self.link.get('id')}: num_from_links={num_from_links}, num_to_links={num_to_links}")
        
        # set vehicle_route and signal_head
        if self.link.get('type') == 'connector':
            self.vehicle_route = self.link.vehicle_route
            self.signal_head = self.link.signal_head

        # set road
        if self.type == 'intersection':
            self.road = self.link.from_link.road
            self.road.data_collection_points.add(self)
        elif self.type in ['input', 'output']:
            self.road = self.link.road
            self.road.data_collection_points.add(self)
        else:
            raise NotImplementedError(f"Not supported data collection point type: {self.type}")

        return
    
class DataCollectionMeasurements(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            # 上位の紐づくオブジェクトを取得
            self.network = upper_object

            # comオブジェクトを取得
            self.com = self.network.com.DataCollectionMeasurements

            # 要素オブジェクトを初期化
            self._initElements()
        
        elif upper_object.__class__.__name__ == 'DataCollectionPoint':
            # 上位の紐づくオブジェクトを取得
            self.data_collection_point = upper_object
        
        return

    def _initElements(self):
        for data_collection_measurement_com in self.com.GetAll():
            self.add(DataCollectionMeasurement(data_collection_measurement_com, self))
    
    def update(self):
        # Comオブジェクトからデータを更新
        measurement_ids = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('No')]
        veh_nums = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('Vehs(Current, Last, All)')]

        # データを要素オブジェクトにセット（非同期処理）
        for index, measurement_id in enumerate(measurement_ids):
            measurement = self[measurement_id]
            self.executor.submit(measurement.update, veh_nums[index])
    
class DataCollectionMeasurement(Object):
    def __init__(self, com, data_collection_measurements):
        super().__init__()

        self.config = data_collection_measurements.config
        self.executor = data_collection_measurements.executor
        self.data_collection_measurements = data_collection_measurements
        self.network = data_collection_measurements.network

        self.com = com

        self._initProps()

        self.data_collection_points = DataCollectionPoints(self)
        return
    
    @property
    def type(self):
        if self.data_collection_points.count() == 1:
            return 'single'
        else:
            return 'multiple'
        
    @property
    def current_time(self):
        return self.network.simulation.get('current_time')

    def _initProps(self):
        self.id = self.com.AttValue('No')

        self.current_num_vehs = 0
        self.num_vehs_record = pd.DataFrame(columns=['time', 'num_vehs'])
        return
    
    def update(self, num_vehs):
        self.current_num_vehs = 0 if num_vehs is None else num_vehs
        self.num_vehs_record.loc[len(self.num_vehs_record)] = [self.current_time, self.current_num_vehs]
        return

    
