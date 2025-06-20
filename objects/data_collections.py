from libs.container import Container
from libs.object import Object

import pandas as pd

class DataCollectionPoints(Container):
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
            self.com = self.network.com.DataCollectionPoints

            # 要素オブジェクトを初期化
            self.makeElements()
        
        elif upper_object.__class__.__name__ == 'Link':
            # 上位の紐づくオブジェクトを取得
            self.link = upper_object
        
        elif upper_object.__class__.__name__ == 'DataCollectionMeasurement':
            # 上位の紐づくオブジェクトを取得
            self.data_collection_measurement = upper_object
        
        return
    
    def makeElements(self):
        for data_collection_point_com in self.com.GetAll():
            self.add(DataCollectionPoint(data_collection_point_com, self))

class DataCollectionPoint(Object):
    def __init__(self, com, data_collection_points):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = data_collection_points.config
        self.executor = data_collection_points.executor

       
        # 上位の紐づくオブジェクトを取得
        self.data_collection_points = data_collection_points

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # networkオブジェクトに紐づける
        self.network = data_collection_points.network

        # linkオブジェクトと紐づける
        self._makeLinkConnection()

        # typeを定義
        self._makeType()

        # roadオブジェクトと紐づける
        self._makeRoadConnection()

        # data_collection_measurementsオブジェクトを初期化
        self.data_collection_measurements = DataCollectionMeasurements(self)
        
        return

    def _makeLinkConnection(self):
        # lane_idとlink_idを取得
        lane_com = self.com.Lane
        lane_id = lane_com.AttValue('Index')
        link_com = lane_com.Link
        link_id = link_com.AttValue('No')

        # linkオブジェクトとlaneオブジェクトを取得
        self.link = self.network.links[link_id]
        self.link.data_collection_points.add(self)
        self.lane = self.lane = self.link.lanes[lane_id]
        self.lane.set('data_collection_point', self)
        return
    
    def _makeType(self):
        # コネクタのとき
        if self.link.get('type') == 'connector':
            self.type = 'intersection'
            return
        
        if self.link.from_links.count() == 0:
            self.type = 'input'
        elif self.link.to_links.count() == 0:
            self.type = 'output'

        return

    def _makeRoadConnection(self):
        # 流入道路または流出道路の計測用のとき
        if self.type != 'intersection':
            self.road = self.link.road
            self.road.data_collection_points.add(self)
            return
        
        # 交差点の計測用のとき
        from_link = self.link.from_link
        self.road = from_link.road
        self.road.data_collection_points.add(self)

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
            self._makeElements()
        
        elif upper_object.__class__.__name__ == 'DataCollectionPoint':
            # 上位の紐づくオブジェクトを取得
            self.data_collection_point = upper_object
        
        return

    def _makeElements(self):
        for data_collection_measurement_com in self.com.GetAll():
            self.add(DataCollectionMeasurement(data_collection_measurement_com, self))
    
    def updateData(self):
        # Comオブジェクトからデータを更新
        measurement_ids = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('No')]
        veh_nums = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('Vehs(Current, Last, All)')]

        # データを要素オブジェクトにセット（非同期処理）
        for index, measurement_id in enumerate(measurement_ids):
            measurement = self[measurement_id]
            self.executor.submit(measurement.updateData, veh_nums[index])
    
class DataCollectionMeasurement(Object):
    def __init__(self, com, data_collection_measurements):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = data_collection_measurements.config
        self.executor = data_collection_measurements.executor

        # 上位の紐づくオブジェクトを取得
        self.data_collection_measurements = data_collection_measurements

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # networkオブジェクトと紐づける
        self.network = data_collection_measurements.network

        # data_collection_pointと紐づける
        self._makeDataCollectionPointConnection()

        # typeを定義
        self._makeType()

        # 自動車の通過台数を初期化
        self.current_num_vehs = 0
        self.num_vehs_record = pd.DataFrame(columns=['time', 'num_vehs'])
        return

    
    def _makeDataCollectionPointConnection(self):
        self.data_collection_points = DataCollectionPoints(self)

        for point_com in self.com.DataCollectionPoints.GetAll():
            point_id = point_com.AttValue('No')
            point = self.network.data_collection_points[point_id]
            self.data_collection_points.add(point)
            point.data_collection_measurements.add(self)

        return

    def _makeType(self):
        if self.data_collection_points.count() == 1:
            self.type = 'single'
        else:
            self.type = 'multiple'
        
        return
    
    def updateData(self, num_vehs):
        self.current_num_vehs = 0 if num_vehs is None else num_vehs
        self.num_vehs_record.loc[len(self.num_vehs_record)] = [self.current_time, self.current_num_vehs]
        return

    @property
    def current_time(self):
        return self.network.simulation.get('current_time')
