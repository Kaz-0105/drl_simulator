from libs.container import Container
from libs.object import Object

class DataCollectionPoints(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = network.config
        self.network = network

        # comオブジェクトを取得
        self.com = self.network.com.DataCollectionPoints

        # 要素オブジェクトを初期化
        self.makeElements()
    
    def makeElements(self):
        for data_collection_point_com in self.com.GetAll():
            self.add(DataCollectionPoint(data_collection_point_com, self))

class DataCollectionPoint(Object):
    def __init__(self, com, data_collection_points):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = data_collection_points.config
        self.data_collection_points = data_collection_points

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')


class DataCollectionMeasurements(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = network.config
        self.network = network

        # comオブジェクトを取得
        self.com = self.network.com.DataCollectionMeasurements

        # 要素オブジェクトを初期化
        self.makeElements()

    def makeElements(self):
        for data_collection_measurement_com in self.com.GetAll():
            self.add(DataCollectionMeasurement(data_collection_measurement_com, self))
    
class DataCollectionMeasurement(Object):
    def __init__(self, com, data_collection_measurements):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = data_collection_measurements.config
        self.data_collection_measurements = data_collection_measurements

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        