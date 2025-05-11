from libs.container import Container
from libs.object import Object

class TravelTimeMeasurements(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = network.config
        self.network = network

        # 対応するComオブジェクトを取得
        self.com = self.network.com.VehicleTravelTimeMeasurements

        # 要素オブジェクトの初期化
        self.makeElements()
    
    def makeElements(self):
        for travel_time_measurement_com in self.com.GetAll():
            self.add(TravelTimeMeasurement(travel_time_measurement_com, self))

class TravelTimeMeasurement(Object):
    def __init__(self, com, travel_time_measurements):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = travel_time_measurements.config
        self.travel_time_measurements = travel_time_measurements

        # 対応するComオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')