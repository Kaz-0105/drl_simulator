from libs.container import Container
from libs.object import Object

class DelayMeasurements(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = network.config
        self.network = network

        # 対応するComオブジェクトを取得
        self.com = self.network.com.DelayMeasurements

        # 要素オブジェクトの初期化
        self.makeElements()
    
    def makeElements(self):
        for delay_measurement_com in self.com.GetAll():
            self.add(DelayMeasurement(delay_measurement_com, self))

class DelayMeasurement(Object):
    def __init__(self, com, delay_measurements):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = delay_measurements.config
        self.delay_measurements = delay_measurements

        # 対応するComオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')