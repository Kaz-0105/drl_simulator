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

        # travel_time_measurementと紐づける
        self.makeTravelTimeConnections()
    
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