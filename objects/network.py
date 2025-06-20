from libs.common import Common
from objects.roads import Roads
from objects.intersections import Intersections
from objects.links import Links
from objects.vehicle_inputs import VehicleInputs
from objects.vehicle_routing_decisions import VehicleRoutingDecisions
from objects.signal_heads import SignalHeads
from objects.signal_controllers import SignalControllers
from objects.queue_counters import QueueCounters
from objects.travel_time_measurements import TravelTimeMeasurements
from objects.delay_measurements import DelayMeasurements
from objects.data_collections import DataCollectionPoints, DataCollectionMeasurements
from objects.master_agents import MasterAgents
from objects.local_agents import LocalAgents
from objects.mpc_controllers import MpcControllers
from objects.bc_buffers import BcBuffers
from objects.bc_agent import BcAgent

class Network(Common):
    def __init__(self, vissim):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = vissim.config
        self.executor = vissim.executor

        # 上位の紐づくオブジェクトを取得
        self.vissim = vissim

        # 対応するComオブジェクトを取得
        self.com = self.vissim.com.Net

        # 制御手法を取得
        simulator_info = self.config.get('simulator_info')
        self.control_method = simulator_info['control_method']

        # 下位の紐づくオブジェクトを初期化
        self.roads = Roads(self)
        self.intersections = Intersections(self)
        self.links = Links(self)
        self.vehicle_inputs = VehicleInputs(self)
        self.vehicle_routing_decisions = VehicleRoutingDecisions(self)
        self.signal_heads = SignalHeads(self)
        self.signal_controllers = SignalControllers(self)
        self.queue_counters = QueueCounters(self)
        self.travel_time_measurements = TravelTimeMeasurements(self)
        self.delay_measurements = DelayMeasurements(self)
        self.data_collection_points = DataCollectionPoints(self)
        self.data_collection_measurements = DataCollectionMeasurements(self)

        
        if self.control_method == 'drl':
            # マスターエージェントとローカルエージェントを初期化
            self.master_agents = MasterAgents(self)
            self.local_agents = LocalAgents(self)

        elif self.control_method == 'mpc':
            # MPCコントローラを初期化
            self.mpc_controllers = MpcControllers(self)

            # 行動クローンのデータ集めをする場合はBCバッファを初期化
            mpc_info = self.config.get('mpc_info')
            bc_buffer_info = mpc_info['bc_buffer']
            self.bc_flg = bc_buffer_info['flg']
            if self.bc_flg:
                self.bc_buffers = BcBuffers(self)
        
        elif self.control_method == 'bc':
            self.bc_agent = BcAgent(self)

        # simulationオブジェクトと紐づける
        self.simulation = self.vissim.simulation
        self.simulation.set('network', self)

        # Vissimに各種パラメータを反映
        self._setParametersToVissim()

    def _setParametersToVissim(self):
        # 流入量をセット
        for vehicle_input in self.vehicle_inputs.getAll():
            input_volume = vehicle_input.link.get('input_volume')
            vehicle_input.com.SetAttValue('Volume(1)', input_volume)

        # 旋回率をセット
        for vehicle_routing_decision in self.vehicle_routing_decisions.getAll():
            for vehicle_route in vehicle_routing_decision.vehicle_routes.getAll():
                vehicle_route.com.SetAttValue('RelFlow(1)', vehicle_route.get('turn_ratio'))
    
    def updateData(self):
        # ネットワークの更新
        self.roads.updateData()
        self.queue_counters.updateData()
        self.delay_measurements.updateData()
        self.data_collection_measurements.updateData()

        # 並列処理が終わるまで待機
        self.executor.wait()

            

            


