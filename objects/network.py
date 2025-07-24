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

import numpy as np
from pathlib import Path
import pickle
import torch

class Network(Common):
    def __init__(self, vissim):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = vissim.config
        self.executor = vissim.executor
        self.shared_resources = vissim.shared_resources

        # 上位の紐づくオブジェクトを取得
        self.vissim = vissim

        # 対応するComオブジェクトを取得
        self.com = self.vissim.com.Net

        # 制御手法を取得
        simulator_info = self.config.get('simulator_info')
        self.control_method = simulator_info['control_method']

        # 保存するデータに関するフラグを設定
        self._getSaveParams()

        # 下位のオブジェクトを初期化
        self._makeLowerObjects()

        # simulationオブジェクトと紐づける
        self.simulation = self.vissim.simulation
        self.simulation.set('network', self)

        # Vissimに各種パラメータを反映
        self._setParametersToVissim()

    def _getSaveParams(self):
        records_info = self.config.get('records_info')
        self.record_flg = records_info['record_flg']
        self.queue_flg = records_info['metric']['queue_flg']
        self.delay_flg = records_info['metric']['delay_flg']
        self.calc_time_flg = records_info['metric']['calc_time_flg']
        self.old_definition_flg = records_info['old_definition_flg']
        return
    
    def _makeLowerObjects(self):
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
            # PyTorchのデバイスを設定
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # マスターエージェントとローカルエージェントを初期化
            self.master_agents = MasterAgents(self, device)
            self.local_agents = LocalAgents(self, device)
            
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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 行動クローンエージェントを初期化
            self.bc_agent = BcAgent(self, device)
        
        return
    
    def _setParametersToVissim(self):
        # 流入量をセット
        for vehicle_input in self.vehicle_inputs.getAll():
            input_volume = vehicle_input.link.get('input_volume')
            vehicle_input.com.SetAttValue('Volume(1)', input_volume)

        # 旋回率をセット
        for vehicle_routing_decision in self.vehicle_routing_decisions.getAll():
            for vehicle_route in vehicle_routing_decision.vehicle_routes.getAll():
                vehicle_route.com.SetAttValue('RelFlow(1)', vehicle_route.get('turn_ratio'))
        
        return
    
    def updateData(self):
        # ネットワークの更新
        self.roads.updateData()
        self.queue_counters.updateData()
        self.delay_measurements.updateData()
        self.data_collection_measurements.updateData()

        # 並列処理が終わるまで待機
        self.executor.wait()
        return

    def saveData(self):
        # データを保存するフラグが立っていない場合は何もしない
        if not self.record_flg:
            return
        
        # save_dataの開始インデックスを取得
        common_save_path_name = 'results/metrics/metric'

        current_idx = 0
        while True:
            current_idx += 1
            save_path = Path(f"{common_save_path_name}_{current_idx}.pkl")
            if not save_path.exists():
                break

        # 交差点を走査
        for intersection_id in self.intersections.getKeys(container_flg=True, sorted_flg=True): 
            intersection = self.intersections[intersection_id]

            input_roads = intersection.input_roads

            # セーブするデータをまとめる
            save_data = {
                'max_queue': None,
                'average_queue': None,
                'max_delay': None,  
                'average_delay': None,
                'phase': None,
                'calc_time': None,
            }

            # キューの最大長と平均長を計算
            if self.queue_flg:
                queue_length_record = None
                for road_order_id in range(1, input_roads.count() + 1):
                    road = input_roads[road_order_id]

                    for queue_counter in road.queue_counters.getAll():
                        tmp_queue_length_record = queue_counter.get('queue_length_record')

                        if queue_length_record is None:
                            queue_length_record = tmp_queue_length_record
                            continue
                        
                        queue_length_record['queue_length'] = np.maximum(
                            queue_length_record['queue_length'].to_numpy(),
                            tmp_queue_length_record['queue_length'].to_numpy(),
                        )
                
                save_data['max_queue'] = queue_length_record.copy()

                road_queue_length_record_map = {}
                for road_order_id in range(1, input_roads.count() + 1):
                    road = input_roads[road_order_id]
                    queue_length_record = None
                    for queue_counter in road.queue_counters.getAll():
                        tmp_queue_length_record = queue_counter.get('queue_length_record')
                        if queue_length_record is None:
                            queue_length_record = tmp_queue_length_record
                            continue
                        queue_length_record['queue_length'] += tmp_queue_length_record['queue_length']
                    
                    # 古い定義では，各道路ごとでは平均化しない（現在の定義では平均化する）
                    if not self.old_definition_flg:
                        queue_length_record['queue_length'] /= road.queue_counters.count()

                    road_queue_length_record_map[road_order_id] = queue_length_record
                
                queue_length_record = None
                for road_order_id in range(1, input_roads.count() + 1):
                    tmp_queue_length_record = road_queue_length_record_map[road_order_id]
                    if queue_length_record is None:
                        queue_length_record = tmp_queue_length_record
                        continue
                    queue_length_record['queue_length'] += tmp_queue_length_record['queue_length']

                queue_length_record['queue_length'] /= input_roads.count()
                save_data['average_queue'] = queue_length_record.copy()

            # 遅延を計算
            if self.delay_flg:
                delay_record = None
                for road_order_id in range(1, input_roads.count() + 1):
                    road = input_roads[road_order_id]

                    for delay_measurement in road.delay_measurements.getAll():
                        tmp_delay_record = delay_measurement.get('delay_record')
                        
                        if delay_record is None:
                            delay_record = tmp_delay_record
                            continue

                        delay_record['delay'] = np.maximum(
                            delay_record['delay'].to_numpy(),
                            tmp_delay_record['delay'].to_numpy(),
                        )
                
                save_data['max_delay'] = delay_record.copy()

                road_delay_record_map = {}
                for road_order_id in range(1, input_roads.count() + 1):
                    road = input_roads[road_order_id]
                    delay_record = None
                    for delay_measurement in road.delay_measurements.getAll():
                        tmp_delay_record = delay_measurement.get('delay_record')
                        if delay_record is None:
                            delay_record = tmp_delay_record
                            continue
                        delay_record['delay'] += tmp_delay_record['delay']
                    
                    delay_record['delay'] /= road.delay_measurements.count()
                    road_delay_record_map[road_order_id] = delay_record
                
                delay_record = None
                for road_order_id in range(1, input_roads.count() + 1):
                    tmp_delay_record = road_delay_record_map[road_order_id]
                    if delay_record is None:
                        delay_record = tmp_delay_record
                        continue
                    delay_record['delay'] += tmp_delay_record['delay']

                delay_record['delay'] /= input_roads.count()
                save_data['average_delay'] = delay_record.copy()

            # 計算時間を記録
            if self.calc_time_flg:
                if intersection.has('local_agent'):
                    local_agent = intersection.local_agent
                    calc_time_record = local_agent.get('calc_time_record')
        
                elif intersection.has('mpc_controller'):
                    mpc_controller = intersection.mpc_controller
                    calc_time_record = mpc_controller.get('calc_time_record')
                
                save_data['calc_time'] = calc_time_record

            # データを保存
            save_path = Path(f"{common_save_path_name}_{current_idx}.pkl")
            with save_path.open('wb') as f:
                pickle.dump(save_data, f)
            
            # 次のインデックスを更新
            current_idx += 1

        return



            

            


