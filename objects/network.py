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
from objects.scoot_controllers import ScootControllers

import numpy as np
import torch
import copy
import yaml
import re

class Network(Common):
    def __init__(self, vissim):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = vissim.config
        self.executor = vissim.executor
        self.shared_resources = vissim.shared_resources

        self.vissim = vissim
        self.com = self.vissim.com.Net

        self._initProps()

        self.simulation = self.vissim.simulation
        self.simulation.set('network', self)

        self._makeLowerObjects()
        self._initSaveDirPath()
        
        self._setParametersToVissim()
        return

    def _initProps(self):
        self.root_dir_path = self.vissim.get('root_dir_path')
        self.layout_dir_path = self.vissim.get('layout_dir_path')

        simulator_info = self.config.get('simulator_info')
        self.control_method = simulator_info['control_method']
        self.inflow_name = simulator_info['inflow_name']
        self.num_simulations = simulator_info['num_simulations']

        save_info = self.config.get('save_info')
        self.queue_flg = save_info['performance_metrics']['queue']
        self.delay_flg = save_info['performance_metrics']['delay']
        self.calc_time_flg = save_info['performance_metrics']['calc_time']
        self.phase_flg = save_info['performance_metrics']['phase']
        return
    
    def _initSaveDirPath(self):
        common_save_dir_path = self.root_dir_path / 'data' / 'performance_metrics'
        common_save_dir_path /= self.layout_dir_path.name
        common_save_dir_path /= self.inflow_name
        common_save_dir_path.mkdir(parents=True, exist_ok=True)

        # if the setting of simulator_info is same as the existing one, use the existing save_dir_path
        simulator_info = copy.deepcopy(self.config.get('simulator_info'))
        simulator_info = {key: simulator_info[key] for key in ['num_red_steps', 'simulation_time','time_step']}
        simulator_info['seed'] = self.simulation.get('random_seed')
        
        simulation_dir_path = None
        for tmp_dir_path in common_save_dir_path.glob('simulator_*'):
            config_file_path = tmp_dir_path / 'config.yaml'
            if not config_file_path.exists():
                continue

            with config_file_path.open('rb') as f:  
                config_yaml = yaml.safe_load(f)
            
            if config_yaml == simulator_info:
                simulation_dir_path = tmp_dir_path
                break
        
        if simulation_dir_path is None:
            config_idx = 1
            while True:
                tmp_dir_path = common_save_dir_path / f"simulator_{config_idx}"
                if not tmp_dir_path.exists():
                    simulation_dir_path = tmp_dir_path
                    simulation_dir_path.mkdir(parents=True, exist_ok=False)

                    with open(simulation_dir_path / 'config.yaml', 'w') as f:
                        yaml.dump(simulator_info, f)
                    break

                config_idx += 1

        control_method_dir_path = simulation_dir_path / self.control_method
        if self.control_method == 'mpc':
            mpc_info = copy.deepcopy(self.config.get('mpc_info'))
            del mpc_info['bc_buffer']

            num_roads_set = set()
            for intersection in self.intersections.getAll():
                num_roads_set.add(intersection.get('num_roads'))
            
            for intersection_type in copy.deepcopy(mpc_info['phases']).keys():
                num_roads = int(re.match(rf"(\d+)-road", intersection_type)[1])
                if num_roads not in num_roads_set:
                    del mpc_info['phases'][intersection_type]

            self.save_dir_path = None
            for tmp_dir_path in control_method_dir_path.glob('config_*'):
                config_file_path = tmp_dir_path / 'config.yaml'
                if not config_file_path.exists():
                    continue

                with config_file_path.open('rb') as f:  
                    config_yaml = yaml.safe_load(f)
                
                if config_yaml == mpc_info:
                    self.save_dir_path = tmp_dir_path
                    break

            if self.save_dir_path is None:
                config_idx = 1
                while True:
                    tmp_dir_path = control_method_dir_path / f"config_{config_idx}"
                    if not tmp_dir_path.exists():
                        self.save_dir_path = tmp_dir_path
                        self.save_dir_path.mkdir(parents=True, exist_ok=False)

                        with open(self.save_dir_path / 'config.yaml', 'w') as f:
                            config_yaml = mpc_info
                            yaml.dump(config_yaml, f)
                        break
                    config_idx += 1
            
        else:
            raise NotImplementedError(f"Not supported control method: {self.control_method}")
        
        return
    
    def _makeLowerObjects(self):
        # make and set vissim objects
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
            # set master_agents and local_agents objects
            device = self.simulation.get('device')
            self.master_agents = MasterAgents(self, device)
            self.local_agents = LocalAgents(self, device)
            
        elif self.control_method == 'mpc':
            # set mpc_controllers object
            self.mpc_controllers = MpcControllers(self)

            # 行動クローンのデータ集めをする場合はBCバッファを初期化
            mpc_info = self.config.get('mpc_info')
            bc_buffer_info = mpc_info['bc_buffer']
            self.bc_flg = bc_buffer_info['flg']
            if self.bc_flg:
                self.bc_buffers = BcBuffers(self)
        
        elif self.control_method == 'bc':
            # set bc_agent object
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bc_agent = BcAgent(self, device)
        
        elif self.control_method == 'scoot':
            # set scoot_controllers object
            self.scoot_controllers = ScootControllers(self)

        else:
            raise NotImplementedError(f"Not supported control method: {self.control_method}")
        
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
        if self.control_method == 'mpc':
            for intersection_id in self.intersections.getKeys(container_flg=True, sorted_flg=True):
                intersection = self.intersections[intersection_id]
                controller = intersection.mpc_controller
                input_roads = intersection.input_roads

                tmp_save_dir_path = self.save_dir_path / f"intersection_{intersection_id}"
                tmp_save_dir_path.mkdir(parents=True, exist_ok=True)

                if self.queue_flg:
                    # save max_queue.csv and average_queue.csv
                    max_queue_df = None
                    average_queue_df = None
                    for road_order_id in range(1, input_roads.count() + 1):
                        road = input_roads[road_order_id]
                        
                        tmp_max_queue_df = None
                        tmp_average_queue_df = None
                        for queue_counter in road.queue_counters.getAll():
                            tmp_queue_df = copy.deepcopy(queue_counter.get('queue_length_record'))

                            if tmp_max_queue_df is None:
                                tmp_max_queue_df = tmp_queue_df
                            else:
                                tmp_max_queue_df['queue_length'] = np.maximum(
                                    tmp_max_queue_df['queue_length'].to_numpy(),
                                    tmp_queue_df['queue_length'].to_numpy(),
                                )

                            if tmp_average_queue_df is None:
                                tmp_average_queue_df = tmp_queue_df
                            else:
                                tmp_average_queue_df['queue_length'] += tmp_queue_df['queue_length']
                        
                        # sum to average
                        tmp_average_queue_df['queue_length'] /= road.queue_counters.count()
                        
                        # update max_queue_df
                        if max_queue_df is None:
                            max_queue_df = tmp_max_queue_df
                        else:
                            max_queue_df['queue_length'] = np.maximum(
                                max_queue_df['queue_length'].to_numpy(),
                                tmp_max_queue_df['queue_length'].to_numpy(),
                            )
                        
                        # update average_queue_df
                        if average_queue_df is None:
                            average_queue_df = tmp_average_queue_df
                        else:
                            average_queue_df['queue_length'] += tmp_average_queue_df['queue_length']
                    
                    # sum to average
                    average_queue_df['queue_length'] /= input_roads.count()

                    # save
                    max_queue_df.to_csv(tmp_save_dir_path / 'max_queue.csv', index=False)
                    average_queue_df.to_csv(tmp_save_dir_path / 'average_queue.csv', index=False)   

                if self.delay_flg:
                    # save max_delay.csv and average_delay.csv
                    max_delay_df = None
                    average_delay_df = None

                    for road_order_id in range(1, input_roads.count() + 1):
                        road = input_roads[road_order_id]

                        tmp_max_delay_df = None
                        tmp_average_delay_df = None
                        for delay_measurement in road.delay_measurements.getAll():
                            tmp_delay_df = copy.deepcopy(delay_measurement.get('delay_record'))

                            if tmp_max_delay_df is None:
                                tmp_max_delay_df = tmp_delay_df
                            else:
                                tmp_max_delay_df['delay'] = np.maximum(
                                    tmp_max_delay_df['delay'].to_numpy(),
                                    tmp_delay_df['delay'].to_numpy(),
                                )
                            
                            if tmp_average_delay_df is None:
                                tmp_average_delay_df = tmp_delay_df
                            else:
                                tmp_average_delay_df['delay'] += tmp_delay_df['delay']
                        
                        # sum to average
                        tmp_average_delay_df['delay'] /= road.delay_measurements.count()

                        # update max_delay_df
                        if max_delay_df is None:
                            max_delay_df = tmp_max_delay_df
                        else:
                            max_delay_df['delay'] = np.maximum(
                                max_delay_df['delay'].to_numpy(),
                                tmp_max_delay_df['delay'].to_numpy(),
                            )

                        # update average_delay_df
                        if average_delay_df is None:
                            average_delay_df = tmp_average_delay_df
                        else:
                            average_delay_df['delay'] += tmp_average_delay_df['delay']
                    
                    # sum to average
                    average_delay_df['delay'] /= input_roads.count()

                    # save
                    max_delay_df.to_csv(tmp_save_dir_path / 'max_delay.csv', index=False)
                    average_delay_df.to_csv(tmp_save_dir_path / 'average_delay.csv', index=False)

                if self.phase_flg:
                    signal_controller = intersection.signal_controller
                    phase_df = signal_controller.get('phase_record_df')
                    phase_df.to_csv(tmp_save_dir_path / 'phases.csv', index=False)
                
                if self.calc_time_flg:
                    calc_time_df = controller.get('calc_time_record')
                    calc_time_df.to_csv(tmp_save_dir_path / 'calc_time.csv', index=False)
        return



            

            


