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
from objects.scoot_controllers2 import ScootControllers

import numpy as np
import torch
import pandas as pd
import re

class Network(Common):
    def __init__(self, vissim):
        super().__init__()

        self.config = vissim.config
        self.executor = vissim.executor
        self.shared_resources = vissim.shared_resources
        self.vissim = vissim

        # connect to simulation
        self.simulation = self.vissim.simulation
        self.simulation.set('network', self)

        return
    
    @property
    def current_time(self):
        return self.simulation.get('current_time')
    
    @property
    def simulation_count(self):
        return self.vissim.get('simulation_count')

    def _initProps(self):
        self.root_dir_path = self.vissim.get('root_dir_path')
        self.layout_dir_path = self.vissim.get('layout_dir_path')

        simulator_info = self.config.get('simulator_info')
        self.control_method = simulator_info['control_method']
        self.inflow_name = simulator_info['inflow_name']

        save_info = self.config.get('save_info')
        self.save_flg_map = save_info['common']['performance_metrics']
        self.road_scale_flg = save_info['common']['road_scale_flg']

        save_dir_path_map = self.config.get('save_dir_path_map')
        self.save_dir_path = save_dir_path_map['metrics']

        if self.control_method == 'drl':
            drl_info = self.config.get('drl_info')
            self.drl_framework = drl_info['framework']['type']
            self.drl_simulation_type = drl_info['simulation_type']

        return
    
    def _initVissimObjects(self):
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
            if self.drl_framework == 'apex':
                self.master_agents = MasterAgents(self, self.vissim.device)
                self.local_agents = LocalAgents(self, self.vissim.device)
            else:
                raise NotImplementedError(f"Not supported DRL framework: {self.drl_framework}")
            
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
    
    def activate(self):
        self.com = self.vissim.com.Net

        self._initProps()
        self._initVissimObjects()
        
        self._setParametersToVissim()
        return
    
    def update(self, type='normal'):
        self.roads.update()
        self.queue_counters.update()
        self.delay_measurements.update()
        self.data_collection_measurements.update()
        if type != 'normal':
            self.signal_controllers.updateRecord(type=type)
        self.executor.wait()
        return

    def save(self):
        # make time series data as dataframe
        self._makeQueueRecordsMap()
        self._makeDelayRecordsMap()
        self._makeSpeedRecordsMap()
        self._makeCalcTimeRecordsMap()
        self._makePhaseRecordsMap()

        # save drl information (only for train simulation, not for test simulation)
        if self.control_method == 'drl' and self.drl_simulation_type == 'train':
            if self.drl_framework == 'apex':
                self.master_agents.update('session')
                self.master_agents.save()
                return
            else:
                raise NotImplementedError(f"Not supported DRL framework: {self.drl_framework}")

        # save csv files
        self._saveCSV()
        return

    def _makeQueueRecordsMap(self):
        if self.save_flg_map['queue'] == False:
            return
        
        # make records_map
        self.queue_records_map = {
            'roads': {},
            'intersections': {},
        }

        self.queue_counters.sync('dataframe')
        for road in self.roads.getAll():
            if road.queue_counters.count() == 0:
                continue

            data_map = {}
            for queue_counter in road.queue_counters.getAll():
                link_type = queue_counter.link.get('type')
                tmp_record_df = queue_counter.get('record_df')
                if data_map == {}:
                    data_map['time'] = tmp_record_df['time'].values
                data_map[link_type] = tmp_record_df['value'].values

            record_df = pd.DataFrame(data_map)
            record_df['avg'] = record_df[[key for key in data_map.keys() if key != 'time']].mean(axis=1)
            record_df['max'] = record_df[[key for key in data_map.keys() if key != 'time']].max(axis=1)
            self.queue_records_map['roads'][road.get('id')] = record_df
        
        for intersection in self.intersections.getAll():
            data_map = {}
            for road in intersection.input_roads.getAll():
                if road.get('id') not in self.queue_records_map['roads']:
                    continue

                if data_map == {}:
                    data_map['time'] = self.queue_records_map['roads'][road.get('id')]['time'].values
                
                data_map[f"road_{road.get('id')}_max"] = self.queue_records_map['roads'][road.get('id')]['max'].values
            
            record_df = pd.DataFrame(data_map)
            record_df['avg'] = record_df[[key for key in data_map.keys() if re.match(rf"road_(\d+)_max", key)]].mean(axis=1)
            record_df['max'] = record_df[[key for key in data_map.keys() if re.match(rf"road_(\d+)_max", key)]].max(axis=1)
            record_df = record_df.drop(columns=[key for key in record_df.columns if re.match(rf"road_(\d+)_max", key)])

            self.queue_records_map['intersections'][intersection.get('id')] = record_df
        return

    def _makeDelayRecordsMap(self):
        if self.save_flg_map['delay'] == False:
            return
        
        self.delay_records_map = {
            'roads': {},
            'intersections': {},
        }

        self.delay_measurements.sync('dataframe')

        for road in self.roads.getAll():
            if road.delay_measurements.count() == 0:
                continue

            if not road.has('output_intersection'):
                continue
            intersection = road.output_intersection

            data_map = {}
            for delay_measurement in road.delay_measurements.getAll():
                route_id = delay_measurement.get('route_id')
                tmp_record_df = delay_measurement.get('record_df')
                if data_map == {}:
                    data_map['time'] = tmp_record_df['time'].values
                
                if f"route_{route_id}" not in data_map:
                    data_map[f"route_{route_id}"] = []
                data_map[f"route_{route_id}"].append(tmp_record_df['value'].values)
            
            # average delay for each direction (NaN is ignored in mean calculation)
            for key in data_map.keys():
                if key == 'time':
                    continue
                
                weighted_sum = np.zeros_like(data_map[key][0])
                total_weight = np.zeros_like(data_map[key][0])

                for array in data_map[key]:
                    mask = ~np.isnan(array)
                    weighted_sum[mask] += array[mask]
                    total_weight[mask] += 1
                
                data_map[key] = np.full(shape=weighted_sum.shape, fill_value=np.nan, dtype=np.float64)
                valid_mask = total_weight > 0
                data_map[key][valid_mask] = weighted_sum[valid_mask] / total_weight[valid_mask]
            
            record_df = pd.DataFrame(data_map)
            
            # make road-average delay (definition: using normal average, and NaN is ignored in calculation)
            weighted_sum = pd.Series(0.0, index=record_df.index)
            total_weight = pd.Series(0.0, index=record_df.index)
            valid_mask = pd.Series(True, index=record_df.index)
            for column in record_df.columns:
                if column == 'time':
                    continue
                
                mask = record_df[column].notna()
                weighted_sum.loc[mask] += record_df.loc[mask, column]
                total_weight.loc[mask] += 1
                valid_mask &= mask
            
            record_df['avg_1'] = np.nan
            record_df.loc[valid_mask, 'avg_1'] = weighted_sum[valid_mask] / total_weight[valid_mask]

            # make road-average delay (definition: using turn ratios as weights, and NaN is ignored in calculation)
            turn_ratios = road.get('turn_ratios')
            weighted_sum = pd.Series(0.0, index=record_df.index)
            total_weight = pd.Series(0.0, index=record_df.index)
            valid_mask = pd.Series(True, index=record_df.index)
            for route_id in range(1, intersection.get('num_roads')):
                turn_ratio = turn_ratios[route_id]
                mask = record_df[f"route_{route_id}"].notna()
                weighted_sum.loc[mask] += record_df.loc[mask, f"route_{route_id}"] * turn_ratio
                total_weight.loc[mask] += turn_ratio
                valid_mask &= mask

            record_df['avg_2'] = np.nan
            record_df.loc[valid_mask, 'avg_2'] = weighted_sum[valid_mask] / total_weight[valid_mask]

            # calculate max delay (if there is NaN, it is ignored in calculation)
            max_delay = pd.Series(0.0, index=record_df.index)
            valid_mask = pd.Series(True, index=record_df.index)
            for route_id in range(1, intersection.get('num_roads')):
                mask = record_df[f"route_{route_id}"].notna()
                max_delay.loc[mask] = np.maximum(max_delay.loc[mask], record_df.loc[mask, f"route_{route_id}"])
                valid_mask &= mask
            
            record_df['max'] = np.nan
            record_df.loc[valid_mask, 'max'] = max_delay[valid_mask]
            
            self.delay_records_map['roads'][road.get('id')] = record_df

        for intersection in self.intersections.getAll():
            data_map = {}
            for road_id, road in intersection.input_roads.items():
                if road.get('id') not in self.delay_records_map['roads']:
                    continue

                tmp_record_df = self.delay_records_map['roads'][road.get('id')]
                if data_map == {}:
                    data_map['time'] = tmp_record_df['time'].values
                
                data_map[f"road_{road_id}_avg_1"] = tmp_record_df['avg_1'].values
                data_map[f"road_{road_id}_avg_2"] = tmp_record_df['avg_2'].values
                data_map[f"road_{road_id}_max"] = tmp_record_df['max'].values

            if not data_map:
                continue
   
            record_df = pd.DataFrame(data_map)

            # calculate average delay (definition: using normal average, and NaN is ignored in calculation)
            weighted_sum = pd.Series(0.0, index=record_df.index)
            total_weight = pd.Series(0.0, index=record_df.index)
            valid_mask = pd.Series(True, index=record_df.index)
            for road_id, road in intersection.input_roads.items():
                mask = record_df[f"road_{road_id}_avg_1"].notna()   
                weighted_sum.loc[mask] += record_df.loc[mask, f"road_{road_id}_avg_1"]
                total_weight.loc[mask] += 1
                valid_mask &= mask
            
            record_df['avg_1'] = np.nan
            record_df.loc[valid_mask, 'avg_1'] = weighted_sum[valid_mask] / total_weight[valid_mask]

            # calculate average delay (definition: using input volumes as weights, and NaN is ignored in calculation)
            calc_flg = True
            for road_id, road in intersection.input_roads.items():
                if not road.has('input_volume'):
                    calc_flg = False
                    break
            
            if calc_flg:
                weighted_sum = pd.Series(0.0, index=record_df.index)
                total_weight = pd.Series(0.0, index=record_df.index)
                valid_mask = pd.Series(True, index=record_df.index)
                for road_id, road in intersection.input_roads.items():
                    input_volume = road.get('input_volume')

                    mask = record_df[f"road_{road_id}_avg_2"].notna()   
                    weighted_sum.loc[mask] += record_df.loc[mask, f"road_{road_id}_avg_2"] * input_volume
                    total_weight.loc[mask] += input_volume 
                    valid_mask &= mask

                record_df['avg_2'] = np.nan
                record_df.loc[valid_mask, 'avg_2'] = weighted_sum[valid_mask] / total_weight[valid_mask]

            # calculate max delay
            max_delay = pd.Series(0.0, index=record_df.index)
            valid_mask = pd.Series(True, index=record_df.index)
            for road_id in range(1, intersection.get('num_roads') + 1):
                mask = record_df[f"road_{road_id}_max"].notna()
                max_delay.loc[mask] = np.maximum(max_delay.loc[mask], record_df.loc[mask, f"road_{road_id}_max"])
                valid_mask &= mask
            
            record_df['max'] = np.nan
            record_df.loc[valid_mask, 'max'] = max_delay[valid_mask]

            # push to delay_records_map
            if calc_flg:
                final_df = record_df[['time', 'avg_1', 'avg_2', 'max']].copy()
            else:
                final_df = record_df[['time', 'avg_1', 'max']].copy()
            self.delay_records_map['intersections'][intersection.get('id')] = final_df

        return
    
    def _makeSpeedRecordsMap(self):
        if self.save_flg_map['speed'] == False:
            return
        
        self.speed_records_map = {
            'roads': {},
            'intersections': {},
        }

        self.roads.sync('dataframe')

        for road in self.roads.getAll():
            if not road.has('output_intersection'):
                continue
            
            record_df = road.get('speed_record_df')
            record_df = record_df.rename(columns={'value': 'avg'})
            self.speed_records_map['roads'][road.get('id')] = record_df

        for intersection in self.intersections.getAll():
            data_map = {}
            total_num_vehs_array = 0
            max_speed_list = []
            for road in intersection.input_roads.getAll():
                if road.get('id') not in self.speed_records_map['roads']:
                    continue

                if data_map == {}:
                    data_map['time'] = self.speed_records_map['roads'][road.get('id')]['time'].values
                
                avg_speed_array = self.speed_records_map['roads'][road.get('id')]['avg'].values
                
                tmp_record_df = road.get('num_vehs_record_df')
                num_vehs_array = tmp_record_df['value'].values
                data_map[f"road_{road.get('id')}_sum"] = avg_speed_array * num_vehs_array

                total_num_vehs_array += num_vehs_array
                max_speed_list.append(road.get('max_speed'))

            record_df = pd.DataFrame(data_map)
            numerator = record_df[[key for key in data_map.keys() if "_sum" in key]].sum(axis=1)
            denominator = total_num_vehs_array
            default_speed = np.mean(max_speed_list)
            record_df['avg'] = np.where(denominator > 0, numerator / denominator, default_speed)
            record_df = record_df.drop(columns=[key for key in record_df.columns if re.match(rf"road_(\d+)_sum", key)])
            self.speed_records_map['intersections'][intersection.get('id')] = record_df
        return

    def _makeCalcTimeRecordsMap(self):
        if not self.save_flg_map['calc_time'] or self.control_method == 'scoot':
            return

        self.calc_time_records_map = {}

        if self.control_method == 'mpc':
            self.mpc_controllers.sync('dataframe')
        elif self.control_method == 'drl':
            self.local_agents.sync(type='dataframe')
        else:
            raise NotImplementedError(f"Not supported control method: {self.control_method}")
        
        for intersection in self.intersections.getAll():
            if self.control_method == 'mpc':
                self.calc_time_records_map[intersection.get('id')] = intersection.mpc_controller.get('record_df')
            elif self.control_method == 'drl':
                self.calc_time_records_map[intersection.get('id')] = intersection.local_agent.get('calc_time_record_df')
            else:
                raise NotImplementedError(f"Not supported control method: {self.control_method}")
        
        return

    def _makePhaseRecordsMap(self):
        if not self.save_flg_map['phase']:
            return

        self.signal_controllers.sync('dataframe')
        self.phase_records_map = {}
        for intersection in self.intersections.getAll():
            self.phase_records_map[intersection.get('id')] = intersection.signal_controller.get('record_df')
        
        return

    def _saveCSV(self):
        # skip if all save flags are false
        if not any(flg for flg in self.save_flg_map.values()):
            return
        
        # intersection scale
        for intersection in self.intersections.getAll():
            save_dir_path = self.save_dir_path / f"intersection_{intersection.get('id')}"
            save_dir_path.mkdir(parents=True, exist_ok=True)

            data_map = {}
            if self.save_flg_map['queue']:
                tmp_record_df = self.queue_records_map['intersections'][intersection.get('id')]
                
                if data_map == {}:
                    data_map['time'] = tmp_record_df['time'].values
                
                data_map['queue_avg'] = tmp_record_df['avg'].values
                data_map['queue_max'] = tmp_record_df['max'].values
            
            if self.save_flg_map['delay']:
                tmp_record_df = self.delay_records_map['intersections'][intersection.get('id')]

                if data_map == {}:
                    data_map['time'] = tmp_record_df['time'].values
                
                data_map['delay_avg_1'] = tmp_record_df['avg_1'].values
                if 'avg_2' in tmp_record_df.columns:
                    data_map['delay_avg_2'] = tmp_record_df['avg_2'].values
                data_map['delay_max'] = tmp_record_df['max'].values
            
            if self.save_flg_map['speed']:
                tmp_record_df = self.speed_records_map['intersections'][intersection.get('id')]

                if data_map == {}:
                    data_map['time'] = tmp_record_df['time'].values
                
                data_map['speed_avg'] = tmp_record_df['avg'].values
            
            if self.save_flg_map['phase']:
                tmp_record_df = self.phase_records_map[intersection.get('id')]

                if data_map == {}:
                    data_map['time'] = tmp_record_df['time'].values
                
                data_map['phase'] = tmp_record_df['value'].values
            
            record_df = pd.DataFrame(data_map)
            record_df.to_csv(
                path_or_buf=save_dir_path / 'performance_metrics.csv', 
                float_format='%.2f',
                index=False
            )

            if self.save_flg_map['calc_time'] and self.control_method != 'scoot':
                tmp_record_df = self.calc_time_records_map[intersection.get('id')]
                tmp_record_df.to_csv(
                    path_or_buf=save_dir_path / 'calc_time.csv', 
                    float_format='%.2f',
                    index=False)

        # road scale
        if not self.road_scale_flg:
            return
        
        for intersection in self.intersections.getAll():
            for road_order_id, road in intersection.input_roads.items():
                save_dir_path = self.save_dir_path / f"intersection_{intersection.get('id')}" / f"road_{road_order_id}"
                save_dir_path.mkdir(parents=True, exist_ok=True)

                data_map = {}
                if self.save_flg_map['queue']:
                    tmp_record_df = self.queue_records_map['roads'][road.get('id')]
                    if data_map == {}:
                        data_map['time'] = tmp_record_df['time'].values
                    
                    data_map['queue_main'] = tmp_record_df['main'].values
                    if 'right' in tmp_record_df.columns:
                        data_map['queue_right'] = tmp_record_df['right'].values
                    if 'left' in tmp_record_df.columns:
                        data_map['queue_left'] = tmp_record_df['left'].values
                    
                    data_map['queue_avg'] = tmp_record_df['avg'].values
                    data_map['queue_max'] = tmp_record_df['max'].values
                    
                if self.save_flg_map['delay']:
                    tmp_record_df = self.delay_records_map['roads'][road.get('id')]
                    if data_map == {}:
                        data_map['time'] = tmp_record_df['time'].values
                    
                    route_list = []
                    for column in tmp_record_df.columns:
                        match_obj = re.match(rf"route_(\d+)", column)
                        if match_obj:
                            route_list.append(int(match_obj.group(1)))
                    
                    for route_id in route_list:
                        data_map[f"delay_route_{route_id}"] = tmp_record_df[f"route_{route_id}"].values
                    
                    data_map['delay_avg_1'] = tmp_record_df['avg_1'].values
                    data_map['delay_avg_2'] = tmp_record_df['avg_2'].values
                    data_map['delay_max'] = tmp_record_df['max'].values
                
                if self.save_flg_map['speed']:
                    tmp_record_df = self.speed_records_map['roads'][road.get('id')]
                    if data_map == {}:
                        data_map['time'] = tmp_record_df['time'].values
                    
                    data_map['speed'] = tmp_record_df['avg'].values
                
                record_df = pd.DataFrame(data_map)
                record_df.to_csv(
                    path_or_buf=save_dir_path / 'performance_metrics.csv', 
                    float_format='%.2f',
                    index=False
                )
        return
    
    



            

            


