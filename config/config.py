from libs.common import Common 

import yaml
import pandas as pd
import re
import copy
import random

class Config(Common):
    def __init__(self, vissim):
        # 継承
        super().__init__()

        self.vissim = vissim
        self.root_dir_path = self.vissim.get('root_dir_path')

        self._initProps()

        # 行動クローンのときは1交差点系限定とする
        self._validateBcEnvironment()

        # epsilonのスケジューリングを取得
        self._getEpsilonSchedule()

        # 状態のtemplateを取得
        self._getNetworkStateMap()

        # drl_infoの整形
        self.reshapeDrlInfo()

        # initialize save_dir_path_map
        self._initSaveDirPathMap()

        # validation
        self._validation()
        return
    
    def _initProps(self):
        with open(self.root_dir_path / 'config' / 'config.yaml', 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

            self.simulator_info = data['simulator']

            self.drl_info = data['drl']

            self.apex_info = data['apex']
            if type(data['apex']['learning_rate']) == str:
                self.apex_info['learning_rate'] = float(data['apex']['learning_rate'])  # 文字列になってしまうのでfloatに変換
            if type(data['apex']['weight_decay']) == str:
                self.apex_info['weight_decay'] = float(data['apex']['weight_decay'])  # 文字列になってしまうのでfloatに変換

            self.mpc_info = data['mpc']

            self.bc_info = data['bc']
            if type(self.bc_info['weight_decay']) == str:
                self.bc_info['weight_decay'] = float(self.bc_info['weight_decay'])  # 文字列になってしまうのでfloatに変換
            if type(self.bc_info['learning_rate']) == str:
                self.bc_info['learning_rate'] = float(self.bc_info['learning_rate'])  # 文字列になってしまうのでfloatに変換

            self.scoot_info = data['scoot']

            self.save_info = data['save']

        # set seed
        if self.simulator_info['seed']['is_random']:
            self.simulator_info['seed'] = random.randint(100 + 1, 10000)
        else:
            self.simulator_info['seed'] = self.simulator_info['seed']['value'] 

        # set layout dir path
        self.layout_dir_path = self.root_dir_path / 'layout' / self.simulator_info['layout_name']

        # set network information
        with open(self.layout_dir_path / 'roads.csv', 'r', encoding='utf-8') as f:
            self.roads = pd.read_csv(f, index_col=False)
        
        with open(self.layout_dir_path / 'road_link_tags.csv', 'r', encoding='utf-8') as f:
            self.road_link_tags = pd.read_csv(f, index_col=False)
        
        with open(self.layout_dir_path / 'intersections.csv', 'r', encoding='utf-8') as f:
            self.intersections = pd.read_csv(f, index_col=False)

        with open(self.layout_dir_path / 'intersection_road_tags.csv', 'r', encoding='utf-8') as f:
            self.intersection_road_tags = pd.read_csv(f, index_col=False)

        with open(self.layout_dir_path / 'intersection_turn_ratio_tags.csv', 'r', encoding='utf-8') as f:
            self.intersection_turn_ratio_tags = pd.read_csv(f, index_col=False) 

        link_input_tags_dir_path = self.layout_dir_path / 'link_input_tags'
        if not link_input_tags_dir_path.exists():
            raise FileNotFoundError(f"Not found: {link_input_tags_dir_path}")
        
        self.link_input_tags_map = {}
        for link_input_tags_file_path in link_input_tags_dir_path.glob('*.csv'):
            with open(link_input_tags_file_path, 'r', encoding='utf-8') as f:
                self.link_input_tags_map[link_input_tags_file_path.stem] = pd.read_csv(f, index_col=False)
        
        # set num_roads_turn_ratio_map
        self.num_roads_turn_ratio_map = {}
        turn_ratio_dir_path = self.root_dir_path / 'config' / 'meta' / 'turn_ratio'
        for tmp_file_path in turn_ratio_dir_path.glob('turn_ratio_*.csv'):
            num_roads = int(re.match(rf"turn_ratio_(\d+).csv", tmp_file_path.name)[1])
            self.num_roads_turn_ratio_map[num_roads] = pd.read_csv(tmp_file_path, index_col=False)
        
        phases_dir_path = self.root_dir_path / 'config' / 'meta' / 'phases'

        # set num_roads_phases_map
        self.num_roads_phases_map = {}
        for tmp_file_path in phases_dir_path.glob('phases_*.csv'):
            num_roads = int(re.match(rf"phases_(\d+).csv", tmp_file_path.name)[1])
            self.num_roads_phases_map[num_roads] = pd.read_csv(tmp_file_path, index_col=False)

        # set symmetry_phase_tags
        self.symmetry_phase_tags = {}
        for tmp_file_path in phases_dir_path.glob('symmetry_phase_tags_*.csv'):
            num_roads = int(re.match(rf"symmetry_phase_tags_(\d+).csv", tmp_file_path.name)[1])
            self.symmetry_phase_tags[num_roads] = pd.read_csv(tmp_file_path, index_col=False)
        return
    
    def _validateBcEnvironment(self):
        if self.simulator_info['control_method'] != 'bc':
            return
        
        if self.intersections.shape[0] != 1:
            raise ValueError('The simulation of behavior cloning is only available for a single intersection environment. \n Please check the network configuration in config.yaml.')

    def _getEpsilonSchedule(self):
        if not self.apex_info['epsilon']['schedule_flg']:
            return
        
        eps_schedule_file_path = self.root_dir_path / 'layout' / 'epsilon_schedule.csv'
        if not eps_schedule_file_path.exists():
            raise FileNotFoundError(f"Not found: {eps_schedule_file_path}")
        self.epsilon_schedule = pd.read_csv(eps_schedule_file_path, index_col=False)
        return

    def _getNetworkStateMap(self):
        self.network_state_map = {}

        for network_id in [1]:
            self.network_state_map[network_id] = pd.read_csv(self.root_dir_path / 'layout' / f"state_template{network_id}.csv", index_col=False)
        return
    
    def reshapeDrlInfo(self):
        state_template_df = self.network_state_map[self.drl_info['network_id']]
        target_record = state_template_df[state_template_df['id'] == self.drl_info['state_id']]
        if target_record.empty:
            raise ValueError(f"State ID {self.drl_info['state_id']} not found in state_template{self.drl_info['network_id']}.csv")
        
        self.drl_info['features'] = {
            'vehicle' : {
                'position': bool(target_record['position'].values[0]),
                'speed': bool(target_record['speed'].values[0]),
                'direction': bool(target_record['direction'].values[0])
            },
            'lane' : {
                'metric': {
                    'num_vehicles': bool(target_record['num_vehicles'].values[0]),
                },
                'shape' : {
                    'length': bool(target_record['lane_length'].values[0]),
                    'type': bool(target_record['lane_type'].values[0])
                }
            }, 
            'road' : {
                'metric': {
                    'queue_length': bool(target_record['queue_length'].values[0]),
                    'delay': bool(target_record['delay'].values[0])
                }
            }
        }
        return
    
        
    def _initSaveDirPathMap(self):
        self.save_dir_path_map = {}

        common_save_dir_path = self.root_dir_path / 'data' / 'performance_metrics'
        common_save_dir_path /= self.layout_dir_path.name
        common_save_dir_path /= self.simulator_info['inflow_name']
        common_save_dir_path.mkdir(parents=True, exist_ok=True)

        # if the setting of simulator_info is same as the existing one, use the existing save_dir_path
        simulator_info = copy.deepcopy(self.simulator_info)
        simulator_info = {key: simulator_info[key] for key in ['num_red_steps', 'simulation_time','time_step', 'seed']}
        
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

        control_method_dir_path = simulation_dir_path / self.simulator_info['control_method']
        if self.simulator_info['control_method'] == 'mpc':
            mpc_info = copy.deepcopy(self.mpc_info)
            del mpc_info['bc_buffer']

            num_roads_set = set()
            for _, intersection_row in self.intersections.iterrows():
                num_roads_set.add(intersection_row['num_roads'])
            
            for intersection_type in copy.deepcopy(mpc_info['phases']).keys():
                num_roads = int(re.match(rf"(\d+)-road", intersection_type)[1])
                if num_roads not in num_roads_set:
                    del mpc_info['phases'][intersection_type]

            save_dir_path = None
            for tmp_dir_path in control_method_dir_path.glob('config_*'):
                config_file_path = tmp_dir_path / 'config.yaml'
                if not config_file_path.exists():
                    continue

                with config_file_path.open('rb') as f:  
                    config_yaml = yaml.safe_load(f)
                
                if config_yaml == mpc_info:
                    save_dir_path = tmp_dir_path
                    break

            if save_dir_path is None:
                config_idx = 1
                while True:
                    tmp_dir_path = control_method_dir_path / f"config_{config_idx}"
                    if not tmp_dir_path.exists():
                        save_dir_path = tmp_dir_path
                        save_dir_path.mkdir(parents=True, exist_ok=False)

                        with open(save_dir_path / 'config.yaml', 'w') as f:
                            config_yaml = mpc_info
                            yaml.dump(config_yaml, f)
                        break
                    config_idx += 1

            self.save_dir_path_map['metrics'] = save_dir_path

        elif self.simulator_info['control_method'] == 'scoot':
            scoot_info = copy.deepcopy(self.scoot_info)

            save_dir_path = None
            for tmp_dir_path in control_method_dir_path.glob('config_*'):
                config_file_path = tmp_dir_path / 'config.yaml'
                if not config_file_path.exists():
                    continue

                with config_file_path.open('rb') as f:  
                    config_yaml = yaml.safe_load(f)
                
                if config_yaml == scoot_info:
                    save_dir_path = tmp_dir_path
                    break
            
            if save_dir_path is None:
                config_idx = 1
                while True:
                    tmp_dir_path = control_method_dir_path / f"config_{config_idx}"
                    if not tmp_dir_path.exists():
                        save_dir_path = tmp_dir_path
                        save_dir_path.mkdir(parents=True, exist_ok=False)

                        with open(save_dir_path / 'config.yaml', 'w') as f:
                            yaml.dump(scoot_info, f)
                        break
                    config_idx += 1

            self.save_dir_path_map['metrics'] = save_dir_path

        else:
            raise NotImplementedError(f"Not supported control method: {self.simulator_info['control_method']}")
    
        return
    
    def _validation(self):
        return
