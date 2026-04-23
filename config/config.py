from libs.common import Common 

import yaml
import pandas as pd
import re
import copy
import random

class Config(Common):
    def __init__(self, vissim):
        super().__init__()

        self.vissim = vissim

        self._initProps()

        # initialize save_dir_path_map 
        self._initSaveDirPathMap()
        
        # initialize num_features_map if control method is drl  
        if self.simulator_info['control_method'] == 'drl':
            self._initNumFeaturesMap()
        return
    
    def _initProps(self):
        # set root_dir_path
        self.root_dir_path = self.vissim.get('root_dir_path')

        # set config.yaml information
        with open(self.root_dir_path / 'config' / 'config.yaml', 'r', encoding='utf-8') as file:
            config_yaml = yaml.safe_load(file)

            self.simulator_info = config_yaml['simulator']

            if self.simulator_info['control_method'] == 'drl':
                self.drl_info = config_yaml['drl']
            elif self.simulator_info['control_method'] == 'mpc':
                self.mpc_info = config_yaml['mpc']
            elif self.simulator_info['control_method'] == 'scoot':
                self.scoot_info = config_yaml['scoot']
            else:
                raise NotImplementedError(f"Not supported control method: {self.simulator_info['control_method']}")

            self.save_info = config_yaml['save']

        # set layout dir path
        self.layout_dir_path = self.root_dir_path / 'layout' / self.simulator_info['layout_name']

        # set network information
        with open(self.layout_dir_path / 'roads.csv', 'r', encoding='utf-8', newline='') as f:
            self.roads_df = pd.read_csv(f, index_col=False)
        
        with open(self.layout_dir_path / 'road_link_tags.csv', 'r', encoding='utf-8', newline='') as f:
            self.road_link_tags_df = pd.read_csv(f, index_col=False)
        
        with open(self.layout_dir_path / 'intersections.csv', 'r', encoding='utf-8', newline='') as f:
            self.intersections_df = pd.read_csv(f, index_col=False)

        with open(self.layout_dir_path / 'intersection_road_tags.csv', 'r', encoding='utf-8', newline='') as f:
            self.intersection_road_tags_df = pd.read_csv(f, index_col=False)

        with open(self.layout_dir_path / 'intersection_turn_ratio_tags.csv', 'r', encoding='utf-8', newline='') as f:
            self.intersection_turn_ratio_tags_df = pd.read_csv(f, index_col=False)
        
        if self.simulator_info['control_method'] == 'drl':
            with open(self.layout_dir_path / 'epsilons.csv', 'r', encoding='utf-8', newline='') as f:
                self.epsilons_df = pd.read_csv(f, index_col=False)

        link_input_tags_dir_path = self.layout_dir_path / 'link_input_tags'
        if not link_input_tags_dir_path.exists():
            raise FileNotFoundError(f"Not found: {link_input_tags_dir_path}")
        
        self.link_input_tags_df_map = {}
        for link_input_tags_file_path in link_input_tags_dir_path.glob('*.csv'):
            with open(link_input_tags_file_path, 'r', encoding='utf-8', newline='') as f:
                self.link_input_tags_df_map[link_input_tags_file_path.stem] = pd.read_csv(f, index_col=False)
        
        # set turn_ratio_df_map
        self.turn_ratio_df_map = {}
        turn_ratio_dir_path = self.root_dir_path / 'config' / 'meta' / 'turn_ratio'
        for tmp_file_path in turn_ratio_dir_path.glob('turn_ratio_*.csv'):
            num_roads = int(re.match(rf"turn_ratio_(\d+).csv", tmp_file_path.name)[1])
            self.turn_ratio_df_map[num_roads] = pd.read_csv(tmp_file_path, index_col=False)
        
        phases_dir_path = self.root_dir_path / 'config' / 'meta' / 'phases'

        # set phases_df_map
        self.phases_df_map = {}
        for tmp_file_path in phases_dir_path.glob('phases_*.csv'):
            num_roads = int(re.match(rf"phases_(\d+).csv", tmp_file_path.name)[1])
            self.phases_df_map[num_roads] = pd.read_csv(tmp_file_path, index_col=False)

        # set symmetry_phases_df_map
        self.symmetry_phases_df_map = {}
        for tmp_file_path in phases_dir_path.glob('symmetry_phases_*.csv'):
            num_roads = int(re.match(rf"symmetry_phases_(\d+).csv", tmp_file_path.name)[1])
            self.symmetry_phases_df_map[num_roads] = pd.read_csv(tmp_file_path, index_col=False)
        return
        
    def _initSaveDirPathMap(self):
        self.save_dir_path_map = {}

        if self.simulator_info['control_method'] == 'drl' and self.drl_info['simulation_type'] == 'train':
            self.save_dir_path_map['metrics'] = None
            return

        common_save_dir_path = self.root_dir_path / 'data' / 'performance_metrics'
        common_save_dir_path /= self.layout_dir_path.name
        common_save_dir_path /= self.simulator_info['inflow_name']
        common_save_dir_path.mkdir(parents=True, exist_ok=True)

        # if the setting of simulator_info is same as the existing one, use the existing save_dir_path
        simulator_info = copy.deepcopy(self.simulator_info)
        simulator_info = {key: simulator_info[key] for key in ['num_red_steps', 'simulation_time','time_step']}
        
        simulation_dir_path = None
        for tmp_dir_path in common_save_dir_path.glob('simulator_*'):
            config_file_path = tmp_dir_path / 'config.yaml'
            if not config_file_path.exists():
                continue

            with open(config_file_path, 'r', encoding='utf-8') as f:
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
            for _, intersection_row in self.intersections_df.iterrows():
                num_roads_set.add(intersection_row['num_roads'])
            
            for intersection_type in copy.deepcopy(mpc_info['phases']).keys():
                num_roads = int(re.match(r"(\d+)-road", intersection_type)[1])
                if num_roads not in num_roads_set:
                    del mpc_info['phases'][intersection_type]

            save_dir_path = None
            for tmp_dir_path in control_method_dir_path.glob('config_*'):
                config_file_path = tmp_dir_path / 'config.yaml'
                if not config_file_path.exists():
                    continue
                
                with open(config_file_path, 'r', encoding='utf-8') as f:
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

                with open(config_file_path, 'r', encoding='utf-8') as f:
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

        elif self.simulator_info['control_method'] == 'drl':
            drl_info = copy.deepcopy(self.drl_info)

            save_dir_path = None
            for tmp_dir_path in control_method_dir_path.glob('config_*'):
                config_file_path = tmp_dir_path / 'config.yaml'
                if not config_file_path.exists():
                    continue

                with open(config_file_path, 'r', encoding='utf-8') as f:
                    config_yaml = yaml.safe_load(f)

                if config_yaml == drl_info:
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
                            yaml.dump(drl_info, f)
                        break
                    config_idx += 1
            
            self.save_dir_path_map['metrics'] = save_dir_path

        else:
            raise NotImplementedError(f"Not supported control method: {self.simulator_info['control_method']}")
    
        return
    
    def _initNumFeaturesMap(self):
        self.num_features_map = {}

        # vehicle features
        tmp_num_features_map = {num_roads: 0 for num_roads in [3, 4, 5]}
        for feature_name, feature_flg in self.drl_info['state']['vehicle'].items():
            if feature_name in ['number']:
                continue

            if not feature_flg:
                continue

            if feature_name in ['position', 'speed']:
                tmp_num_features_map = {num_roads: tmp_num_features_map[num_roads] + 1 for num_roads in tmp_num_features_map.keys()}
            elif feature_name in ['route']:
                tmp_num_features_map = {num_roads: tmp_num_features_map[num_roads] + num_roads for num_roads in tmp_num_features_map.keys()}
            else:
                raise NotImplementedError(f"Not supported feature: {feature_name}")
        tmp_num_features_map = {num_roads: tmp_num_features_map[num_roads] + 1 for num_roads in tmp_num_features_map.keys()} # for existence of vehicle
        self.num_features_map['vehicle'] = tmp_num_features_map

        # lane features
        tmp_num_features = 0
        for feature_name, feature_flg in self.drl_info['state']['lane'].items():
            if not feature_flg:
                continue

            if feature_name in ['length', 'num_vehicles']:
                tmp_num_features += 1
            elif feature_name in ['type']:
                tmp_num_features += 3
            else:
                raise NotImplementedError(f"Not supported feature: {feature_name}")
        self.num_features_map['lane'] = tmp_num_features

        # road features
        tmp_num_features_map = {num_roads: 0 for num_roads in [3, 4, 5]}
        for feature_name, feature_flg in self.drl_info['state']['road'].items():
            if not feature_flg:
                continue

            if feature_name in ['queue', 'delay']:
                tmp_num_features_map = {num_roads: tmp_num_features_map[num_roads] + 1 for num_roads in tmp_num_features_map.keys()}
            elif feature_name in ['route']:
                tmp_num_features_map = {num_roads: tmp_num_features_map[num_roads] + (num_roads - 1) for num_roads in tmp_num_features_map.keys()}
            else:
                raise NotImplementedError(f"Not supported feature: {feature_name}")
        self.num_features_map['road'] = tmp_num_features_map
    
        # intersection features
        self.num_features_map['intersection'] = {num_roads: len(phases) for num_roads, phases in self.phases_df_map.items()}
        return
