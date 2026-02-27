import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import re

from libs.figure_config import init_figure_config

# load config.yaml
with open(root_dir_path / 'misc' / 'analysis' / 'performance_trajectory_plots' / 'multiple_intersection' / 'config.yaml', 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# load figure configuration
init_figure_config()


data_dir_path = root_dir_path / 'data'
analysis_dir_path = data_dir_path / 'analysis'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'

layout_dir_path = performance_metrics_dir_path / config_yaml['target']['layout']
if not layout_dir_path.exists():
    raise FileNotFoundError(f"Directory not found: {layout_dir_path}")

inflow_dir_path = layout_dir_path / config_yaml['target']['inflow']
if not inflow_dir_path.exists():
    raise FileNotFoundError(f"Directory not found: {inflow_dir_path}")

time_series_df_map = {}
for simulator_dir_path in inflow_dir_path.glob('simulator_*'):
    simulator_config_file_path = simulator_dir_path / 'config.yaml'
    with open(simulator_config_file_path, 'r', encoding='utf-8') as f:
        simulator_config_yaml = yaml.safe_load(f)
    
    if simulator_config_yaml != config_yaml['simulator']:
        continue
    
    # load mpc performance metrics
    mpc_dir_path = simulator_dir_path / 'mpc'
    for config_dir_path in mpc_dir_path.glob('config_*'):
        mpc_config_file_path = config_dir_path / 'config.yaml'
        with open(mpc_config_file_path, 'r', encoding='utf-8') as f:
            mpc_config_yaml = yaml.safe_load(f)
        
        # check num_phases
        num_phases = mpc_config_yaml['phases']['4-road'] # TODO: currently only support 4-road intersection
        if num_phases not in [4, 8, 17]:
            raise ValueError(f"Not supported num_phases: {num_phases}")
        if not config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
            continue 
        del mpc_config_yaml['phases']
        
        # check if the configuration is the same as the target configuration for MPC
        config_yaml['mpc']['objective_function']['signal_change']['weight'] = config_yaml['target']['signal_change_weight'][f"{num_phases}-phase"]
        if mpc_config_yaml != config_yaml['mpc']:
            continue
        
        time_series_df = None
        for intersection_dir_path in config_dir_path.glob('intersection_*'):
            with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                tmp_time_series_df = pd.read_csv(f)
            
            tmp_time_series_df = tmp_time_series_df[['time'] + list(config_yaml['target']['performance_metrics'].values())].copy()
            tmp_time_series_df['intersection'] = int(re.match(r"intersection_(\d+)", intersection_dir_path.name).group(1))
            tmp_time_series_df = tmp_time_series_df.dropna(subset=config_yaml['target']['performance_metrics'].values()).reset_index(drop=True)

            if time_series_df is None:
                time_series_df = tmp_time_series_df.copy()    
            else:
                time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True)
        
        time_series_df = time_series_df.groupby('time')[list(config_yaml['target']['performance_metrics'].values())].mean().reset_index()
        time_series_df = time_series_df.rolling(window=5, min_periods=1).mean()
        time_series_df = time_series_df.dropna(subset=list(config_yaml['target']['performance_metrics'].values())).reset_index(drop=True)
        print('test')

            

            


        print('test')

print('Finish!')



