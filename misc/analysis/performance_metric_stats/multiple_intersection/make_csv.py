import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import pandas as pd
import yaml
import re

from libs.figure_config import init_figure_config

# reflect figure configuration
init_figure_config()

# get config_yaml
config_file_path = root_dir_path / 'misc' / 'analysis' / 'performance_metric_stats' / 'multiple_intersection' / 'config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set data and performance_metrics directory paths
data_dir_path = root_dir_path / 'data'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'
save_base_dir_path = data_dir_path / 'analysis' / 'performance_metric_stats'

# make simulator_dir_path_map
simulator_dir_path_map = {}
for layout, route_selection in config_yaml['target']['layout'].items():
    layout_dir_path = performance_metrics_dir_path / layout
    for simulator_dir_path in layout_dir_path.rglob('simulator_*'):
        config_file_path = simulator_dir_path / 'config.yaml'
        if not config_file_path.exists():
            continue
        with open(config_file_path, 'r', encoding='utf-8') as f:
            simulator_config = yaml.safe_load(f)
        
        # check the simulator config matches the target config
        if config_yaml['target']['seed']['fix_flg'] and simulator_config['seed'] != config_yaml['target']['seed']['fix_value']:
            continue
        del simulator_config['seed'] 

        if simulator_config != config_yaml['simulator']:
            continue
        
        inflow = simulator_dir_path.parent.name
        simulator_dir_path_map[(inflow, route_selection)] = simulator_dir_path

# initialize performance_metric_map_list
performance_metric_map_list = []

# regarding mpc
if any(config_yaml['target']['control_method']['mpc'].values()):
    for (inflow, route_selection), simulator_dir_path in simulator_dir_path_map.items():
        mpc_dir_path = simulator_dir_path / 'mpc'
        if not mpc_dir_path.exists():
            continue
        
        for method_dir_path in mpc_dir_path.glob('config_*'):
            config_file_path = method_dir_path / 'config.yaml'
            if not config_file_path.exists():
                continue
            with open(config_file_path, 'r', encoding='utf-8') as f:
                method_config = yaml.safe_load(f)

            # check if the method_config matches the target config
            num_phases = method_config['phases']['4-road']
            if not config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
                continue 
            del method_config['phases']

            signal_change_weight = method_config['objective_function']['signal_change']['weight']
            if config_yaml['target']['signal_change_weight'][f"{num_phases}-phase"] != signal_change_weight:
                continue
            del method_config['objective_function']['signal_change']['weight']
            
            if method_config != config_yaml['mpc']:
                continue
            
            time_series_df = None
            for intersection_dir_path in method_dir_path.glob('intersection_*'):
                performance_metrics_file_path = intersection_dir_path / 'performance_metrics.csv'
                if not performance_metrics_file_path.exists():
                    continue
                with open(performance_metrics_file_path, 'r', encoding='utf-8') as f:
                    tmp_time_series_df = pd.read_csv(f)
                
                tmp_time_series_df = tmp_time_series_df[['time'] + config_yaml['target']['performance_metrics']].copy()
                tmp_time_series_df['intersection'] = int(re.match(r"intersection_(\d+)", intersection_dir_path.name).group(1))

                if time_series_df is None:
                    time_series_df = tmp_time_series_df.copy()  
                else:
                    time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True) 

            if time_series_df is None:
                continue

            invalid_ids = time_series_df[config_yaml['target']['performance_metrics']].isna().any(axis=1)
            invalid_times = time_series_df.loc[invalid_ids, 'time'].unique()
            time_series_df = time_series_df[~time_series_df['time'].isin(invalid_times)].reset_index(drop=True)
            time_series_df = time_series_df.groupby('time')[config_yaml['target']['performance_metrics']].mean()
            time_series_df = time_series_df.dropna(subset=config_yaml['target']['performance_metrics'])
            time_series_df = time_series_df.reset_index()

            performance_metric_map = {
                'id': len(performance_metric_map_list) + 1,
                'method': f"{num_phases}-phase MPC", 
                'inflow': inflow,
                'route_selection': route_selection,
            }
            for performance_metric in config_yaml['target']['performance_metrics']:
                performance_metric_map[performance_metric] = time_series_df[performance_metric].mean()

            performance_metric_map_list.append(performance_metric_map)
            
# regarding scoot
if config_yaml['target']['control_method']['scoot']:
    for (inflow, route_selection), simulator_dir_path in simulator_dir_path_map.items():
        scoot_dir_path = simulator_dir_path / 'scoot'
        if not scoot_dir_path.exists():
            continue

        for method_dir_path in scoot_dir_path.glob('config_*'):
            # check if the method_config matches the target config
            config_file_path = method_dir_path / 'config.yaml'
            if not config_file_path.exists():
                continue
            with open(config_file_path, 'r', encoding='utf-8') as f:
                method_config = yaml.safe_load(f)
            
            if method_config != config_yaml['scoot']:
                continue    

            time_series_df = None
            for intersection_dir_path in method_dir_path.glob('intersection_*'):
                performance_metrics_file_path = intersection_dir_path / 'performance_metrics.csv'
                if not performance_metrics_file_path.exists():
                    continue
                with open(performance_metrics_file_path, 'r', encoding='utf-8') as f:
                    tmp_time_series_df = pd.read_csv(f)
                
                tmp_time_series_df = tmp_time_series_df[['time'] + config_yaml['target']['performance_metrics']].copy()
                tmp_time_series_df['intersection'] = int(re.match(r"intersection_(\d+)", intersection_dir_path.name).group(1))

                if time_series_df is None:
                    time_series_df = tmp_time_series_df.copy()  
                else:
                    time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True)
            
            if time_series_df is None:
                continue

            invalid_ids = time_series_df[config_yaml['target']['performance_metrics']].isna().any(axis=1)
            invalid_times = time_series_df.loc[invalid_ids, 'time'].unique()
            time_series_df = time_series_df[~time_series_df['time'].isin(invalid_times)].reset_index(drop=True)
            time_series_df = time_series_df.groupby('time')[config_yaml['target']['performance_metrics']].mean()
            time_series_df = time_series_df.dropna(subset=config_yaml['target']['performance_metrics'])
            time_series_df = time_series_df.reset_index()

            performance_metric_map = {
                'id': len(performance_metric_map_list) + 1,
                'method': 'SCOOT',
                'inflow': inflow,
                'route_selection': route_selection,
            }
            for performance_metric in config_yaml['target']['performance_metrics']:
                performance_metric_map[performance_metric] = time_series_df[performance_metric].mean()
            
            performance_metric_map_list.append(performance_metric_map)

performance_metric_df = pd.DataFrame(performance_metric_map_list)

# make save_dir
save_dir_path = data_dir_path / 'analysis' / 'performance_metric_stats' / 'multiple_intersection'
save_dir_path.mkdir(parents=True, exist_ok=True)

# make performance_metric_stat_df
performance_metric_stat_map_list = []
for method in performance_metric_df['method'].unique().tolist():
    tmp_performance_metric_df = performance_metric_df[performance_metric_df['method'] == method]
    if tmp_performance_metric_df.empty:
        continue
    for performance_metric in config_yaml['target']['performance_metrics']:
        performance_metric_stat_map_list.append({
            'id': len(performance_metric_stat_map_list) + 1,
            'performance_metric': performance_metric,
            'method': method,
            'mean': tmp_performance_metric_df[performance_metric].mean(),
            'worst': tmp_performance_metric_df[performance_metric].min() if performance_metric == 'speed_avg' else tmp_performance_metric_df[performance_metric].max(),
            'std': tmp_performance_metric_df[performance_metric].std(),
        })
performance_metric_stat_df = pd.DataFrame(performance_metric_stat_map_list)

# add improve_rate column
improve_rate_list = [0] * len(performance_metric_stat_df)
scoot_stat_df = performance_metric_stat_df[performance_metric_stat_df['method'] == 'SCOOT'].copy()
for _, scoot_stat_row in scoot_stat_df.iterrows():
    performance_metric = scoot_stat_row['performance_metric']
    reference_value = scoot_stat_row['mean']
    for method in ['4-phase MPC', '8-phase MPC', '17-phase MPC']:
        target_stat_row = performance_metric_stat_df[
            (performance_metric_stat_df['method'] == method) &
            (performance_metric_stat_df['performance_metric'] == performance_metric)
        ]
        if target_stat_row.empty:
            continue
        target_id = target_stat_row['id'].values[0]
        target_value = target_stat_row['mean'].values[0]
        improve_rate_list[target_id - 1] = (target_value - reference_value) / reference_value * 100
performance_metric_stat_df['improve_rate'] = improve_rate_list

# add num_best column
best_count_map = {
    performance_metric: {method: 0 for method in performance_metric_stat_df['method'].unique().tolist()}
    for performance_metric in config_yaml['target']['performance_metrics']
}
for inflow in performance_metric_df['inflow'].unique().tolist():
    for route_selection in performance_metric_df['route_selection'].unique().tolist():
        tmp_performance_metric_df = performance_metric_df[
            (performance_metric_df['inflow'] == inflow) &
            (performance_metric_df['route_selection'] == route_selection)   
        ]
        if tmp_performance_metric_df.empty:
            continue
        for performance_metric in config_yaml['target']['performance_metrics']:
            if performance_metric == 'speed_avg':
                best_id = tmp_performance_metric_df[performance_metric].idxmax()
            elif performance_metric in ['queue_avg', 'queue_max', 'delay_avg_1', 'delay_max']:
                best_id = tmp_performance_metric_df[performance_metric].idxmin()
            else:
                raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
            best_method = tmp_performance_metric_df.loc[best_id, 'method']
            best_count_map[performance_metric][best_method] += 1

num_best_list = [0] * len(performance_metric_stat_df)
for id, stat_row in performance_metric_stat_df.iterrows():
    performance_metric = stat_row['performance_metric']
    method = stat_row['method']
    num_best_list[id] = best_count_map[performance_metric][method]

performance_metric_stat_df['num_best'] = num_best_list

performance_metric_stat_df[['mean', 'worst', 'std', 'improve_rate']] = performance_metric_stat_df[['mean', 'worst', 'std', 'improve_rate']].round(2)
performance_metric_stat_df.to_csv(save_dir_path / 'performance_metric_stat.csv', index=False, encoding='utf-8')

print('Finished!')