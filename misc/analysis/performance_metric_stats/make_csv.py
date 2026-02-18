import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import re

from libs.figure_config import initFigureConfig

# reflect figure configuration
initFigureConfig()

# get config_yaml
config_file_path = root_dir_path / 'misc' / 'analysis' / 'performance_metric_stats' / 'config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set data and performance_metrics directory paths
data_dir_path = root_dir_path / 'data'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'
save_base_dir_path = data_dir_path / 'analysis' / 'performance_metric_stats'

# initialize performance_df
performance_metric_map_list = []
data_dir_path = root_dir_path / 'data'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'
layout_dir_path = performance_metrics_dir_path / config_yaml['figure']['layout']
for simulator_dir_path in layout_dir_path.rglob('simulator_*'):
    with open(simulator_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
        simulator_config = yaml.safe_load(f)
    
    # if seed is fixed in the plot config, check it
    if config_yaml['figure']['seed']['fix_flg'] and simulator_config['seed'] != config_yaml['figure']['seed']['fix_value']:
        continue
    del simulator_config['seed'] 

    # check simulator config matches the plot config
    if simulator_config != config_yaml['simulator']:
        continue

    # get inflow file name
    inflow = simulator_dir_path.parent.name
    
    # regarding mpc
    mpc_dir_path = simulator_dir_path / 'mpc'
    if not mpc_dir_path.exists():
        continue

    for method_dir_path in mpc_dir_path.glob('config_*'):
        with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
            method_config = yaml.safe_load(f)

        # check num_phases
        num_phases = method_config['phases']['4-road'] # TODO: currently only support 4-road intersection
        if num_phases not in [4, 8, 17]:
            raise ValueError(f"Not supported num_phases: {num_phases}")
        if not config_yaml['figure']['control_method']['mpc'][f"{num_phases}-phase"]:
            continue 
        del method_config['phases']

        if method_config != config_yaml['mpc']:
            continue
        
        for intersection_dir_path in method_dir_path.glob('intersection_*'):
            # make performance_metric_map
            performance_metric_map = {
                'id': len(performance_metric_map_list) + 1,
                'method': f"{num_phases}-phase MPC",
                'inflow': inflow,
                'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
            }
            for performance_metric in config_yaml['figure']['performance_metrics']:
                with open(intersection_dir_path / f"{performance_metric}.csv", 'r', encoding='utf-8'):
                    time_series_df = pd.read_csv(intersection_dir_path / f"{performance_metric}.csv")
                    
                    if performance_metric in ['average_queue', 'max_queue']:
                        performance_value = time_series_df['queue_length'].mean()
                    elif performance_metric in ['average_delay', 'max_delay']:
                        performance_value = time_series_df['delay'].mean()
                    elif performance_metric == 'speed':
                        performance_value = time_series_df['value'].mean()
                    elif performance_metric == 'phases':
                        time_series_df['change'] = time_series_df['phase'].ne(time_series_df['phase'].shift(1))
                        time_series_df.loc[0, 'change'] = False
                        performance_value = time_series_df['change'].sum()
                    else:
                        raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                
                performance_metric_map[performance_metric] = performance_value
            
            performance_metric_map_list.append(performance_metric_map)

    # regarding scoot
    if not config_yaml['figure']['control_method']['scoot']:
        continue

    scoot_dir_path = simulator_dir_path / 'scoot'
    if not scoot_dir_path.exists():
        continue

    for method_dir_path in scoot_dir_path.glob('config_*'):
        with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
            method_config = yaml.safe_load(f)

        if method_config != config_yaml['scoot']:
            continue
        
        for intersection_dir_path in method_dir_path.glob('intersection_*'):
            # make performance_metric_map
            performance_metric_map = {
                'id': len(performance_metric_map_list) + 1,
                'method': 'SCOOT',
                'inflow': inflow,
                'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
            }
            for performance_metric in config_yaml['figure']['performance_metrics']:
                with open(intersection_dir_path / f"{performance_metric}.csv", 'r', encoding='utf-8'):
                    time_series_df = pd.read_csv(intersection_dir_path / f"{performance_metric}.csv")
                    
                    if performance_metric in ['average_queue', 'max_queue']:
                        performance_value = time_series_df['queue_length'].mean()
                    elif performance_metric in ['average_delay', 'max_delay']:
                        performance_value = time_series_df['delay'].mean()
                    elif performance_metric == 'speed':
                        performance_value = time_series_df['value'].mean()
                    elif performance_metric == 'phases':
                        time_series_df['change'] = time_series_df['phase'].ne(time_series_df['phase'].shift(1))
                        time_series_df.loc[0, 'change'] = False
                        performance_value = time_series_df['change'].sum()
                    else:
                        raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                
                performance_metric_map[performance_metric] = performance_value
            
            performance_metric_map_list.append(performance_metric_map)
            
performance_metric_df = pd.DataFrame(
    performance_metric_map_list, 
    columns=['id', 'method', 'inflow', 'intersection'] + config_yaml['figure']['performance_metrics']
)

# make save_dir
save_dir_path = data_dir_path / 'analysis' / 'performance_metric_stats' / config_yaml['figure']['layout']
save_dir_path.mkdir(parents=True, exist_ok=True)

# make performance_metric_stat_df
performance_metric_stat_map_list = []
for method in performance_metric_df['method'].unique().tolist():
    tmp_performance_metric_df = performance_metric_df[performance_metric_df['method'] == method]
    if tmp_performance_metric_df.empty:
        continue
    for performance_metric in config_yaml['figure']['performance_metrics']:
        performance_metric_stat_map_list.append({
            'id': len(performance_metric_stat_map_list) + 1,
            'performance_metric': performance_metric,
            'method': method,
            'mean': tmp_performance_metric_df[performance_metric].mean(),
            'worst': tmp_performance_metric_df[performance_metric].min() if performance_metric == 'speed' else tmp_performance_metric_df[performance_metric].max(),
            'std': tmp_performance_metric_df[performance_metric].std(),
        })
performance_metric_stat_df = pd.DataFrame(
    performance_metric_stat_map_list, 
    columns=['id', 'performance_metric', 'method', 'mean', 'worst', 'std']
)

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
    for performance_metric in config_yaml['figure']['performance_metrics']
}
for inflow in performance_metric_df['inflow'].unique().tolist():
    for intersection_id in performance_metric_df['intersection'].unique().tolist():
        tmp_performance_metric_df = performance_metric_df[
            (performance_metric_df['intersection'] == intersection_id) &
            (performance_metric_df['inflow'] == inflow)
        ]
        if tmp_performance_metric_df.empty:
            continue
        for performance_metric in config_yaml['figure']['performance_metrics']:
            if performance_metric == 'speed':
                best_id = tmp_performance_metric_df[performance_metric].idxmax()
            elif performance_metric in ['average_queue', 'max_queue', 'average_delay', 'max_delay']:
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