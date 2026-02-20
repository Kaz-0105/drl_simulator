import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import re
import copy

from libs.figure_config import initFigureConfig

# reflect figure configuration
initFigureConfig()

# get config_yaml
config_file_path = root_dir_path / 'misc' / 'analysis' / 'performance_metric_time_series_plots' / 'road_level' / 'config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set data and performance_metrics directory paths
data_dir_path = root_dir_path / 'data'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'
save_base_dir_path = data_dir_path / 'analysis' / 'performance_metric_time_series_plots' / 'road_level'

# make time_series_df_map
time_series_df_map = {}
layout_dir_path = performance_metrics_dir_path / config_yaml['target']['layout']
inflow_dir_path = layout_dir_path / config_yaml['target']['inflow']
for simulator_dir_path in inflow_dir_path.rglob('simulator_*'):
    with open(simulator_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
        simulator_config = yaml.safe_load(f)
    
    # if seed is fixed in the plot config, check it
    if config_yaml['target']['seed']['fix_flg'] and simulator_config['seed'] != config_yaml['target']['seed']['fix_value']:
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
        if not config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
            continue 
        del method_config['phases']

        if method_config != config_yaml['mpc']:
            continue

        found_flg = False
        for intersection_dir_path in method_dir_path.glob('intersection_*'):
            intersection_id = int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1))
            if intersection_id == config_yaml['target']['intersection_id']:
                found_flg = True
                break
        
        if not found_flg:
            continue
        
        time_series_df = None
        for road_dir_path in intersection_dir_path.glob('road_*'):
            road_id = int(re.match(rf"road_(\d+)", road_dir_path.name).group(1))

            with open(road_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                tmp_time_series_df = pd.read_csv(f) 

            tmp_time_series_df['road'] = road_id

            if time_series_df is None:
                time_series_df = tmp_time_series_df.copy()
            else:
                time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True)

        time_series_df_map[f"{num_phases}_phase_mpc"] = time_series_df

    # regarding scoot
    if not config_yaml['target']['control_method']['scoot']:
        continue

    scoot_dir_path = simulator_dir_path / 'scoot'
    if not scoot_dir_path.exists():
        continue

    for method_dir_path in scoot_dir_path.glob('config_*'):
        with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
            method_config = yaml.safe_load(f)

        if method_config != config_yaml['scoot']:
            continue

        found_flg = False
        for intersection_dir_path in method_dir_path.glob('intersection_*'):
            intersection_id = int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1))
            if intersection_id == config_yaml['target']['intersection_id']:
                found_flg = True
                break
        
        if not found_flg:
            continue
        
        time_series_df = None
        for road_dir_path in intersection_dir_path.glob('road_*'):
            road_id = int(re.match(rf"road_(\d+)", road_dir_path.name).group(1))

            with open(road_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                tmp_time_series_df = pd.read_csv(f) 

            tmp_time_series_df['road'] = road_id

            if time_series_df is None:
                time_series_df = tmp_time_series_df.copy()
            else:
                time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True)

        time_series_df_map['scoot'] = time_series_df

max_queue_value = 0
for method, time_series_df in time_series_df_map.items():
        tmp_time_series_df = pd.DataFrame(tmp_time_series_data_map)

        road_id_list = list(set([int(re.match(rf"road_(\d+)_*", column).group(1)) for column in tmp_time_series_df.columns if column != 'time']))
        
        for road_id in road_id_list:
            tmp_time_series_df[f"road_{road_id}_queue_max"] = tmp_time_series_df[[column for column in tmp_time_series_df.columns if re.match(rf"road_{road_id}_queue_*", column)]].sum(axis=1)
            max_queue_value = max(max_queue_value, tmp_time_series_df[f"road_{road_id}_queue_max"].max())
        
        tmp_time_series_df = tmp_time_series_df.melt(
            id_vars='time',
            value_vars=[column for column in tmp_time_series_df.columns if column != 'time'],
            var_name='group',
            value_name='value'
        )
        time_series_df_map[method] = tmp_time_series_df

save_dir_path = save_base_dir_path / config_yaml['target']['layout'] / f"intersection_{config_yaml['target']['intersection_id']}"
save_dir_path.mkdir(parents=True, exist_ok=True)

# for each road
for method, tmp_time_series_df in time_series_df_map.items():
    fig, ax = plt.subplots()
    target_time_series_df = tmp_time_series_df[tmp_time_series_df['group'].str.contains('queue_max')]
    sns.lineplot(
        data=target_time_series_df,
        x='time',
        y='value',
        hue='group',
        ax=ax
    )
    ax.set_title(config_yaml['figure']['title'])
    ax.set_xlabel(config_yaml['figure']['x_axis']['label'])
    ax.set_ylabel(config_yaml['figure']['y_axis']['label'])
    ax.set_xlim(left=0, right=target_time_series_df['time'].max())
    ax.set_ylim(bottom=0, top=max_queue_value * 1.1)
    
    handles, labels = ax.get_legend_handles_labels()
    for id, label in enumerate(copy.deepcopy(labels)):
        road_id = int(re.match(rf"road_(\d+)_queue_max", label).group(1))
        labels[id] = config_yaml['figure']['legend'][road_id]
    ax.legend(handles, labels, title='')

    fig.tight_layout()
    fig.savefig(save_dir_path / f"{method}_queue_time_series.png")

# for each lane
fig, ax = plt.subplots()
print('Finished!')



