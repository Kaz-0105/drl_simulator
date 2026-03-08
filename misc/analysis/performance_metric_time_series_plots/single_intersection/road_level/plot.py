import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import re

from libs.figure_config import init_figure_config

# reflect figure configuration
init_figure_config()

# get config_yaml
config_file_path = root_dir_path / 'misc' / 'analysis' / 'performance_metric_time_series_plots' / 'single_intersection' / 'road_level' / 'config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set data and performance_metrics directory paths
data_dir_path = root_dir_path / 'data'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'
save_base_dir_path = data_dir_path / 'analysis' / 'performance_metric_time_series_plots' / 'single_intersection' / 'road_level'

# make time_series_df_map
time_series_df_map = {}
layout_dir_path = performance_metrics_dir_path / config_yaml['target']['layout']
inflow_dir_path = layout_dir_path / config_yaml['target']['inflow']
for simulator_dir_path in inflow_dir_path.rglob('simulator_*'):
    with open(simulator_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
        simulator_config = yaml.safe_load(f)

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

        config_yaml['mpc']['objective_function']['signal_change']['weight'] = config_yaml['target']['signal_change_weight'][f"{num_phases}-phase"]

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
            
            # add road column and queue_max column
            tmp_time_series_df['road'] = road_id
            exist_queue_columns = [column for column in ['queue_main', 'queue_right', 'queue_left'] if column in tmp_time_series_df.columns]
            tmp_time_series_df['queue_max'] = tmp_time_series_df[exist_queue_columns].max(axis=1)

            # push to time_series_df
            target_time_series_df = tmp_time_series_df[['time', 'road', 'queue_max'] + exist_queue_columns]
            if time_series_df is None:
                time_series_df = target_time_series_df.copy()
            else:
                time_series_df = pd.concat([time_series_df, target_time_series_df], ignore_index=True)

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

            # add road column and queue_max column
            tmp_time_series_df['road'] = road_id
            exist_queue_columns = [column for column in ['queue_main', 'queue_right', 'queue_left'] if column in tmp_time_series_df.columns]
            tmp_time_series_df['queue_max'] = tmp_time_series_df[exist_queue_columns].max(axis=1)
            
            # push to time_series_df
            target_time_series_df = tmp_time_series_df[['time', 'road', 'queue_max']]
            if time_series_df is None:
                time_series_df = target_time_series_df.copy()
            else:
                time_series_df = pd.concat([time_series_df, target_time_series_df], ignore_index=True)

        time_series_df_map['scoot'] = time_series_df

save_dir_path = save_base_dir_path / config_yaml['target']['layout'] / f"intersection_{config_yaml['target']['intersection_id']}"
save_dir_path.mkdir(parents=True, exist_ok=True)

# calculate max queue length for setting y axis limit
max_queue_length = 0
for method, tmp_time_series_df in time_series_df_map.items():
    max_queue_length = max(max_queue_length, tmp_time_series_df['queue_max'].max())

# for each road
for method, tmp_time_series_df in time_series_df_map.items():
    # add line_style column
    tmp_time_series_df['line_style'] = ''
    for road_id in tmp_time_series_df['road'].unique():
        tmp_time_series_df.loc[tmp_time_series_df['road'] == road_id, 'line_style'] = config_yaml['figure']['line_style'][road_id]
    fig, ax = plt.subplots()
    
    sns.lineplot(
        ax=ax,
        data=tmp_time_series_df,
        x='time',
        y='queue_max',
        hue='road',
        style='line_style',
        dashes={
            'solid': (None, None),
            'dashed': (2, 1),
        },
        palette=config_yaml['figure']['palette'],
    )

    if method == 'scoot':
        figure_title = config_yaml['figure']['title']['scoot']
    elif method.endswith('_phase_mpc'):
        num_phases = int(re.match(r"(\d+)_phase_mpc", method).group(1))
        figure_title = config_yaml['figure']['title']['mpc'][f"{num_phases}-phase"]
    else: 
        raise NotImplementedError(f"Not supported method: {method}")
    ax.set_title(figure_title)
    ax.set_xlabel(config_yaml['figure']['x_axis']['label'])
    ax.set_ylabel(config_yaml['figure']['y_axis']['label'])
    ax.set_xlim(left=0, right=tmp_time_series_df['time'].max())
    ax.set_ylim(bottom=0, top= max_queue_length* 1.1)
    
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    for handle, label in zip(handles, labels):
        match_obj = re.match(r"(\d+)", label)
        if match_obj:
            new_handles.append(handle)
            new_labels.append(config_yaml['figure']['legend'][int(match_obj.group(1))])
    ax.legend(new_handles, new_labels, title='', ncol=len(new_labels) / 2)

    fig.tight_layout()
    fig.savefig(save_dir_path / f"{method}_queue_time_series.png")

# for each lane
fig, ax = plt.subplots()
print('Finished!')



