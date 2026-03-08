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
with open(root_dir_path / 'misc' / 'analysis' / 'performance_metric_time_series_plots' / 'multiple_intersection' / 'config.yaml', 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# load figure configuration
init_figure_config()

# set directory paths
data_dir_path = root_dir_path / 'data'
analysis_dir_path = data_dir_path / 'analysis'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'

# set target_dir_path
target_dir_path = performance_metrics_dir_path / config_yaml['target']['layout'] / config_yaml['target']['inflow']
if not target_dir_path.exists():
    raise FileNotFoundError(f'{target_dir_path} does not exist.')

found_flg = False
for simulator_dir_path in target_dir_path.glob('simulator_*'):
    # check the configuration of simulator matches the target one
    simulator_config_file_path = simulator_dir_path / 'config.yaml'
    if not simulator_config_file_path.exists():
        continue
    with open(simulator_config_file_path, 'r', encoding='utf-8') as f:
        simulator_config_yaml = yaml.safe_load(f)

    if simulator_config_yaml == config_yaml['simulator']:
        found_flg = True
        break

if not found_flg:
    raise ValueError('No simulator configuration matches the target one.')

# make time_series_df
time_series_df_map = {}

# regarding mpc
mpc_dir_path = simulator_dir_path / 'mpc'
if mpc_dir_path.exists() and any(flg for flg in config_yaml['target']['control_method']['mpc'].values()):
    time_series_df_map['mpc'] = {}
    for config_dir_path in mpc_dir_path.glob('config_*'):
        # check the configuration of mpc matches the target one
        mpc_config_file_path = config_dir_path / 'config.yaml'
        if not mpc_config_file_path.exists():
            continue
        with open(mpc_config_file_path, 'r', encoding='utf-8') as f:
            mpc_config_yaml = yaml.safe_load(f)
        
        # check num_phases
        num_phases = mpc_config_yaml['phases']['4-road']
        if not config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
            continue
        del mpc_config_yaml['phases']

        # check signal_change_weight
        signal_change_weight = mpc_config_yaml['objective_function']['signal_change']['weight']
        if not config_yaml['target']['signal_change_weight'][f"{num_phases}-phase"] == signal_change_weight:
            continue
        del mpc_config_yaml['objective_function']['signal_change']['weight']
        
        # check other parameters
        if mpc_config_yaml != config_yaml['mpc']:
            continue

        # load time_series_df
        time_series_df = None
        for intersection_dir_path in config_dir_path.glob('intersection_*'):
            time_series_file_path = intersection_dir_path / 'performance_metrics.csv'
            if not time_series_file_path.exists():
                continue
            with open(time_series_file_path, 'r', encoding='utf-8') as f:
                tmp_time_series_df = pd.read_csv(f)
            
            tmp_time_series_df = tmp_time_series_df[['time', config_yaml['target']['performance_metric']]].copy()
            tmp_time_series_df = tmp_time_series_df.dropna().reset_index(drop=True)
            tmp_time_series_df['intersection'] = int(re.match(r"intersection_(\d+)", intersection_dir_path.name).group(1))
            
            if time_series_df is None:
                time_series_df = tmp_time_series_df.copy()
            else:
                time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True)
        
        time_series_df = time_series_df.groupby('time')[config_yaml['target']['performance_metric']].mean()
        time_series_df = time_series_df.rolling(window=config_yaml['target']['moving_average_window'], min_periods=1).mean()
        time_series_df = time_series_df.dropna()
        time_series_df = time_series_df.reset_index()
        time_series_df_map['mpc'][num_phases] = time_series_df


# regarding scoot
scoot_dir_path = simulator_dir_path / 'scoot'
if scoot_dir_path.exists() and config_yaml['target']['control_method']['scoot']:
    for config_dir_path in scoot_dir_path.glob('config_*'):
        # check the configuration of scoot matches the target one
        scoot_config_file_path = config_dir_path / 'config.yaml'
        if not scoot_config_file_path.exists():
            continue
        with open(scoot_config_file_path, 'r', encoding='utf-8') as f:
            scoot_config_yaml = yaml.safe_load(f)

        if scoot_config_yaml != config_yaml['scoot']:
            continue

        # load time_series_df
        time_series_df = None
        for intersection_dir_path in config_dir_path.glob('intersection_*'):
            time_series_file_path = intersection_dir_path / 'performance_metrics.csv'
            if not time_series_file_path.exists():
                continue
            with open(time_series_file_path, 'r', encoding='utf-8') as f:
                tmp_time_series_df = pd.read_csv(f)
            
            tmp_time_series_df = tmp_time_series_df[['time', config_yaml['target']['performance_metric']]].copy()
            tmp_time_series_df = tmp_time_series_df.dropna().reset_index(drop=True)
            tmp_time_series_df['intersection'] = int(re.match(r"intersection_(\d+)", intersection_dir_path.name).group(1))
            
            if time_series_df is None:
                time_series_df = tmp_time_series_df.copy()
            else:
                time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True)
        
        time_series_df = time_series_df.groupby('time')[config_yaml['target']['performance_metric']].mean()
        time_series_df = time_series_df.rolling(window=config_yaml['target']['moving_average_window'], min_periods=1).mean()
        time_series_df = time_series_df.dropna()
        time_series_df = time_series_df.reset_index()
        time_series_df_map['scoot'] = time_series_df

# set save_dir_path 
save_dir_path = analysis_dir_path / 'performance_metric_time_series_plots' / 'multiple_intersection' / config_yaml['target']['layout'] / config_yaml['target']['inflow']
save_dir_path.mkdir(parents=True, exist_ok=True)

# plot time series
time_series_df = None
if 'mpc' in time_series_df_map:
    for num_phases, tmp_time_series_df in time_series_df_map['mpc'].items():
        tmp_time_series_df['method'] = f"{num_phases}-phase MPC"
        if time_series_df is None:
            time_series_df = tmp_time_series_df.copy()
        else:
            time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True)
if 'scoot' in time_series_df_map:
    tmp_time_series_df = time_series_df_map['scoot']
    tmp_time_series_df['method'] = 'SCOOT'
    if time_series_df is None:
        time_series_df = tmp_time_series_df.copy()
    else:
        time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True)

fig, ax = plt.subplots()

sns.lineplot(
    ax=ax,
    data=time_series_df,
    x='time',
    y=config_yaml['target']['performance_metric'],
    hue='method',
    palette=config_yaml['figure']['palette'],
)

ax.set_title(config_yaml['figure']['title']['label'][config_yaml['target']['performance_metric']])
ax.set_xlabel(config_yaml['figure']['x_axis']['label'])
ax.set_ylabel(config_yaml['figure']['y_axis']['label'][config_yaml['target']['performance_metric']]) 
ax.legend(title='')

fig.tight_layout()
fig.savefig(save_dir_path / f"time_series_plot_{config_yaml['target']['performance_metric']}.png", format='png')

print('Finished!')
    





