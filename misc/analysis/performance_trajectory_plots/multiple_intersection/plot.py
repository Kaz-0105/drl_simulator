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

# set data_dir_path, analysis_dir_path, performance_metrics_dir_path, layout_dir_path, inflow_dir_path
data_dir_path = root_dir_path / 'data'
analysis_dir_path = data_dir_path / 'analysis'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'

layout_dir_path = performance_metrics_dir_path / config_yaml['target']['layout']
if not layout_dir_path.exists():
    raise FileNotFoundError(f"Directory not found: {layout_dir_path}")

inflow_dir_path = layout_dir_path / config_yaml['target']['inflow']
if not inflow_dir_path.exists():
    raise FileNotFoundError(f"Directory not found: {inflow_dir_path}")

# make time_series_df_map
time_series_df_map = {}
for simulator_dir_path in inflow_dir_path.glob('simulator_*'):
    simulator_config_file_path = simulator_dir_path / 'config.yaml'
    with open(simulator_config_file_path, 'r', encoding='utf-8') as f:
        simulator_config_yaml = yaml.safe_load(f)
    
    if simulator_config_yaml != config_yaml['simulator']:
        continue
    
    # load mpc performance metrics
    tmp_time_series_df_map = {}

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
            
        # check if the num_phases is included in the target control method for MPC
        if not config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
            continue
        
        time_series_df = None
        for intersection_dir_path in config_dir_path.glob('intersection_*'):
            with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                tmp_time_series_df = pd.read_csv(f)
            
            tmp_time_series_df = tmp_time_series_df[['time'] + list(config_yaml['target']['performance_metrics'].values())].copy()
            tmp_time_series_df['intersection'] = int(re.match(r"intersection_(\d+)", intersection_dir_path.name).group(1))
            tmp_time_series_df = tmp_time_series_df.dropna(subset=config_yaml['target']['performance_metrics'].values()).reset_index(drop=True)
            tmp_time_series_df = tmp_time_series_df[tmp_time_series_df['time'] >= config_yaml['target']['remove_time']].reset_index(drop=True)
            if time_series_df is None:
                time_series_df = tmp_time_series_df.copy()    
            else:
                time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True)
        

        time_series_df = time_series_df.groupby('time')[list(config_yaml['target']['performance_metrics'].values())].mean()
        time_series_df = time_series_df.iloc[::config_yaml['target']['down_sampling_rate'], :]
        time_series_df = time_series_df.rolling(window=config_yaml['target']['moving_average_window'], min_periods=1).mean()
        time_series_df = time_series_df.dropna(subset=list(config_yaml['target']['performance_metrics'].values()))
        time_series_df = time_series_df.reset_index()
        tmp_time_series_df_map[num_phases] = time_series_df
    
    if len(tmp_time_series_df_map) > 0: 
        time_series_df_map['mpc'] = tmp_time_series_df_map
    
    # load scoot performance metrics
    scoot_dir_path = simulator_dir_path / 'scoot'
    
    if config_yaml['target']['control_method']['scoot']:
        for config_dir_path in scoot_dir_path.glob('config_*'):
            scoot_config_file_path = config_dir_path / 'config.yaml'
            if not scoot_config_file_path.exists():
                continue

            with open(scoot_config_file_path, 'r', encoding='utf-8') as f:
                scoot_config_yaml = yaml.safe_load(f)
            
            if scoot_config_yaml != config_yaml['scoot']:
                continue

            time_series_df = None
            for intersection_dir_path in config_dir_path.glob('intersection_*'):
                performance_metrics_file_path = intersection_dir_path / 'performance_metrics.csv'
                if not performance_metrics_file_path.exists():
                    continue
                with open(performance_metrics_file_path, 'r', encoding='utf-8') as f:
                    tmp_time_series_df = pd.read_csv(f)
                
                tmp_time_series_df = tmp_time_series_df[['time'] + list(config_yaml['target']['performance_metrics'].values())].copy()
                tmp_time_series_df['intersection'] = int(re.match(r"intersection_(\d+)", intersection_dir_path.name).group(1))
                tmp_time_series_df = tmp_time_series_df.dropna(subset=config_yaml['target']['performance_metrics'].values()).reset_index(drop=True)
                tmp_time_series_df = tmp_time_series_df[tmp_time_series_df['time'] >= config_yaml['target']['remove_time']].reset_index(drop=True)

                if time_series_df is None:
                    time_series_df = tmp_time_series_df.copy()  
                else:
                    time_series_df = pd.concat([time_series_df, tmp_time_series_df], ignore_index=True) 

            time_series_df = time_series_df.groupby('time')[list(config_yaml['target']['performance_metrics'].values())].mean()
            time_series_df = time_series_df.iloc[::config_yaml['target']['down_sampling_rate'], :]
            time_series_df = time_series_df.rolling(window=config_yaml['target']['moving_average_window'], min_periods=1).mean()
            time_series_df = time_series_df.dropna(subset=list(config_yaml['target']['performance_metrics'].values()))
            time_series_df = time_series_df.reset_index()
            time_series_df_map['scoot'] = time_series_df


# make save_dir_path
save_dir_path = analysis_dir_path / 'performance_trajectory_plots' / 'multiple_intersection' / config_yaml['target']['layout'] / config_yaml['target']['inflow']
save_dir_path.mkdir(parents=True, exist_ok=True)

# plot performance trajectory
fig, ax = plt.subplots()

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

sns.lineplot(
    ax=ax,
    data=time_series_df,
    x=config_yaml['target']['performance_metrics']['x'],
    y=config_yaml['target']['performance_metrics']['y'],
    hue='method',      
    marker=config_yaml['figure']['marker']['shape']['normal'],        
    markersize=config_yaml['figure']['marker']['size']['normal'], 
    alpha=config_yaml['figure']['alpha'],      
    palette=config_yaml['figure']['palette'],
    sort=False,         
) 

# plot start point and end point
for method_name in time_series_df['method'].unique():
    method_data = time_series_df[time_series_df['method'] == method_name]
    color = config_yaml['figure']['palette'][method_name]
    
    # start point
    ax.scatter(
        x=method_data[config_yaml['target']['performance_metrics']['x']].iloc[0],
        y=method_data[config_yaml['target']['performance_metrics']['y']].iloc[0],
        color=config_yaml['figure']['palette'][method_name],
        marker=config_yaml['figure']['marker']['shape']['start'],
        s=config_yaml['figure']['marker']['size']['start'] ** 2,
        edgecolors='black', 
        linewidth=1.5, 
        zorder=5, 
        label='_nolegend_'
    )
    
    # end point
    ax.scatter(
        x=method_data[config_yaml['target']['performance_metrics']['x']].iloc[-1],
        y=method_data[config_yaml['target']['performance_metrics']['y']].iloc[-1],
        color=config_yaml['figure']['palette'][method_name], 
        marker=config_yaml['figure']['marker']['shape']['end'], 
        s=config_yaml['figure']['marker']['size']['end'] ** 2, 
        edgecolors='black', 
        linewidth=1.5, 
        zorder=5, 
        label='_nolegend_'
    )

ax.set_title(config_yaml['figure']['title']['label'])
ax.set_xlabel(config_yaml['figure']['axis']['label'][config_yaml['target']['performance_metrics']['x']])
ax.set_ylabel(config_yaml['figure']['axis']['label'][config_yaml['target']['performance_metrics']['y']])
ax.legend(title='')
fig.tight_layout()

fig.savefig(save_dir_path / 'performance_trajectory.png', format='png')

print('Finish!')



