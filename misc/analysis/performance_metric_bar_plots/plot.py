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
config_file_path = root_dir_path / 'misc' / 'analysis' / 'performance_metric_bars' / 'config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set data and performance_metrics directory paths
data_dir_path = root_dir_path / 'data'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'
save_base_dir_path = data_dir_path / 'analysis' / 'performance_metric_bar_plots'

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


for inflow in performance_metric_df['inflow'].unique().tolist():
    tmp_performance_metric_df = performance_metric_df[performance_metric_df['inflow'] == inflow]

    # make hue_order_list
    hue_list = tmp_performance_metric_df['method'].unique().tolist()
    ideal_hue_order_list = ['SCOOT'] + [f"{num_phases}-phase MPC" for num_phases in [4, 8, 17]]
    hue_order_list = [method for method in ideal_hue_order_list if method in hue_list]

    for performance_metric in config_yaml['figure']['performance_metrics']:
        fig, ax = plt.subplots()

        sns.barplot(
            ax = ax,
            data = tmp_performance_metric_df,
            x = 'intersection',
            hue = 'method',
            y = performance_metric,
            palette=config_yaml['figure']['palette'],
            order=sorted(tmp_performance_metric_df['intersection'].unique().tolist()),
            hue_order=hue_order_list,
        )

        ax.set_title(config_yaml['figure']['title'][inflow])
        ax.set_xlabel(config_yaml['figure']['x_axis']['label'])
        ax.set_xticks(range(len(config_yaml['figure']['x_axis']['tick_labels'])))
        ax.set_xticklabels(config_yaml['figure']['x_axis']['tick_labels'])
        ax.set_ylabel(config_yaml['figure']['y_axis']['label'][performance_metric])
        ax.set_ylim(0, tmp_performance_metric_df[performance_metric].max() * 1.3)
        ax.legend(title='')

        fig.tight_layout()

        save_dir_path = save_base_dir_path / config_yaml['figure']['layout'] / inflow
        save_dir_path.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_dir_path / f"{performance_metric}_bar.png", format='png')

        plt.close(fig)

print('Finished!')






            