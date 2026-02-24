import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from libs.figure_config import initFigureConfig


# load config.yaml
with open(root_dir_path / 'misc' / 'analysis' / 'performance_metric_box_plots' / 'config.yaml', 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set figure configuration
initFigureConfig()

# initialize performance_df and intersection_dir_path_map
performance_metric_map_list = []
intersection_dir_path_map = {}
data_dir_path = root_dir_path / 'data'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'
layout_dir_path = performance_metrics_dir_path / config_yaml['target']['layout']
for simulator_dir_path in layout_dir_path.rglob('simulator_*'):
    with open(simulator_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
        simulator_config = yaml.safe_load(f)
    
    # if seed is fixed in the plot config, check it
    if config_yaml['target']['seed']['fix_flg'] and simulator_config['seed'] != config_yaml['target']['seed']['fix_value']:
        continue
    del simulator_config['seed'] 

    # check simulator config matches the plot config
    if simulator_config != config_yaml['simulator']:
        continue
    
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
        
        for intersection_dir_path in method_dir_path.glob('intersection_*'):
            # make performance_metric_map
            performance_metric_map = {
                'id': len(performance_metric_map_list) + 1,
                'method': f"{num_phases}-phase MPC",
            }
            with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                time_series_df = pd.read_csv(f)
            for performance_metric in config_yaml['target']['performance_metrics']:
                if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                    average_value = time_series_df[performance_metric].dropna().mean()
                elif performance_metric == 'delay_avg':
                    average_value = time_series_df[f"{performance_metric}_{config_yaml['target']['delay_type']}"].dropna().mean()
                else:
                    raise NotImplementedError(f"Not supported performance metric: {performance_metric}")  
                
                performance_metric_map[performance_metric] = average_value
            
            performance_metric_map_list.append(performance_metric_map)
            intersection_dir_path_map[performance_metric_map['id']] = intersection_dir_path

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
        
        for intersection_dir_path in method_dir_path.glob('intersection_*'):
            # make performance_metric_map
            performance_metric_map = {
                'id': len(performance_metric_map_list) + 1,
                'method': 'SCOOT',
            }
            with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                time_series_df = pd.read_csv(f)
            for performance_metric in config_yaml['target']['performance_metrics']:
                if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                    average_value = time_series_df[performance_metric].dropna().mean()
                elif performance_metric == 'delay_avg':
                    average_value = time_series_df[f"{performance_metric}_{config_yaml['target']['delay_type']}"].dropna().mean()
                else:
                    raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                
                performance_metric_map[performance_metric] = average_value
            
            performance_metric_map_list.append(performance_metric_map)
            intersection_dir_path_map[performance_metric_map['id']] = intersection_dir_path
            
performance_metric_df = pd.DataFrame(
    performance_metric_map_list, 
    columns=['id', 'method'] + config_yaml['target']['performance_metrics']
)

# make save_dir
save_dir_path = data_dir_path / 'analysis' / 'performance_box_plot' / config_yaml['target']['layout']
save_dir_path.mkdir(parents=True, exist_ok=True)

# make information used for making plots
order_list = []
if config_yaml['target']['control_method']['scoot']:
    order_list.append('SCOOT')
for num_phases in [4, 8, 17]:
    if config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
        order_list.append(f"{num_phases}-phase MPC")

# plot figure
for performance_metric in config_yaml['target']['performance_metrics']:
    fig, ax = plt.subplots()

    sns.boxplot(
        ax=ax,
        x='method',
        y=performance_metric,
        data=performance_metric_df,
        hue='method',
        legend=False,
        palette='Set2',
        width=0.5,
        linewidth=2.5,
        showmeans=True,
        meanprops={
            "marker": "o",           # 形をダイヤ(D)や丸(o)に変更（三角より目立ちます）
            "markerfacecolor": "white", # 中の色を白抜きにすると、箱の色の上でも見やすい
            "markeredgecolor": "black", # 縁取りを黒にしてハッキリさせる
            "markersize": 10,        # サイズを大きく（デフォルトはかなり小さいです）
            "markeredgewidth": 2     # 縁取りの線の太さ
        },
        showfliers=False,
        order=order_list,
    )

    sns.stripplot(
        ax=ax,
        x='method',
        y=performance_metric,
        data=performance_metric_df,
        color='black',
        alpha=0.4,
        jitter=True,
        order=order_list,
        size=8,
    )

    ax.set_title(config_yaml['figure']['title'][performance_metric])
    ax.set_xlabel('')
    ax.set_ylabel(config_yaml['figure']['y_axis']['label'][performance_metric])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(save_dir_path / f"{performance_metric}.png", format='png')
    plt.close(fig)

print('Finished!')