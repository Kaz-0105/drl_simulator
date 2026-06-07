import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import yaml
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from libs.figure_config import init_figure_config


def main():
    # reflect figure configuration
    init_figure_config()

    # get config_yaml
    config_file_path = Path(__file__).parent / 'config.yaml'
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config_yaml = yaml.safe_load(f)

    # get performance_metric_df
    performance_metric_df = getPerformanceMetricDf(config_yaml)

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'performance_metric_box_plots'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # get order_list
    order_list = getOrderList(config_yaml)

    # plot figure
    plotFigure(config_yaml, performance_metric_df, order_list, save_dir_path)
    return

def getPerformanceMetricDf(config_yaml):
    performance_metric_map_list = []

    data_dir_path = root_dir_path / 'data'
    performance_metrics_dir_path = data_dir_path / 'performance_metrics'

    for layout in config_yaml['target']['layout']:
        layout_dir_path = performance_metrics_dir_path / layout
        for simulator_dir_path in layout_dir_path.rglob('simulator_*'):
            with open(simulator_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                simulator_config = yaml.safe_load(f)
            
            if config_yaml['target']['seed']['fix_flg'] and simulator_config['seed'] != config_yaml['target']['seed']['fix_value']:
                continue

            simulator_config.pop('seed')

            if simulator_config != config_yaml['simulator']:
                continue

            # regarding mpc
            mpc_dir_path = simulator_dir_path / 'mpc'
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
                        'layout': layout,
                        'inflow': simulator_dir_path.parent.name,
                        'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    for performance_metric in config_yaml['target']['performance_metrics']:
                        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
                        elif performance_metric == 'delay_avg':
                            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_yaml['target']['delay_type']}"].dropna().mean()
                        elif performance_metric == 'num_phase_changes':
                            performance_metric_map[performance_metric] = time_series_df['phase'].diff().fillna(0).ne(0).sum()
                        else:
                            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                    
                    performance_metric_map_list.append(performance_metric_map)

            # regarding scoot
            if not config_yaml['target']['control_method']['scoot']:
                continue

            scoot_dir_path = simulator_dir_path / 'scoot'
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
                        'layout': layout,
                        'inflow': simulator_dir_path.parent.name,
                        'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    for performance_metric in config_yaml['target']['performance_metrics']:
                        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
                        elif performance_metric == 'delay_avg':
                            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_yaml['target']['delay_type']}"].dropna().mean()
                        elif performance_metric == 'num_phase_changes':
                            performance_metric_map[performance_metric] = time_series_df['phase'].diff().fillna(0).ne(0).sum()
                        else:
                            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                    
                    performance_metric_map_list.append(performance_metric_map)

            # regarding drl
            drl_dir_path = simulator_dir_path / 'drl'
            for method_dir_path in drl_dir_path.glob('config_*'):
                with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                    method_config = yaml.safe_load(f)

                vehicle_state_info = method_config['state']['vehicle']

                if all(vehicle_state_info[key] for key in ['position', 'speed', 'route']):
                    method_name = 'Micro DRL'
                elif all(not vehicle_state_info[key] for key in ['position', 'speed', 'route']):
                    method_name = 'Macro DRL'
                else:
                    continue

                del vehicle_state_info['position'], vehicle_state_info['speed'], vehicle_state_info['route']

                if method_config != config_yaml['drl']:
                    continue
                
                for intersection_dir_path in method_dir_path.glob('intersection_*'):
                    # make performance_metric_map
                    performance_metric_map = {
                        'id': len(performance_metric_map_list) + 1,
                        'method': method_name,
                        'layout': layout,
                        'inflow': simulator_dir_path.parent.name,
                        'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    for performance_metric in config_yaml['target']['performance_metrics']:
                        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
                        elif performance_metric == 'delay_avg':
                            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_yaml['target']['delay_type']}"].dropna().mean()
                        elif performance_metric == 'num_phase_changes':
                            performance_metric_map[performance_metric] = time_series_df['phase'].diff().fillna(0).ne(0).sum()
                        else:
                            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                    
                    performance_metric_map_list.append(performance_metric_map)


    performance_metric_df = pd.DataFrame(
        performance_metric_map_list, 
        columns=['id', 'method', 'layout', 'inflow', 'intersection'] + config_yaml['target']['performance_metrics']
    )
    return performance_metric_df

def getOrderList(config_yaml):
    order_list = []
    if config_yaml['target']['control_method']['scoot']:
        order_list.append('SCOOT')
    
    for num_phases in [4, 8, 17]:
        if config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
            order_list.append(f"{num_phases}-phase MPC")

    if config_yaml['target']['control_method']['drl']['macro']:
        order_list.append('Macro DRL')

    if config_yaml['target']['control_method']['drl']['micro']:
        order_list.append('Micro DRL')
    
    return order_list

def plotFigure(config_yaml, performance_metric_df, order_list, save_dir_path):
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
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        ax.set_ylabel(config_yaml['figure']['y_axis']['label'][performance_metric], fontweight='bold')
        ax.set_ylim(bottom=0)

        fig.tight_layout()
        fig.savefig(save_dir_path / f"{performance_metric}.png", format='png')
        plt.close(fig)

    return


if __name__ == "__main__":
    main()
