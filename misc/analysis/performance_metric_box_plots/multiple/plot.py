import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..'/ '..').resolve()
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
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'performance_metric_box_plots' / 'multiple'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # plot figure
    plotFigure(config_yaml, performance_metric_df, save_dir_path)
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

            # get inflow
            inflow = simulator_dir_path.parent.name
            if inflow not in config_yaml['target']['inflow']: continue

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
                    performance_metric_map = getPerformanceMetricMap(
                        id=len(performance_metric_map_list) + 1,
                        method=config_yaml['figure']['x_axis']['label']['mpc'][f"{num_phases}-phase"],
                        layout=layout,
                        inflow=inflow,
                        intersection_dir_path=intersection_dir_path,
                        config_yaml=config_yaml
                    )
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
                    performance_metric_map = getPerformanceMetricMap(
                        id=len(performance_metric_map_list) + 1,
                        method=config_yaml['figure']['x_axis']['label']['scoot'],
                        layout=layout,
                        inflow=inflow,
                        intersection_dir_path=intersection_dir_path,
                        config_yaml=config_yaml
                    )
                    performance_metric_map_list.append(performance_metric_map)

            # regarding drl
            drl_dir_path = simulator_dir_path / 'drl'
            for method_dir_path in drl_dir_path.glob('config_*'):
                with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                    method_config = yaml.safe_load(f)

                vehicle_state_info = method_config['state']['vehicle']

                if all(not vehicle_state_info[key] for key in ['position', 'speed', 'route']):
                    method_name = config_yaml['figure']['x_axis']['label']['drl']['macro']

                elif all(vehicle_state_info[key] for key in ['position', 'speed', 'route']) and method_config['num_phases'] == 4:
                    method_name = config_yaml['figure']['x_axis']['label']['drl']['4-phase']

                elif all(vehicle_state_info[key] for key in ['position', 'speed', 'route']) and method_config['num_phases'] == 17:
                    method_name = config_yaml['figure']['x_axis']['label']['drl']['proposed']

                else:
                    continue
                
                del method_config['num_phases']
                del vehicle_state_info['position'], vehicle_state_info['speed'], vehicle_state_info['route']

                if method_config != config_yaml['drl']:
                    continue
                
                for intersection_dir_path in method_dir_path.glob('intersection_*'):
                    # make performance_metric_map
                    performance_metric_map = getPerformanceMetricMap(
                        id=len(performance_metric_map_list) + 1,
                        method=method_name,
                        layout=layout,
                        inflow=simulator_dir_path.parent.name,
                        intersection_dir_path=intersection_dir_path,
                        config_yaml=config_yaml
                    )
                    performance_metric_map_list.append(performance_metric_map)


    performance_metric_df = pd.DataFrame(
        performance_metric_map_list, 
        columns=['id', 'method', 'layout', 'inflow', 'intersection'] + config_yaml['target']['performance_metrics']
    )

    performance_metric_map_list = []
    for (method, layout, inflow), group_df in performance_metric_df.groupby(by=['method', 'layout', 'inflow'])[config_yaml['target']['performance_metrics']]:
        performance_metric_map = {
            'id': len(performance_metric_map_list) + 1,
            'method': method,
            'layout': layout,
            'inflow': inflow,
        }
        for performance_metric in config_yaml['target']['performance_metrics']:
            performance_metric_map[f"{performance_metric}_mean"] = group_df[performance_metric].mean()
            performance_metric_map[f"{performance_metric}_std"] = group_df[performance_metric].std()

        performance_metric_map['count'] = group_df.shape[0]

        performance_metric_map_list.append(performance_metric_map)

    performance_metric_df = pd.DataFrame(
        performance_metric_map_list,    
        columns=['id', 'method', 'layout', 'inflow', 'count'] + [f"{performance_metric}_mean" for performance_metric in config_yaml['target']['performance_metrics']] + [f"{performance_metric}_std" for performance_metric in config_yaml['target']['performance_metrics']]
    )
    performance_metric_df = performance_metric_df.reset_index()
    performance_metric_df.to_csv(root_dir_path / 'data' / 'analysis' / 'performance_metric_box_plots' / 'multiple' / 'performance_metric_df.csv', index=False)
    return performance_metric_df

def getPerformanceMetricMap(id, method, layout, inflow, intersection_dir_path, config_yaml):
    performance_metric_map = {
        'id': id,
        'method': method,
        'layout': layout,
        'inflow': inflow,
        'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
    }
    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')

    for performance_metric in config_yaml['target']['performance_metrics']:
        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
        elif performance_metric == 'delay_avg':
            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_yaml['target']['delay_type']}"].dropna().mean()
        elif performance_metric == 'phase_changes':
            performance_metric_map[performance_metric] = time_series_df['phase'].diff().fillna(0).ne(0).sum()
        elif performance_metric == 'reward':
            if 'reward' in time_series_df.columns:
                performance_metric_map[performance_metric] = time_series_df['reward'].fillna(0).sum()
        elif performance_metric == 'spillback_events':
            if config_yaml['target']['spillback']['count_type'] == 'intersection':
                performance_metric_map[performance_metric] = (time_series_df['queue_max'] > config_yaml['target']['spillback']['threshold'][layout]).sum() * config_yaml['simulator']['time_step']
            elif config_yaml['target']['spillback']['count_type'] == 'road':
                pass
            else:
                raise NotImplementedError(f"Not supported spillback count type: {config_yaml['target']['spillback']['count_type']}")
        else:
            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")

    if not config_yaml['target']['spillback']['count_type'] == 'road':
        return performance_metric_map

    performance_metric_map['spillback_events'] = 0
    for road_dir_path in intersection_dir_path.glob('road_*'):
        with open(road_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
            time_series_df = pd.read_csv(road_dir_path / 'performance_metrics.csv')
        performance_metric_map['spillback_events'] += (time_series_df['queue_max'] > config_yaml['target']['spillback']['threshold'][layout]).sum() * config_yaml['simulator']['time_step']
        
    return performance_metric_map

def getOrderList(config_yaml, performance_metric):
    order_list = []

    if performance_metric != 'reward':
        if config_yaml['target']['control_method']['scoot']:
            order_list.append(config_yaml['figure']['x_axis']['label']['scoot'])
        
        for num_phases in [4, 8, 17]:
            if config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
                order_list.append(config_yaml['figure']['x_axis']['label']['mpc'][f"{num_phases}-phase"])

    if config_yaml['target']['control_method']['drl']['macro']:
        order_list.append(config_yaml['figure']['x_axis']['label']['drl']['macro'])

    if config_yaml['target']['control_method']['drl']['4-phase']:
        order_list.append(config_yaml['figure']['x_axis']['label']['drl']['4-phase'])

    if config_yaml['target']['control_method']['drl']['proposed']:
        order_list.append(config_yaml['figure']['x_axis']['label']['drl']['proposed'])
    
    return order_list

def getColorMap(config_yaml, figure_type):
    if figure_type == 'boxplot':
        method_list = []
        if config_yaml['target']['control_method']['scoot']:
            method_list.append(config_yaml['figure']['x_axis']['label']['scoot'])
        for num_phases in [4, 8, 17]:
            if config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
                method_list.append(config_yaml['figure']['x_axis']['label']['mpc'][f"{num_phases}-phase"])
        if config_yaml['target']['control_method']['drl']['macro']:
            method_list.append(config_yaml['figure']['x_axis']['label']['drl']['macro'])
        if config_yaml['target']['control_method']['drl']['4-phase']:
            method_list.append(config_yaml['figure']['x_axis']['label']['drl']['4-phase'])
        if config_yaml['target']['control_method']['drl']['proposed']:
            method_list.append(config_yaml['figure']['x_axis']['label']['drl']['proposed'])
        
        color_list = sns.color_palette(config_yaml['figure']['boxplot']['color']['palette'], n_colors=len(method_list))
        return dict(zip(method_list, color_list))

    elif figure_type == 'stripplot':
        color_map = {}
        color_list = sns.color_palette(config_yaml['figure']['stripplot']['color']['palette'], n_colors=max([group['color_id'] for group in config_yaml['figure']['stripplot']['color']['group']]))
        for _, group in enumerate(config_yaml['figure']['stripplot']['color']['group']):
            color_map[config_yaml['figure']['legend']['label'][group['id']]] = color_list[group['color_id'] - 1]
        return color_map

    else:
        raise NotImplementedError(f"Not supported figure type: {figure_type}")

def plotFigure(config_yaml, performance_metric_df, save_dir_path):
    # get layout_group_map
    layout_group_map = {}
    for group in config_yaml['figure']['stripplot']['color']['group']:
        for layout in group['layout']:
            layout_group_map[layout] = config_yaml['figure']['legend']['label'][group['id']]
    
    if len(layout_group_map) != len(config_yaml['target']['layout']):
        raise ValueError(f"Invalid layout group setting for stripplot. The number of groups must match the number of layouts. layout_group_map: {layout_group_map}, target_layout: {config_yaml['target']['layout']}")

    # set group column for performance_metric_df
    performance_metric_df['group'] = performance_metric_df['layout'].map(layout_group_map)

    for performance_metric in config_yaml['target']['performance_metrics']:
        fig, ax = plt.subplots()
        
        sns.boxplot(
            ax=ax,
            x='method',
            y=f"{performance_metric}_mean",
            data=performance_metric_df,
            hue='method',
            legend=False,
            palette=getColorMap(config_yaml, 'boxplot'),
            width=0.5,
            linewidth=2.5,
            showmeans=True,
            meanprops={
                'marker': 'o',           # 形をダイヤ(D)や丸(o)に変更（三角より目立ちます）
                'markerfacecolor': 'white', # 中の色を白抜きにすると、箱の色の上でも見やすい
                'markeredgecolor': 'black', # 縁取りを黒にしてハッキリさせる
                'markersize': 10,        # サイズを大きく（デフォルトはかなり小さいです）
                'markeredgewidth': 2     # 縁取りの線の太さ
            },
            showfliers=False,
            order=getOrderList(config_yaml, performance_metric),
        )

        sns.stripplot(
            ax=ax,
            x='method',
            y=f"{performance_metric}_mean",
            data=performance_metric_df,
            hue='group',
            palette=getColorMap(config_yaml, 'stripplot'),
            alpha=config_yaml['figure']['stripplot']['alpha'],
            jitter=True,
            order=getOrderList(config_yaml, performance_metric),
            size=8,
        )
        ax.set_title(config_yaml['figure']['title'][performance_metric])
        ax.set_xlabel('')
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        ax.set_ylabel(config_yaml['figure']['y_axis']['label'][performance_metric], fontweight='bold')
        if performance_metric == 'spillback_events':
            ax.set_ylim(bottom=-10, top=performance_metric_df[f"{performance_metric}_mean"].max() * 1.2)
        else:
            ax.set_ylim(bottom=0, top=performance_metric_df[f"{performance_metric}_mean"].max() * 1.2)

        ax.legend(title=config_yaml['figure']['legend']['title'], ncol=config_yaml['figure']['legend']['ncol'])

        fig.tight_layout()
        fig.savefig(save_dir_path / f"{performance_metric}.png", format='png')
        plt.close(fig)

    return


if __name__ == "__main__":
    main()
