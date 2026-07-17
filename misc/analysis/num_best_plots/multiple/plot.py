import sys
from pathlib import Path

root_dir_path = (Path(__file__).parent / '..' / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import pandas as pd
import yaml
import re

import matplotlib.pyplot as plt
import seaborn as sns

from libs.figure_config import init_figure_config

def main():
    # initialize figure config
    init_figure_config()

    # get config_yaml
    config_file_path = Path(__file__).parent / 'config.yaml'
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config_yaml = yaml.safe_load(f)

    # get performance_metric_df
    performance_metric_df = getPerformanceMetricDf(config_yaml)

    # get performance_metric_stat_df
    performance_metric_stat_df = getPerformanceMetricStatDf(performance_metric_df, config_yaml)

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'num_best_plots' / 'multiple'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # plot num_best
    plotNumBest(performance_metric_stat_df, config_yaml, save_dir_path)
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
                    for performance_metric, flg in config_yaml['target']['performance_metrics'].items():
                        if not flg: continue
                        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
                        elif performance_metric == 'delay_avg':
                            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_yaml['target']['delay_type']}"].dropna().mean()
                        elif performance_metric == 'reward':
                            pass
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
                        'method': config_yaml['figure']['x_axis']['label']['scoot'],
                        'layout': layout,
                        'inflow': simulator_dir_path.parent.name,
                        'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    for performance_metric, flg in config_yaml['target']['performance_metrics'].items():
                        if not flg: continue
                        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
                        elif performance_metric == 'delay_avg':
                            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_yaml['target']['delay_type']}"].dropna().mean()
                        elif performance_metric == 'reward':
                            pass
                        else:
                            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                    
                    performance_metric_map_list.append(performance_metric_map)

            # regarding drl
            drl_dir_path = simulator_dir_path / 'drl'
            for method_dir_path in drl_dir_path.glob('config_*'):
                with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                    method_config = yaml.safe_load(f)

                vehicle_state_info = method_config['state']['vehicle']
                
                if all (not vehicle_state_info[key] for key in ['position', 'speed', 'route']):
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
                    performance_metric_map = {
                        'id': len(performance_metric_map_list) + 1,
                        'method': method_name,
                        'layout': layout,
                        'inflow': simulator_dir_path.parent.name,
                        'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    for performance_metric, flg in config_yaml['target']['performance_metrics'].items():
                        if not flg: continue
                        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
                        elif performance_metric == 'delay_avg':
                            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_yaml['target']['delay_type']}"].dropna().mean()
                        elif performance_metric == 'reward':
                            performance_metric_map[performance_metric] = time_series_df['reward'].fillna(0).sum()
                        else:
                            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                    
                    performance_metric_map_list.append(performance_metric_map)


    performance_metric_df = pd.DataFrame(
        performance_metric_map_list, 
        columns=['id', 'method', 'layout', 'inflow', 'intersection'] + [performance_metric for performance_metric, flg in config_yaml['target']['performance_metrics'].items() if flg]
    )

    grouped_performance_metric_df = performance_metric_df.groupby(by=['method', 'layout', 'inflow'])[[performance_metric for performance_metric, flg in config_yaml['target']['performance_metrics'].items() if flg]]
    performance_metric_df = grouped_performance_metric_df.mean()
    performance_metric_df['count'] = grouped_performance_metric_df.count().iloc[:, 0]
    performance_metric_df = performance_metric_df.reset_index()
    performance_metric_df['id'] = range(1, len(performance_metric_df) + 1)
    performance_metric_df = performance_metric_df[['id', 'method', 'layout', 'inflow', 'count'] + [performance_metric for performance_metric, flg in config_yaml['target']['performance_metrics'].items() if flg]]
    return performance_metric_df

def getPerformanceMetricStatDf(performance_metric_df, config_yaml):
    # make performance_metric_stat_df
    performance_metric_stat_map_list = []
    for method in performance_metric_df['method'].unique().tolist():
        tmp_performance_metric_df = performance_metric_df[performance_metric_df['method'] == method]
        if tmp_performance_metric_df.empty:
            continue
        for performance_metric, flg in config_yaml['target']['performance_metrics'].items():
            if not flg: continue
            performance_metric_stat_map_list.append({
                'id': len(performance_metric_stat_map_list) + 1,
                'performance_metric': performance_metric,
                'method': method,
                'mean': tmp_performance_metric_df[performance_metric].mean(),
                'worst': tmp_performance_metric_df[performance_metric].min() if performance_metric in ['speed_avg', 'reward'] else tmp_performance_metric_df[performance_metric].max(),
                'std': tmp_performance_metric_df[performance_metric].std(),
            })
    performance_metric_stat_df = pd.DataFrame(
        performance_metric_stat_map_list, 
        columns=['id', 'performance_metric', 'method', 'mean', 'worst', 'std']
    )

    # add improve_rate column
    improve_rate_list = [None] * len(performance_metric_stat_df)
    scoot_stat_df = performance_metric_stat_df[performance_metric_stat_df['method'] == config_yaml['figure']['x_axis']['label']['scoot']].copy()
    for _, scoot_stat_row in scoot_stat_df.iterrows():
        performance_metric = scoot_stat_row['performance_metric']
        reference_value = scoot_stat_row['mean']
        for method in config_yaml['figure']['x_axis']['label']['mpc'].values():
            target_stat_row = performance_metric_stat_df[
                (performance_metric_stat_df['method'] == method) &
                (performance_metric_stat_df['performance_metric'] == performance_metric)
            ]
            if target_stat_row.empty:
                continue
            target_id = target_stat_row['id'].values[0]
            target_value = target_stat_row['mean'].values[0]
            improve_rate_list[target_id - 1] = (target_value - reference_value) / reference_value * 100
        
        for method in config_yaml['figure']['x_axis']['label']['drl'].values():
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
        for performance_metric, flg in config_yaml['target']['performance_metrics'].items() if flg
    }
    for layout in performance_metric_df['layout'].unique().tolist():
        for inflow in performance_metric_df['inflow'].unique().tolist():
            
            tmp_performance_metric_df = performance_metric_df[
                (performance_metric_df['layout'] == layout) &
                (performance_metric_df['inflow'] == inflow)
            ]
            if tmp_performance_metric_df.empty:
                continue
            for performance_metric, flg in config_yaml['target']['performance_metrics'].items():
                if not flg: continue
                if performance_metric in ['speed_avg', 'reward']:
                    best_id = tmp_performance_metric_df[performance_metric].idxmax()
                elif performance_metric in ['queue_avg', 'queue_max', 'delay_avg', 'delay_max']:
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

    return performance_metric_stat_df


def plotNumBest(performance_metric_stat_df, config_yaml, save_dir_path):
    # rename performance_metric column
    performance_metric_list = []
    for _, performance_metric_stat_row in performance_metric_stat_df.iterrows():
        performance_metric_list.append(config_yaml['figure']['legend']['label'][performance_metric_stat_row['performance_metric']])
    performance_metric_stat_df['performance_metric'] = performance_metric_list
    
    fig, ax = plt.subplots()

    sns.barplot(
        ax=ax,
        data=performance_metric_stat_df,
        x='method',
        y='num_best',
        hue='performance_metric',
        order=getOrderList(config_yaml),
        palette=getColorMap(config_yaml),
        edgecolor='black',
        linewidth=1.5,
    )

    ax.set_xlabel('')
    ax.set_ylabel(config_yaml['figure']['y_axis']['label'])
    ax.set_ylim(bottom=-3)
    ax.set_title(config_yaml['figure']['title']['label'])
    ax.legend(title='')

    fig.tight_layout()
    fig.savefig(save_dir_path / 'num_best_plot.png')
    plt.close(fig)
    
    return


def getOrderList(config_yaml):
    order_list = []
    if config_yaml['target']['control_method']['scoot']:
        order_list.append(config_yaml['figure']['x_axis']['label']['scoot'])
    
    for num_phases in [4, 8, 17]:
        if config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
            order_list.append(config_yaml['figure']['x_axis']['label'][f"{num_phases}-phase"])
    
    for method, flg in config_yaml['target']['control_method']['drl'].items():
        if not flg: continue
        order_list.append(config_yaml['figure']['x_axis']['label']['drl'][method])
    return order_list

def getColorMap(config_yaml):
    performance_metric_list = [config_yaml['figure']['legend']['label'][performance_metric] for performance_metric, flg in config_yaml['target']['performance_metrics'].items() if flg]
    color_list = sns.color_palette(config_yaml['figure']['color_palette'], n_colors=len(performance_metric_list))
    color_map = {performance_metric: color_list[i] for i, performance_metric in enumerate(performance_metric_list)}
    return color_map

if __name__ == '__main__':
    main()