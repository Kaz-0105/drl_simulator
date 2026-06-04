import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import pandas as pd
import yaml
import re

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

    # get performance_metric_stat_df
    performance_metric_stat_df = getPerformanceMetricStatDf(performance_metric_df, config_yaml)

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'performance_metric_stats' / 'single_intersection'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    performance_metric_stat_df[['mean', 'worst', 'std', 'improve_rate']] = performance_metric_stat_df[['mean', 'worst', 'std', 'improve_rate']].round(2)
    performance_metric_stat_df.to_csv(save_dir_path / 'performance_metric_stat.csv', index=False, encoding='utf-8')
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
                        else:
                            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                    
                    performance_metric_map_list.append(performance_metric_map)


    performance_metric_df = pd.DataFrame(
        performance_metric_map_list, 
        columns=['id', 'method', 'layout', 'inflow', 'intersection'] + config_yaml['target']['performance_metrics']
    )
    return performance_metric_df

def getPerformanceMetricStatDf(performance_metric_df, config_yaml):
    # make performance_metric_stat_df
    performance_metric_stat_map_list = []
    for method in performance_metric_df['method'].unique().tolist():
        tmp_performance_metric_df = performance_metric_df[performance_metric_df['method'] == method]
        if tmp_performance_metric_df.empty:
            continue
        for performance_metric in config_yaml['target']['performance_metrics']:
            performance_metric_stat_map_list.append({
                'id': len(performance_metric_stat_map_list) + 1,
                'performance_metric': performance_metric,
                'method': method,
                'mean': tmp_performance_metric_df[performance_metric].mean(),
                'worst': tmp_performance_metric_df[performance_metric].min() if performance_metric == 'speed_avg' else tmp_performance_metric_df[performance_metric].max(),
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
        
        for method in ['Macro DRL', 'Micro DRL']:
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
        for performance_metric in config_yaml['target']['performance_metrics']
    }
    for layout in performance_metric_df['layout'].unique().tolist():
        for inflow in performance_metric_df['inflow'].unique().tolist():
            for intersection_id in performance_metric_df['intersection'].unique().tolist():
                tmp_performance_metric_df = performance_metric_df[
                    (performance_metric_df['layout'] == layout) &
                    (performance_metric_df['intersection'] == intersection_id) &
                    (performance_metric_df['inflow'] == inflow)
                ]
                if tmp_performance_metric_df.empty:
                    continue
                for performance_metric in config_yaml['target']['performance_metrics']:
                    if performance_metric == 'speed_avg':
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


if __name__ == '__main__':
    main()