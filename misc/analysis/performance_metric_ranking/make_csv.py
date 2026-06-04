import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import re

from libs.figure_config import init_figure_config


def main():
    # reflect figure configuration
    init_figure_config()

    # get config_info
    config_file_path = Path(__file__).parent / 'config.yaml'
    with open(config_file_path, 'r') as f:
        config_info = yaml.safe_load(f)
    
    # get performance_metric_df
    performance_metric_df = getPerformanceMetricDf(config_info)

    # get ranking_df_map
    ranking_df_map = getRankingDfMap(config_info, performance_metric_df)

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / Path(__file__).parent.name
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # save ranking_df as csv
    saveRankingCsv(ranking_df_map, save_dir_path)
    return

def getPerformanceMetricDf(config_info):
    performance_metric_map_list = []

    data_dir_path = root_dir_path / 'data'
    performance_metrics_dir_path = data_dir_path / 'performance_metrics'

    for layout in config_info['target']['layout']:
        layout_dir_path = performance_metrics_dir_path / layout
        for simulator_dir_path in layout_dir_path.rglob('simulator_*'):
            with open(simulator_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                simulator_config = yaml.safe_load(f)
            
            if config_info['target']['seed']['fix_flg'] and simulator_config['seed'] != config_info['target']['seed']['fix_value']:
                continue

            simulator_config.pop('seed')

            if simulator_config != config_info['simulator']:
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
                if not config_info['target']['control_method']['mpc'][f"{num_phases}-phase"]:
                    continue 
                del method_config['phases']

                config_info['mpc']['objective_function']['signal_change']['weight'] = config_info['target']['signal_change_weight'][f"{num_phases}-phase"]

                if method_config != config_info['mpc']:
                    continue
                
                for intersection_dir_path in method_dir_path.glob('intersection_*'):
                    # make performance_metric_map
                    performance_metric_map = {
                        'id': len(performance_metric_map_list) + 1,
                        'method': f"mpc_{num_phases}",
                        'layout': layout,
                        'inflow': simulator_dir_path.parent.name,
                        'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    for performance_metric in config_info['target']['performance_metrics']:
                        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
                        elif performance_metric == 'delay_avg':
                            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_info['target']['delay_type']}"].dropna().mean()
                        else:
                            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                    
                    performance_metric_map_list.append(performance_metric_map)

            # regarding scoot
            if not config_info['target']['control_method']['scoot']:
                continue

            scoot_dir_path = simulator_dir_path / 'scoot'
            for method_dir_path in scoot_dir_path.glob('config_*'):
                with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                    method_config = yaml.safe_load(f)

                if method_config != config_info['scoot']:
                    continue
                
                for intersection_dir_path in method_dir_path.glob('intersection_*'):
                    # make performance_metric_map
                    performance_metric_map = {
                        'id': len(performance_metric_map_list) + 1,
                        'method': 'scoot',
                        'layout': layout,
                        'inflow': simulator_dir_path.parent.name,
                        'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)),
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    for performance_metric in config_info['target']['performance_metrics']:
                        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
                        elif performance_metric == 'delay_avg':
                            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_info['target']['delay_type']}"].dropna().mean()
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
                    method_name = 'drl_micro'
                elif all(not vehicle_state_info[key] for key in ['position', 'speed', 'route']):
                    method_name = 'drl_macro'
                else:
                    continue

                del vehicle_state_info['position'], vehicle_state_info['speed'], vehicle_state_info['route']

                if method_config != config_info['drl']:
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
                    for performance_metric in config_info['target']['performance_metrics']:
                        if performance_metric in ['queue_avg', 'queue_max', 'delay_max', 'speed_avg']:
                            performance_metric_map[performance_metric] = time_series_df[performance_metric].dropna().mean()
                        elif performance_metric == 'delay_avg':
                            performance_metric_map[performance_metric] = time_series_df[f"delay_avg_{config_info['target']['delay_type']}"].dropna().mean()
                        else:
                            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                    
                    performance_metric_map_list.append(performance_metric_map)


    performance_metric_df = pd.DataFrame(
        performance_metric_map_list, 
        columns=['id', 'method', 'layout', 'inflow', 'intersection'] + config_info['target']['performance_metrics']
    )
    return performance_metric_df

def getRankingDfMap(config_info, performance_metric_df):
    ranking_df_map = {}

    for method in performance_metric_df['method'].unique().tolist():
        tmp_performance_metric_df = performance_metric_df[performance_metric_df['method'] == method]

        if config_info['target']['sort_by'] in ['queue_avg', 'queue_max', 'delay_avg', 'delay_max']:
            tmp_performance_metric_df = tmp_performance_metric_df.sort_values(by=config_info['target']['sort_by'], ascending=True).reset_index(drop=True)
        elif config_info['target']['sort_by'] == 'speed_avg':
            tmp_performance_metric_df = tmp_performance_metric_df.sort_values(by=config_info['target']['sort_by'], ascending=False).reset_index(drop=True)
        else:
            raise NotImplementedError(f"Not supported performance metric: {config_info['target']['sort_by']}")
        
        tmp_performance_metric_df['id'] = tmp_performance_metric_df.index + 1
        ranking_df = tmp_performance_metric_df.drop(columns=[performance_metric for performance_metric in config_info['target']['performance_metrics'] if performance_metric != config_info['target']['sort_by']])
        ranking_df = ranking_df.drop(columns=['method'])
        ranking_df_map[method] = ranking_df
    
    return ranking_df_map

def saveRankingCsv(ranking_df_map, save_dir_path):
    for method, ranking_df in ranking_df_map.items():
        tmp_save_dir_path = save_dir_path / method
        tmp_save_dir_path.mkdir(parents=True, exist_ok=True)
        ranking_df.to_csv(tmp_save_dir_path / 'ranking.csv', index=False)

    return


if __name__ == '__main__':
    main()
