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

def main():
    # reflect figure configuration
    init_figure_config()

    # get config_yaml
    config_file_path = Path(__file__).parent / 'config.yaml'
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config_yaml = yaml.safe_load(f)

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'performance_metric_time_series_plots' / 'single_intersection' / 'road_level' / config_yaml['target']['layout'] / f"intersection_{config_yaml['target']['intersection_id']}"
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # get time_series_df_map
    time_series_df_map = getTimeSeriesDfMap(config_yaml)

    # plot time series
    plotTimeSeries(time_series_df_map, config_yaml, save_dir_path)
    return

def getTimeSeriesDfMap(config_yaml):
    time_series_df_map = {}
    scenario_dir_path = root_dir_path / 'data' / 'performance_metrics' / config_yaml['target']['layout'] / config_yaml['target']['inflow']
    
    for simulator_dir_path in scenario_dir_path.rglob('simulator_*'):
        # chech simulator config
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
            # check mpc config
            with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                method_config = yaml.safe_load(f)

            num_phases = method_config['phases']['4-road'] # TODO: currently only support 4-road intersection
            
            if num_phases not in [4, 8, 17]:
                raise ValueError(f"Not supported num_phases: {num_phases}")
            
            if not config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]: continue 
            
            del method_config['phases']

            config_yaml['mpc']['objective_function']['signal_change']['weight'] = config_yaml['target']['signal_change_weight'][f"{num_phases}-phase"]

            if method_config != config_yaml['mpc']: continue 
            
            # get time_series_df
            time_series_df_map[config_yaml['figure']['title']['mpc'][f"{num_phases}-phase"]] = getTimeSeriesDf(
                intersection_dir_path=method_dir_path / f"intersection_{config_yaml['target']['intersection_id']}",
            )

        # regarding scoot
        scoot_dir_path = simulator_dir_path / 'scoot'
        for method_dir_path in scoot_dir_path.glob('config_*'):
            with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                method_config = yaml.safe_load(f)

            if not config_yaml['target']['control_method']['scoot']: continue

            if method_config != config_yaml['scoot']: continue
            
            time_series_df_map[config_yaml['figure']['title']['scoot']] = getTimeSeriesDf(
                intersection_dir_path=method_dir_path / f"intersection_{config_yaml['target']['intersection_id']}",
            )

        # regarding drl
        drl_dir_path = simulator_dir_path / 'drl'
        for method_dir_path in drl_dir_path.glob('config_*'):
            with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                method_config = yaml.safe_load(f)

            vehicle_state_info = method_config['state']['vehicle']

            if all(not vehicle_state_info[key] for key in ['position', 'speed', 'route']):
                if not config_yaml['target']['control_method']['drl']['macro']: continue

                method = config_yaml['figure']['title']['drl']['macro']
            
            elif all(vehicle_state_info[key] for key in ['position', 'speed', 'route']) and method_config['num_phases'] == 4:
                if not config_yaml['target']['control_method']['drl']['4-phase']: continue

                method = config_yaml['figure']['title']['drl']['4-phase']
            
            elif all(vehicle_state_info[key] for key in ['position', 'speed', 'route']) and method_config['num_phases'] == 17:
                if not config_yaml['target']['control_method']['drl']['proposed']: continue

                method = config_yaml['figure']['title']['drl']['proposed']
            
            else:
                continue
            
            del method_config['num_phases']
            del vehicle_state_info['position'], vehicle_state_info['speed'], vehicle_state_info['route']

            if method_config != config_yaml['drl']:
                continue
            
            time_series_df_map[method] = getTimeSeriesDf(
                intersection_dir_path=method_dir_path / f"intersection_{config_yaml['target']['intersection_id']}",
            )

    if len(time_series_df_map) != 2:
        raise ValueError(f"Not supported time_series_df_map length: {len(time_series_df_map)}")

    return time_series_df_map

def getTimeSeriesDf(intersection_dir_path):
    time_series_df = None
    for road_dir_path in intersection_dir_path.glob('road_*'): 
        with open(road_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
            tmp_time_series_df = pd.read_csv(f)
        
        tmp_time_series_df['road'] = int(re.match(r"road_(\d+)", road_dir_path.name).group(1))

        if time_series_df is None:
            time_series_df = tmp_time_series_df.copy()
        else:
            time_series_df = pd.concat([time_series_df, tmp_time_series_df], axis=0, ignore_index=True)

    return time_series_df

def plotTimeSeries(time_series_df_map, config_yaml, save_dir_path):
    # get max_value
    max_value = 0
    for method, tmp_time_series_df in time_series_df_map.items():
        max_value = max(max_value, tmp_time_series_df[config_yaml['target']['metric']].max())

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    for idx, (ax, method) in enumerate(zip(axes, [config_yaml['figure']['title']['drl']['4-phase'], config_yaml['figure']['title']['drl']['proposed']])):
        tmp_time_series_df = time_series_df_map[method]
        
        # add line_style column
        tmp_time_series_df['line_style'] = ''
        for road_id in tmp_time_series_df['road'].unique():
            tmp_time_series_df.loc[tmp_time_series_df['road'] == road_id, 'line_style'] = config_yaml['figure']['line_style'][road_id]
        
        sns.lineplot(
            ax=ax,
            data=tmp_time_series_df,
            x='time',
            y=config_yaml['target']['metric'],
            hue='road',
            style='line_style',
            dashes={
                'solid': (None, None),
                'dashed': (2, 1),
            },
            palette=config_yaml['figure']['palette'],
        )
        
        if idx == 0:
            ax.set_title(method)
            ax.set_xlabel('')
            ax.set_ylabel(config_yaml['figure']['y_axis']['label'], fontweight='bold')
            ax.set_xlim(left=-20, right=tmp_time_series_df['time'].max() + 20)
            ax.set_ylim(bottom=-20, top= max_value + 20)
            ax.get_legend().remove()

        elif idx == 1:
            ax.set_title(method)
            ax.set_xlabel(config_yaml['figure']['x_axis']['label'], fontweight='bold')
            ax.set_ylabel(config_yaml['figure']['y_axis']['label'], fontweight='bold')
            ax.set_xlim(left=-20, right=tmp_time_series_df['time'].max() + 20)
            ax.set_ylim(bottom=-20, top= max_value + 20)
        
            handles, labels = ax.get_legend_handles_labels()
            new_handles = []
            new_labels = []
            for handle, label in zip(handles, labels):
                match_obj = re.match(r"(\d+)", label)
                if match_obj:
                    new_handles.append(handle)
                    new_labels.append(config_yaml['figure']['legend'][int(match_obj.group(1))])
            ax.legend(new_handles, new_labels, title='', ncol=len(new_labels), loc='upper center', bbox_to_anchor=(0.5, -0.3))

    fig.tight_layout()
    fig.savefig(save_dir_path / f"queue_time_series.png")
    return

if __name__ == '__main__':
    main()



