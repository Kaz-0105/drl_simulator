import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from libs.figure_config import init_figure_config

NUM_PHASES = 17

TYPE_PHASE_MAP = {
    1: [1, 2],
    2: [3, 4],
    3: [5, 6, 7, 8],
    4: [9, 10, 11, 12],
    5: [13, 14, 15, 16],
    6: [17],
    'others': [9, 10, 11, 12, 13, 14, 15, 16, 17],
}

PHASE_TYPE_MAP = {}
for phase_type, phase_list in TYPE_PHASE_MAP.items():
    for phase_id in phase_list:
        PHASE_TYPE_MAP[phase_id] = phase_type

def main():
    # reflect figure configuration
    init_figure_config()

    # get config_yaml
    config_file_path = Path(__file__).parent / 'config.yaml'
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config_yaml = yaml.safe_load(f)

    # get phase_distribution_df
    phase_distribution_df = getPhaseDistributionDf(config_yaml)

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'phase_distribution'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # plot figure
    plotFigure(config_yaml, phase_distribution_df, save_dir_path)
    return

def getPhaseDistributionDf(config_yaml):
    phase_distribution_map_list = []

    for layout in config_yaml['target']['layout']:
        layout_dir_path = root_dir_path / 'data' / 'performance_metrics' / layout
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
                    # make phase_distribution_map
                    phase_distribution_map = {
                        'id': len(phase_distribution_map_list) + 1,
                        'method': f"mpc_{num_phases}",
                        'method_label': config_yaml['figure']['x_axis']['label']['mpc'][f"{num_phases}-phase"],
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')

                    for phase_type in [1, 2, 3, 'others']:
                        phase_list = TYPE_PHASE_MAP[phase_type]
                        
                        if phase_type == 'others':
                            phase_distribution_map['others'] = time_series_df['phase'].isin(phase_list).sum()
                        else:
                            phase_distribution_map[f"type_{phase_type}"] = time_series_df['phase'].isin(phase_list).sum()
                    
                    phase_distribution_map_list.append(phase_distribution_map)

            # regarding scoot
            scoot_dir_path = simulator_dir_path / 'scoot'
            for method_dir_path in scoot_dir_path.glob('config_*'):
                with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                    method_config = yaml.safe_load(f)

                if not config_yaml['target']['control_method']['scoot']:
                    continue

                if method_config != config_yaml['scoot']:
                    continue
                
                for intersection_dir_path in method_dir_path.glob('intersection_*'):
                    # make phase_distribution_map
                    phase_distribution_map = {
                        'id': len(phase_distribution_map_list) + 1,
                        'method': 'scoot',
                        'method_label': config_yaml['figure']['x_axis']['label']['scoot'],
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    
                    for phase_type in [1, 2, 3, 'others']:
                        phase_list = TYPE_PHASE_MAP[phase_type]

                        if phase_type == 'others':
                            phase_distribution_map['others'] = time_series_df['phase'].isin(phase_list).sum()
                        else:
                            phase_distribution_map[f"type_{phase_type}"] = time_series_df['phase'].isin(phase_list).sum()
                    
                    phase_distribution_map_list.append(phase_distribution_map)

            # regarding drl
            drl_dir_path = simulator_dir_path / 'drl'
            for method_dir_path in drl_dir_path.glob('config_*'):
                with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                    method_config = yaml.safe_load(f)

                vehicle_state_info = method_config['state']['vehicle']

                if all(vehicle_state_info[key] for key in ['position', 'speed', 'route']):
                    method = 'drl_micro'
                    method_label = config_yaml['figure']['x_axis']['label']['drl']['micro']
                elif all(not vehicle_state_info[key] for key in ['position', 'speed', 'route']):
                    method = 'drl_macro'
                    method_label = config_yaml['figure']['x_axis']['label']['drl']['macro']
                else:
                    continue

                del vehicle_state_info['position'], vehicle_state_info['speed'], vehicle_state_info['route']

                if method_config != config_yaml['drl']:
                    continue
                
                for intersection_dir_path in method_dir_path.glob('intersection_*'):
                    # make phase_distribution_map
                    phase_distribution_map = {
                        'id': len(phase_distribution_map_list) + 1,
                        'method': method,
                        'method_label': method_label,
                    }
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    for phase_type in [1, 2, 3, 'others']:
                        phase_list = TYPE_PHASE_MAP[phase_type]

                        if phase_type == 'others':
                            phase_distribution_map['others'] = time_series_df['phase'].isin(phase_list).sum()
                        else:
                            phase_distribution_map[f"type_{phase_type}"] = time_series_df['phase'].isin(phase_list).sum()
                    
                    phase_distribution_map_list.append(phase_distribution_map)

    phase_columns = [f"type_{phase_type}" for phase_type in [1, 2, 3]] + ['others']
    phase_distribution_df = pd.DataFrame(
        phase_distribution_map_list, 
        columns=['id', 'method', 'method_label'] + phase_columns,
    )

    # group by method
    phase_distribution_df = phase_distribution_df.groupby(['method', 'method_label'])[phase_columns].sum().reset_index()
    
    # change count to ratio
    phase_distribution_df[phase_columns] = phase_distribution_df[phase_columns].div(phase_distribution_df[phase_columns].sum(axis=1), axis=0)
    
    # method sort
    method_order = [
        'scoot',
        'mpc_4',
        'mpc_8',
        'mpc_17',
        'drl_macro',
        'drl_micro',
    ]
    phase_distribution_df['method'] = pd.Categorical(phase_distribution_df['method'], categories=method_order, ordered=True)
    phase_distribution_df = phase_distribution_df.sort_values('method').reset_index(drop=True)

    return phase_distribution_df

def getColorMap(phase_columns, config_yaml):
    color_list = sns.color_palette(config_yaml['figure']['color_palette'], n_colors=len(phase_columns))
    color_map = {phase_type: color_list[i] for i, phase_type in enumerate(phase_columns)}
    return color_map

def plotFigure(config_yaml, phase_distribution_df, save_dir_path):
    fig, ax = plt.subplots()
    bottom = np.zeros(len(phase_distribution_df))

    # get phase_columns and color_map
    phase_column_list = [f"type_{phase_type}" for phase_type in [1, 2, 3]] + ['others']
    color_map = getColorMap(phase_column_list, config_yaml)

    bars_list = []
    bottom_list_map = {method_id: [0] for method_id in range(len(phase_distribution_df['method']))}
    x_positions = np.arange(len(phase_distribution_df['method']))
    for phase_type in phase_column_list:
        heights = phase_distribution_df[phase_type].fillna(0)
        
        # add bars for current phase_type
        bars = ax.bar(
            x=x_positions,
            height=heights,
            bottom=bottom,
            label=config_yaml['figure']['legend']['labels'][phase_type],
            width=config_yaml['figure']['bar']['width'],              
            color=color_map[phase_type],       
            edgecolor='white',
        )
        bars_list.append(bars)   

        # update bottom_list_map
        for method_id in range(len(phase_distribution_df['method'])):
            bottom_list_map[method_id].append(bottom[method_id] + heights[method_id])

        # update bottom for next stack
        bottom = bottom + heights.values

    # add dashed lines to separate methods
    x_positions = np.arange(len(phase_distribution_df['method']))
    for method_id in range(len(phase_distribution_df['method']) - 1):
        x1 = x_positions[method_id] + config_yaml['figure']['bar']['width'] / 2      
        x2 = x_positions[method_id+1] - config_yaml['figure']['bar']['width'] / 2
        
        for phase_id, phase_type in enumerate(phase_column_list):
            y1, y2 = bottom_list_map[method_id][phase_id], bottom_list_map[method_id+1][phase_id]
            
            if phase_id == 0:
                continue 

            if y1 == y2 == 1:
                continue
            
            ax.plot(
                [x1, x2], 
                [y1, y2], 
                linestyle='--',
                color='gray',
            )

    # set values to bars
    for bars in bars_list:
        labels = []
        for bar in bars:
            height = bar.get_height()
            if height < 0.05:
                labels.append('')
            else:
                labels.append(f"{height:.2f}")
        
        ax.bar_label(
            bars,
            labels=labels,
            label_type='center',
            fontsize=20,
            fontweight='bold',
            color='white',
        )

    # set title label and axis labels
    ax.set_title(config_yaml['figure']['title'])
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    
    ax.set_ylabel(config_yaml['figure']['y_axis']['label'])

    # set x-ticks and x-tick labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(phase_distribution_df['method_label'])

    # set x-limits and y-limits
    ax.set_xlim(-0.5, len(phase_distribution_df['method']) - 1 + 0.5)
    ax.set_ylim(0, 1)
 

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    ax.legend(
        title=config_yaml['figure']['legend']['title'],
        bbox_to_anchor=(0.5, -0.1),
        loc='upper center',
        ncol=4,
    )

    fig.tight_layout()

    save_file_path = save_dir_path / f"phase_distribution.png"
    fig.savefig(save_file_path)

    return


if __name__ == "__main__":
    main()
