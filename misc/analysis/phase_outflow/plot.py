import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

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
    'all': list(range(1, NUM_PHASES + 1)),
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

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'phase_outflow'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # get outflow_df
    outflow_df = getOutflowDf(config_yaml, save_dir_path)

    # plot figure
    plotFigure(config_yaml, outflow_df, save_dir_path)
    return

def getOutflowDf(config_yaml, save_dir_path):
    outflow_map_list = []
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
                    outflow_map = {
                        'id': len(outflow_map_list) + 1,
                        'method': config_yaml['figure']['x_axis']['label']['mpc'][f"{num_phases}-phase"],
                        'layout': layout,
                        'inflow': inflow,
                        'intersection': re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1),
                    }
                    
                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')

                    for phase_type in [1, 2, 3, 'others', 'all']:
                        phase_list = TYPE_PHASE_MAP[phase_type]
                        
                        tmp_time_series_df = time_series_df[time_series_df['phase'].isin(phase_list)]
                        if phase_type == 'others':
                            outflow_map['outflow_others'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() 
                            outflow_map['phase_others'] = tmp_time_series_df.shape[0] * config_yaml['time_step']
                            outflow_map['outflow_rate_others'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['time_step'])
                        elif phase_type == 'all':
                            outflow_map['outflow_all'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() 
                            outflow_map['phase_all'] = tmp_time_series_df.shape[0] * config_yaml['time_step']
                            outflow_map['outflow_rate_all'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['time_step'])
                        else:
                            outflow_map[f"outflow_type_{phase_type}"] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() 
                            outflow_map[f"phase_type_{phase_type}"] = tmp_time_series_df.shape[0] * config_yaml['time_step']
                            outflow_map[f"outflow_rate_type_{phase_type}"] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['time_step'])

                    outflow_map_list.append(outflow_map)

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
                    outflow_map = {
                        'id': len(outflow_map_list) + 1,
                        'method': config_yaml['figure']['x_axis']['label']['scoot'],
                        'layout': layout,
                        'inflow': inflow,
                        'intersection': re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)
                    }

                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')
                    
                    for phase_type in [1, 2, 3, 'others', 'all']:
                        phase_list = TYPE_PHASE_MAP[phase_type]

                        tmp_time_series_df = time_series_df[time_series_df['phase'].isin(phase_list)]
                        if phase_type == 'others':
                            outflow_map['outflow_others'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() 
                            outflow_map['phase_others'] = tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step']
                            outflow_map['outflow_rate_others'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step'])
                        elif phase_type == 'all':
                            outflow_map['outflow_all'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() 
                            outflow_map['phase_all'] = tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step']
                            outflow_map['outflow_rate_all'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step'])
                        else:
                            outflow_map[f"outflow_type_{phase_type}"] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() 
                            outflow_map[f"phase_type_{phase_type}"] = tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step']
                            outflow_map[f"outflow_rate_type_{phase_type}"] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step'])

                    outflow_map_list.append(outflow_map)

            # regarding drl
            drl_dir_path = simulator_dir_path / 'drl'
            for method_dir_path in drl_dir_path.glob('config_*'):
                with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                    method_config = yaml.safe_load(f)

                vehicle_state_info = method_config['state']['vehicle']

                if all(not vehicle_state_info[key] for key in ['position', 'speed', 'route']):
                    if not config_yaml['target']['control_method']['drl']['macro']: continue

                    method = config_yaml['figure']['x_axis']['label']['drl']['macro']
                
                elif all(vehicle_state_info[key] for key in ['position', 'speed', 'route']) and method_config['num_phases'] == 4:
                    if not config_yaml['target']['control_method']['drl']['4-phase']: continue

                    method = config_yaml['figure']['x_axis']['label']['drl']['4-phase']
                
                elif all(vehicle_state_info[key] for key in ['position', 'speed', 'route']) and method_config['num_phases'] == 17:
                    if not config_yaml['target']['control_method']['drl']['proposed']: continue

                    method = config_yaml['figure']['x_axis']['label']['drl']['proposed']
                
                else:
                    continue
                
                del method_config['num_phases']
                del vehicle_state_info['position'], vehicle_state_info['speed'], vehicle_state_info['route']

                if method_config != config_yaml['drl']:
                    continue
                
                for intersection_dir_path in method_dir_path.glob('intersection_*'):
                    outflow_map = {
                        'id': len(outflow_map_list) + 1,
                        'method': method,
                        'layout': layout,
                        'inflow': inflow,
                        'intersection': re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1)
                    }

                    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
                        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')

                    for phase_type in [1, 2, 3, 'others', 'all']:
                        phase_list = TYPE_PHASE_MAP[phase_type]
                        
                        tmp_time_series_df = time_series_df[time_series_df['phase'].isin(phase_list)]

                        if phase_type == 'others':
                            outflow_map['outflow_others'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() 
                            outflow_map['phase_others'] = tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step']
                            outflow_map['outflow_rate_others'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step'])
                        elif phase_type == 'all':
                            outflow_map['outflow_all'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() 
                            outflow_map['phase_all'] = tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step']
                            outflow_map['outflow_rate_all'] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step'])
                        else:
                            outflow_map[f"outflow_type_{phase_type}"] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() 
                            outflow_map[f"phase_type_{phase_type}"] = tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step']
                            outflow_map[f"outflow_rate_type_{phase_type}"] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step'])

                    outflow_map_list.append(outflow_map)

    
    outflow_columns = [f"outflow_type_{phase_type}" for phase_type in [1, 2, 3]] + ['outflow_others', 'outflow_all']
    phase_columns = [f"phase_type_{phase_type}" for phase_type in [1, 2, 3]] + ['phase_others', 'phase_all']
    outflow_rate_columns = [f"outflow_rate_type_{phase_type}" for phase_type in [1, 2, 3]] + ['outflow_rate_others', 'outflow_rate_all']
    outflow_df = pd.DataFrame(
        outflow_map_list, 
        columns=['id', 'method', 'layout', 'inflow', 'intersection'] + outflow_columns + phase_columns + outflow_rate_columns
    )


    # sort by method, layout, inflow, intersection
    for param in ['method', 'layout', 'inflow']:
        order_list = getOrderList(config_yaml, param)
        outflow_df[param] = pd.Categorical(outflow_df[param], categories=order_list, ordered=True)
    outflow_df = outflow_df.sort_values(by=['method', 'layout', 'inflow', 'intersection']).reset_index(drop=True)
    
    outflow_df['id'] = range(1, outflow_df.shape[0] + 1)
    return outflow_df

def getOrderList(config_yaml, param):
    order_list = []
    if param == 'method':
        if config_yaml['target']['control_method']['scoot']:
            order_list.append(config_yaml['figure']['x_axis']['label']['scoot'])
        for method, flg in config_yaml['target']['control_method']['mpc'].items():
            if not flg: continue
            order_list.append(config_yaml['figure']['x_axis']['label']['mpc'][method])
        for method, flg in config_yaml['target']['control_method']['drl'].items():
            if not flg: continue
            order_list.append(config_yaml['figure']['x_axis']['label']['drl'][method])
        return order_list
    
    elif param == 'layout':
        return config_yaml['target']['layout']
    
    elif param == 'inflow':
        return config_yaml['target']['inflow']
    
    else:
        raise NotImplementedError(f"Not implemented param: {param}")

def getColorMap(hue_list, config_yaml):
    color_list = sns.color_palette(config_yaml['figure']['color_palette'], n_colors=len(hue_list))
    color_map = {hue: color_list[i] for i, hue in enumerate(hue_list)}
    return color_map

def plotFigure(config_yaml, outflow_df, save_dir_path):
    # get bad_id_list_map
    if config_yaml['target']['control_method']['drl']['4-phase'] and config_yaml['target']['control_method']['drl']['proposed']:
        for layout in config_yaml['target']['layout']:
            for inflow in config_yaml['target']['inflow']:
                target_outflow_df = outflow_df[
                    (outflow_df['layout'] == layout) &
                    (outflow_df['inflow'] == inflow)
                ]
                bad_id_list_map = {
                    '4-phase': [],
                    'proposed': []
                }
                grouped_outflow_df = target_outflow_df.groupby(['layout', 'inflow', 'intersection'])
                for _, tmp_outflow_df in grouped_outflow_df:
                    proposed_drl_row = tmp_outflow_df[
                        tmp_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['proposed']
                    ]
                    action_ablation_drl_row = tmp_outflow_df[
                        tmp_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['4-phase']
                    ]

                    if proposed_drl_row['outflow_rate_all'].values[0] < action_ablation_drl_row['outflow_rate_all'].values[0]:
                        bad_id_list_map['proposed'].append(proposed_drl_row['id'].values[0])
                        bad_id_list_map['4-phase'].append(action_ablation_drl_row['id'].values[0])
                
                
                plot_outflow_map_list = []
                for method in ['4-phase', 'proposed']:
                    plot_outflow_map = {
                        'id': len(plot_outflow_map_list) + 1,
                        'method': f"{method}-bad",
                    }
                    tmp_outflow_df = target_outflow_df[
                        (target_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl'][method]) &
                        (target_outflow_df['id'].isin(bad_id_list_map[method]))
                    ]

                    for phase_type in [1, 2, 3, 'others', 'all']:
                        if phase_type == 'others':
                            outflow = tmp_outflow_df['outflow_others'].sum()
                            phase = tmp_outflow_df['phase_others'].sum()
                            plot_outflow_map['others'] = outflow / phase if phase > 0 else None
                        elif phase_type == 'all':
                            outflow = tmp_outflow_df['outflow_all'].sum()
                            phase = tmp_outflow_df['phase_all'].sum()
                            plot_outflow_map['all'] = outflow / phase if phase > 0 else None
                        else:
                            outflow = tmp_outflow_df[f"outflow_type_{phase_type}"].sum()
                            phase = tmp_outflow_df[f"phase_type_{phase_type}"].sum()
                            plot_outflow_map[f"type_{phase_type}"] = outflow / phase if phase > 0 else None
                    
                    plot_outflow_map_list.append(plot_outflow_map)
                
                for method in ['4-phase', 'proposed']:
                    plot_outflow_map = {
                        'id': len(plot_outflow_map_list) + 1,
                        'method': f"{method}-good",
                    }
                    tmp_outflow_df = target_outflow_df[
                        (target_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl'][method]) &
                        (~target_outflow_df['id'].isin(bad_id_list_map[method]))
                    ]

                    for phase_type in [1, 2, 3, 'others', 'all']:
                        if phase_type == 'others':
                            outflow = tmp_outflow_df['outflow_others'].sum()
                            phase = tmp_outflow_df['phase_others'].sum()
                            plot_outflow_map['others'] = outflow / phase if phase > 0 else None
                        elif phase_type == 'all':
                            outflow = tmp_outflow_df['outflow_all'].sum()
                            phase = tmp_outflow_df['phase_all'].sum()
                            plot_outflow_map['all'] = outflow / phase if phase > 0 else None
                        else:
                            outflow = tmp_outflow_df[f"outflow_type_{phase_type}"].sum()
                            phase = tmp_outflow_df[f"phase_type_{phase_type}"].sum()
                            plot_outflow_map[f"type_{phase_type}"] = outflow / phase if phase > 0 else None
                    
                    plot_outflow_map_list.append(plot_outflow_map)
                
                inflow_rate_columns = [f"type_{phase_type}" for phase_type in [1, 2, 3]] + ['others', 'all']
                plot_outflow_df = pd.DataFrame(
                    plot_outflow_map_list, 
                    columns=['id', 'method'] + inflow_rate_columns
                )

                plot_outflow_df = plot_outflow_df.melt(
                    id_vars=['id', 'method'],
                    value_vars=inflow_rate_columns,
                    var_name='phase_type',
                    value_name='outflow_rate',
                )
                plot_outflow_df['outflow_rate'] = plot_outflow_df['outflow_rate'].fillna(0)
                plot_outflow_df = plot_outflow_df[plot_outflow_df['phase_type'] != 'all']
                plot_outflow_df = plot_outflow_df[plot_outflow_df['method'].isin(['proposed-good', 'proposed-bad'])]

                fig, ax = plt.subplots()

                sns.barplot(
                    data=plot_outflow_df,
                    x='method',
                    y='outflow_rate',
                    hue='phase_type',
                    palette=getColorMap(plot_outflow_df['phase_type'].unique(), config_yaml),
                    ax=ax
                )

                fig.tight_layout()
                fig.savefig(save_dir_path / f"outflow_rate_{layout}_{inflow}.png")
                plt.close(fig)

    # all in one figure
    bad_id_list_map = {
        '4-phase': [],
        'proposed': []
    }
    for _, target_outflow_df in outflow_df.groupby(['layout', 'inflow', 'intersection']):
        proposed_drl_row = target_outflow_df[
            target_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['proposed']
        ]
        action_ablation_drl_row = target_outflow_df[
            target_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['4-phase']
        ]

        if proposed_drl_row['outflow_rate_all'].values[0] < action_ablation_drl_row['outflow_rate_all'].values[0]:
            bad_id_list_map['proposed'].append(proposed_drl_row['id'].values[0])
            bad_id_list_map['4-phase'].append(action_ablation_drl_row['id'].values[0])
    
    outflow_map_list = []
    for method in ['4-phase', 'proposed']:
        method_outflow_df = outflow_df[outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl'][method]]
        for type in ['good', 'bad']:
            outflow_map = {
                'id': len(outflow_map_list) + 1,
                'method': f"{method}-{type}"
            }

            if type == 'good':
                target_outflow_df = method_outflow_df[~method_outflow_df['id'].isin(bad_id_list_map[method])]
            else:
                target_outflow_df = method_outflow_df[method_outflow_df['id'].isin(bad_id_list_map[method])]

            for phase_type in ['type_1', 'type_2', 'type_3', 'others', 'all']:
                outflow = target_outflow_df[f"outflow_{phase_type}"].sum()
                phase = target_outflow_df[f"phase_{phase_type}"].sum()
                outflow_map[phase_type] = outflow / phase if phase > 0 else None
            
            outflow_map_list.append(outflow_map)
        
    outflow_rate_columns = ['type_1', 'type_2', 'type_3', 'others', 'all']
    plot_outflow_df = pd.DataFrame(
        outflow_map_list, 
        columns=['id', 'method'] + outflow_rate_columns
    )
    
    plot_outflow_df = plot_outflow_df[plot_outflow_df['method'].isin(['proposed-good', 'proposed-bad'])]

    plot_outflow_df = plot_outflow_df.melt(
        id_vars=['id', 'method'],
        value_vars=outflow_rate_columns,
        var_name='phase_type',
        value_name='outflow_rate',
    )
    plot_outflow_df = plot_outflow_df[plot_outflow_df['phase_type'] != 'all']
    plot_outflow_df['outflow_rate'] = plot_outflow_df['outflow_rate'].fillna(0)

    fig, ax = plt.subplots()

    sns.barplot(
        data=plot_outflow_df,
        x='method',
        y='outflow_rate',
        hue='phase_type',
        palette=getColorMap(plot_outflow_df['phase_type'].unique(), config_yaml),
        ax=ax
    )
    fig.tight_layout()
    fig.savefig(save_dir_path / "outflow_rate_all.png")
    plt.close(fig)




    return


if __name__ == "__main__":
    main()