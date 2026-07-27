import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.container import BarContainer

from libs.figure_config import init_figure_config

NUM_PHASES = 17

TYPE_PHASE_MAP = {
    'type_1': [1, 2],
    'type_2': [3, 4],
    'type_3': [5, 6, 7, 8],
    'type_4': [9, 10, 11, 12],
    'type_5': [13, 14, 15, 16],
    'type_6': [17],
    'others': [9, 10, 11, 12, 13, 14, 15, 16, 17],
    'all': list(range(1, NUM_PHASES + 1)),
}

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

    # get phase_outflow_df
    phase_outflow_df = getPhaseOutflowDf(config_yaml, save_dir_path)

    # plot figure
    plotFigure(config_yaml, phase_outflow_df, save_dir_path)
    return

def getPhaseOutflowMap(id, layout, method, inflow, intersection_dir_path, config_yaml):
    phase_outflow_map = {
        'id': id,
        'method': method,
        'layout': layout,
        'inflow': inflow,
        'intersection': int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1))
    }

    with open(intersection_dir_path / 'performance_metrics.csv', 'r', encoding='utf-8') as f:
        time_series_df = pd.read_csv(intersection_dir_path / 'performance_metrics.csv')

    # queue, delay, speed
    phase_outflow_map['queue_avg'] = time_series_df['queue_avg'].mean()
    phase_outflow_map['delay_avg'] = time_series_df[f"delay_avg_{config_yaml['target']['delay_type']}"].mean()
    phase_outflow_map['speed_avg'] = time_series_df['speed_avg'].mean()
    
    for phase_type in ['type_1', 'type_2', 'type_3', 'others', 'all']:
        phase_list = TYPE_PHASE_MAP[phase_type]

        tmp_time_series_df = time_series_df[time_series_df['phase'].isin(phase_list)]
        tmp_time_series_df = tmp_time_series_df[tmp_time_series_df['time'] > config_yaml['target']['start_time']] # remove the first 100 seconds to avoid the effect of initial condition
        phase_outflow_map[f"outflow_{phase_type}"] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum()
        phase_outflow_map[f"phase_{phase_type}"] = tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step'] / 3600
        phase_outflow_map[f"outflow_rate_{phase_type}"] = None if tmp_time_series_df.shape[0] == 0 else tmp_time_series_df['outflow'].sum() / (tmp_time_series_df.shape[0] * config_yaml['simulator']['time_step'])
    
    return phase_outflow_map

def getPhaseOutflowDf(config_yaml, save_dir_path):
    phase_outflow_map_list = []
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

            # regarding drl
            drl_dir_path = simulator_dir_path / 'drl'
            for method_dir_path in drl_dir_path.glob('config_*'):
                with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                    method_config = yaml.safe_load(f)

                vehicle_state_info = method_config['state']['vehicle']
                
                if all(vehicle_state_info[key] for key in ['position', 'speed', 'route']) and method_config['num_phases'] == 4:
                    method = config_yaml['figure']['x_axis']['label']['drl']['4-phase']          
                elif all(vehicle_state_info[key] for key in ['position', 'speed', 'route']) and method_config['num_phases'] == 17:
                    method = config_yaml['figure']['x_axis']['label']['drl']['proposed']
                else:
                    continue
                
                del method_config['num_phases']
                del vehicle_state_info['position'], vehicle_state_info['speed'], vehicle_state_info['route']

                if method_config != config_yaml['drl']:
                    continue
                
                for intersection_dir_path in method_dir_path.glob('intersection_*'):
                    phase_outflow_map_list.append(getPhaseOutflowMap(
                        id=len(phase_outflow_map_list) + 1,
                        method=method,
                        layout=layout,
                        inflow=inflow,
                        intersection_dir_path=intersection_dir_path,
                        config_yaml=config_yaml,
                    ))

    outflow_columns = [f"outflow_type_{phase_type}" for phase_type in [1, 2, 3]] + ['outflow_others', 'outflow_all']
    phase_columns = [f"phase_type_{phase_type}" for phase_type in [1, 2, 3]] + ['phase_others', 'phase_all']
    outflow_rate_columns = [f"outflow_rate_type_{phase_type}" for phase_type in [1, 2, 3]] + ['outflow_rate_others', 'outflow_rate_all']
    performance_columns = ['queue_avg', 'delay_avg', 'speed_avg']
    phase_outflow_df = pd.DataFrame(
        phase_outflow_map_list, 
        columns=['id', 'method', 'layout', 'inflow', 'intersection'] + outflow_columns + phase_columns + outflow_rate_columns + performance_columns
    )

    # sort by method, layout, inflow, intersection
    for param in ['method', 'layout', 'inflow']:
        order_list = getOrderList(config_yaml, param)
        phase_outflow_df[param] = pd.Categorical(phase_outflow_df[param], categories=order_list, ordered=True)
    phase_outflow_df = phase_outflow_df.sort_values(by=['method', 'layout', 'inflow', 'intersection']).reset_index(drop=True)
    
    phase_outflow_df['id'] = range(1, phase_outflow_df.shape[0] + 1)

    # validate outflow_df
    if phase_outflow_df[phase_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['proposed']].shape[0] != phase_outflow_df[phase_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['4-phase']].shape[0]:
        raise ValueError(f"The number of scenarios for proposed DRL and 4-phase DRL are not equal: proposed = {phase_outflow_df[phase_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['proposed']].shape[0]}, 4-phase = {phase_outflow_df[phase_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['4-phase']].shape[0]}")
    
    return phase_outflow_df

def getOrderList(config_yaml, param):
    if param == 'method':
        return [config_yaml['figure']['x_axis']['label']['drl']['4-phase'], config_yaml['figure']['x_axis']['label']['drl']['proposed']]
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

def getWorseListMap(config_yaml, phase_outflow_df):
    worse_list_map = {'4-phase': [], 'proposed': []}
    for _, tmp_outflow_df in phase_outflow_df.groupby(['layout', 'inflow', 'intersection'], observed=True):
        proposed_drl_row = tmp_outflow_df[tmp_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['proposed']].iloc[0]
        action_ablation_drl_row = tmp_outflow_df[tmp_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl']['4-phase']].iloc[0]

        proposed_metric = proposed_drl_row[config_yaml['target']['performance_metric']]
        action_ablation_metric = action_ablation_drl_row[config_yaml['target']['performance_metric']]

        if config_yaml['target']['worse_definition']['absolute']['flg']:
            if config_yaml['target']['performance_metric'] in ['queue_avg', 'delay_avg']:
                if proposed_metric < config_yaml['target']['worse_definition']['absolute']['value']: continue
            elif config_yaml['target']['performance_metric'] in ['speed_avg']:
                if proposed_metric > config_yaml['target']['worse_definition']['absolute']['value']: continue


        if config_yaml['target']['performance_metric'] in ['speed_avg']:
            worse_rate = (action_ablation_metric - proposed_metric) / action_ablation_metric * 100
        elif config_yaml['target']['performance_metric'] in ['queue_avg', 'delay_avg']:
            worse_rate = (proposed_metric - action_ablation_metric) / action_ablation_metric * 100
        else:
            raise NotImplementedError(f"Not implemented performance metric: {config_yaml['target']['performance_metric']}")
        
        threshold = config_yaml['target']['worse_definition']['rate']['value'] if config_yaml['target']['worse_definition']['rate']['flg'] else 0
        if worse_rate > threshold:
            worse_list_map['proposed'].append(proposed_drl_row['id'])
            worse_list_map['4-phase'].append(action_ablation_drl_row['id'])
            print(f"worse case: layout = {proposed_drl_row['layout']}, inflow = {proposed_drl_row['inflow']}, intersection = {proposed_drl_row['intersection']}, worse_rate={worse_rate:.2f}%, 4-phase DRL = {action_ablation_metric:.2f}, proposed DRL = {proposed_metric:.2f}")
        
    return worse_list_map

def plotFigure(config_yaml, phase_outflow_df, save_dir_path):
    worse_list_map = getWorseListMap(config_yaml, phase_outflow_df)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # upper subplot
    plotSubplot(
        config_yaml=config_yaml,
        phase_outflow_df=phase_outflow_df,
        worse_list_map=worse_list_map,
        ax=axes[0],
        metric=config_yaml['target']['type']['upper']['metric'],
        group=config_yaml['target']['type']['upper']['group'],
        pos='upper',
    )

    # lower subplot
    plotSubplot(
        config_yaml=config_yaml,
        phase_outflow_df=phase_outflow_df,
        worse_list_map=worse_list_map,
        ax=axes[1],
        metric=config_yaml['target']['type']['lower']['metric'],
        group=config_yaml['target']['type']['lower']['group'],
        pos='lower',
    )

    fig.tight_layout()
    fig.savefig(save_dir_path / f"outflow_rate.png")
    plt.close(fig)
    return

def getPhaseCategoryList(config_yaml, metric):
    phase_category_list = []
    for phase_category, flg in config_yaml['target']['phase_category'].items():
        if not flg: continue
        if phase_category == 'all' and metric == 'phase': continue
        phase_category_list.append(phase_category)
    return phase_category_list

def plotSubplot(config_yaml, phase_outflow_df, worse_list_map, ax, metric, group, pos):
    plot_map_list = []

    phase_category_list = getPhaseCategoryList(config_yaml, metric)
    for method in ['4-phase', 'proposed']:
        plot_map = {
            'id': len(plot_map_list) + 1,
            'method': config_yaml['figure']['x_axis']['label']['drl'][method],
        }

        method_outflow_df = phase_outflow_df[phase_outflow_df['method'] == config_yaml['figure']['x_axis']['label']['drl'][method]]

        if group == 'worse':
            tmp_outflow_df = method_outflow_df[
                (method_outflow_df['id'].isin(worse_list_map[method]))
            ]
        elif group == 'specific':
            tmp_outflow_df = method_outflow_df[
                (method_outflow_df['layout'] == config_yaml['target']['specific']['layout']) &
                (method_outflow_df['inflow'] == config_yaml['target']['specific']['inflow']) &
                (method_outflow_df['intersection'] == config_yaml['target']['specific']['intersection'])
            ]
        elif group == 'not_worse':
            tmp_outflow_df = method_outflow_df[
                (~method_outflow_df['id'].isin(worse_list_map[method]))
            ]
        elif group == 'others':
            tmp_outflow_df = method_outflow_df[
                ~(
                    (method_outflow_df['layout'] == config_yaml['target']['specific']['layout']) &
                    (method_outflow_df['inflow'] == config_yaml['target']['specific']['inflow']) &
                    (method_outflow_df['intersection'] == config_yaml['target']['specific']['intersection'])
                )
            ]
        elif group == 'all':
            tmp_outflow_df = method_outflow_df
        else:
            raise NotImplementedError(f"Not implemented group: {group}")
        
        if metric == 'outflow_rate':
            for phase_category in phase_category_list:
                outflow = tmp_outflow_df[f"outflow_{phase_category}"].sum()
                phase = tmp_outflow_df[f"phase_{phase_category}"].sum()
                plot_map[phase_category] = None if (tmp_outflow_df.shape[0] == 0 or phase == 0) else outflow / phase
        elif metric == 'phase':
            for phase_category in phase_category_list:
                plot_map[phase_category] = tmp_outflow_df[f"phase_{phase_category}"].sum()
            
            sum_phase = sum([plot_map[phase_category] for phase_category in phase_category_list])
            for phase_category in phase_category_list:
                plot_map[phase_category] = None if plot_map[phase_category] == 0 else plot_map[phase_category] / sum_phase
        else:
            raise NotImplementedError(f"Not implemented metric: {metric}")
        
        plot_map_list.append(plot_map)

    plot_df = pd.DataFrame(
        plot_map_list, 
        columns=['id', 'method'] + phase_category_list
    )
    
    if metric == 'outflow_rate':
        x_pos_map = {
            config_yaml['figure']['x_axis']['label']['drl']['4-phase']: 0,
            config_yaml['figure']['x_axis']['label']['drl']['proposed']: 1,
        }
        color_map = getColorMap(phase_category_list, config_yaml)
            
        for _, plot_row in plot_df.iterrows():
            tmp_phase_category_list = [phase_type for phase_type in phase_category_list if pd.notna(plot_row[phase_type])]
            for i, phase_type in enumerate(tmp_phase_category_list):
                offset = (i - len(tmp_phase_category_list) / 2) * config_yaml['figure']['bar']['width'][metric] + config_yaml['figure']['bar']['width'][metric] / 2
                rects = ax.bar(
                    x_pos_map[plot_row['method']] + offset, 
                    plot_row[phase_type], 
                    config_yaml['figure']['bar']['width'][metric], 
                    label=phase_type if plot_row['method'] == config_yaml['figure']['x_axis']['label']['drl']['proposed'] else "",
                    color=color_map[phase_type], 
                    edgecolor='black'
                )

                ax.bar_label(
                    rects,
                    fmt="%.0f",
                    label_type='center',
                    color='white',
                    fontsize=20,
                    fontweight='bold'
                )
    elif metric == 'phase':
        # get color_map
        color_map = getColorMap(phase_category_list, config_yaml)

        x_pos_map = {
            config_yaml['figure']['x_axis']['label']['drl']['4-phase']: 0,
            config_yaml['figure']['x_axis']['label']['drl']['proposed']: 1,
        }
        bottom_list_map = {
            config_yaml['figure']['x_axis']['label']['drl']['4-phase']: [0],
            config_yaml['figure']['x_axis']['label']['drl']['proposed']: [0],
        }

        bars_list = []
        phase_value_list = []
        for _, plot_row in plot_df.iterrows():
            for phase_category in phase_category_list:
                phase_value_list.append(plot_row[phase_category] if pd.notna(plot_row[phase_category]) else 0)
                bars = ax.bar(
                    x=x_pos_map[plot_row['method']],
                    height= phase_value_list[-1],
                    bottom=bottom_list_map[plot_row['method']][-1],
                    width=config_yaml['figure']['bar']['width'][metric],
                    color=color_map[phase_category],
                    edgecolor='black',
                )
                bars_list.append(bars[0])
    
                bottom_list_map[plot_row['method']].append(
                    bottom_list_map[plot_row['method']][-1] + phase_value_list[-1]
                )
        
        bars = BarContainer(bars_list, datavalues=phase_value_list)
        
        # add dashed lines between bars
        action_bottom_list = bottom_list_map[config_yaml['figure']['x_axis']['label']['drl']['4-phase']]
        proposed_bottom_list = bottom_list_map[config_yaml['figure']['x_axis']['label']['drl']['proposed']]
        x1 = x_pos_map[config_yaml['figure']['x_axis']['label']['drl']['4-phase']] + config_yaml['figure']['bar']['width'][metric] / 2
        x2 = x_pos_map[config_yaml['figure']['x_axis']['label']['drl']['proposed']] - config_yaml['figure']['bar']['width'][metric] / 2
        for y1, y2 in zip(action_bottom_list, proposed_bottom_list):
            if y1 == y2 == 0: continue

            ax.plot(
                [x1, x2],
                [y1, y2],
                linestyle='--',
                color='gray',
            )

        # set values to bars
        label_list = []
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:
                label_list.append(f"{height:.2f}")
            else:
                label_list.append('')
        
        ax.bar_label(
            bars,
            labels=label_list,
            label_type='center',
            color='white',
            fontsize=20,
            fontweight='bold'
        )
    else:
        raise NotImplementedError(f"Not implemented metric: {metric}")
    

    ax.set_title(config_yaml['figure']['title'][metric], fontweight='bold')
    ax.set_xticks(list(x_pos_map.values()))
    ax.set_xticklabels(list(x_pos_map.keys()) if pos == 'lower' else [''] * len(x_pos_map), fontweight='bold')
    ax.set_xlim(left=-0.5, right=1+0.5)
    ax.set_xlabel(config_yaml['figure']['x_axis']['title'] if pos == 'lower' else '', fontweight='bold')
    if metric == 'outflow_rate':
        ax.set_ylim(bottom=-config_yaml['figure']['y_axis']['lim'][metric], top=plot_df[tmp_phase_category_list].max().max() + config_yaml['figure']['y_axis']['lim'][metric])
    elif metric == 'phase':
        ax.set_ylim(bottom=-config_yaml['figure']['y_axis']['lim'][metric], top=1 + config_yaml['figure']['y_axis']['lim'][metric])
    else:
        raise NotImplementedError(f"Not implemented metric: {metric}")
    ax.set_ylabel(config_yaml['figure']['y_axis']['label'][metric], fontweight='bold')
    if pos == 'upper':
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    elif pos == 'lower':
        new_label_list = []
        handles, label_list = ax.get_legend_handles_labels()
        for label in label_list:
            new_label_list.append(config_yaml['figure']['legend']['label'][label])

        ax.legend(
            title=config_yaml['figure']['legend']['title'], 
            ncol=len(phase_category_list), 
            bbox_to_anchor=(0.5, -0.1), 
            loc='upper center',
            handles=handles,
            labels=new_label_list
        )
    else:
        raise NotImplementedError(f"Not implemented pos: {pos}")

    return


if __name__ == "__main__":
    main()