import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import copy
import re

from libs.figure_config import initFigureConfig


# configuration
initFigureConfig()
config_file_path = root_dir_path / 'misc' / 'analysis' / 'performance_metric_bars' / 'config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set wild_card_keys
if config_yaml['figure']['wild_card_type'] == 'num_phases':
    wild_card_keys = ['mpc', 'phases', '4-road']
else:
    raise NotImplementedError(f"Not supported wild card type: {config_yaml['figure']['wild_card_type']}")


# set data_dir_path
data_dir_path = root_dir_path / 'data'

# set performance_metrics_dir_path
performance_metrics_dir_path = data_dir_path / 'performance_metrics'

# set target_dir_path_map
def getSubSetFlg(sub_config, main_config, wild_card_key_list=None, key_list=[]):
    for sub_key, sub_value in sub_config.items():
        tmp_keys = copy.deepcopy(key_list)
        tmp_keys.append(sub_key)

        main_value = main_config[sub_key]

        if isinstance(sub_value, dict):
            if getSubSetFlg(sub_value, main_value, wild_card_key_list, tmp_keys):
                continue
            else:
                return False

        if wild_card_key_list is not None and tmp_keys == wild_card_key_list:
            continue

        if sub_value != main_config[sub_key]:
            return False
        
    return True

# set target_dir_paths_map
target_dir_paths_map = {}
for simulator_dir_path in performance_metrics_dir_path.rglob('simulator_*'):
    keys = (
        simulator_dir_path.parts[-3], # layout_name
        simulator_dir_path.parts[-2], # inflow_name
        int(re.match(rf"simulator_(\d+)", simulator_dir_path.parts[-1]).group(1)), # simulator_id
    )
    path_info = {}
    match_config_count = 0
    for control_method in ['mpc', 'scoot']:
        # skip if flg is False
        if not config_yaml['figure']['plot_flg'][control_method]:
            continue
        
        # set wild_card_exist_flg
        wild_card_exist_flg = wild_card_keys is not None and wild_card_keys[0] == control_method
        
        # update path_info
        path_info[control_method] = {} if wild_card_exist_flg else None
        control_method_dir_path = simulator_dir_path / control_method
        for config_dir_path in control_method_dir_path.glob('config_*'):
            # set main_config_yaml and sub_config_yaml
            main_config_yaml = config_yaml[control_method]
            with open(config_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                sub_config_yaml = yaml.safe_load(f)
            
            # check if it is sub set or not
            if wild_card_exist_flg:
                sub_set_flg = getSubSetFlg(sub_config_yaml, main_config_yaml, wild_card_keys[1:])
            else:
                sub_set_flg = getSubSetFlg(sub_config_yaml, main_config_yaml)

            # skip if it is not sub set
            if not sub_set_flg:
                continue

            # increment match_config_count
            match_config_count += 1
            
            # push to path_info
            if wild_card_exist_flg:
                # set wild_card_value
                wild_card_value = sub_config_yaml
                for wild_card_key in wild_card_keys[1:]:
                    wild_card_value = wild_card_value[wild_card_key]

                path_info[control_method][wild_card_value] = config_dir_path
            else:
                path_info[control_method] = config_dir_path  

    # skip if no matched configuration
    if match_config_count == 0:
        continue

    target_dir_paths_map[keys] = path_info      

# set analysis_dir_path
analysis_dir_path = data_dir_path / 'analysis'

# set layout_dir_path
layout_dir_path = root_dir_path / 'layout'

# set bar_graph_df_map
bar_graph_df_map = {}
performance_metric = config_yaml['figure']['performance_metric']
for keys, path_info in target_dir_paths_map.items():
    # set bar_graph_data
    bar_graph_data = {
        'intersection_id': [],
        'intersection_name': [],
    }

    # set intersection_name_map
    layout_name = keys[0]
    with open(layout_dir_path / layout_name / 'intersections.csv', 'r', encoding='utf-8') as f:
        intersections_df = pd.read_csv(f)
    intersection_name_map = {}
    for _, intersection_row in intersections_df.iterrows():
        if 'name' in intersection_row:
            intersection_name_map[int(intersection_row['id'])] = intersection_row['name']
        else:
            intersection_name_map[int(intersection_row['id'])] = f"ID:{int(intersection_row['id'])}"
    
    # set num_intersections
    num_intersections = len(intersection_name_map)

    # updata bar_graph_data
    bar_graph_data['intersection_id'] = list(range(1, num_intersections + 1))
    intersection_name_list = [''] * num_intersections
    for intersection_id in range(1, num_intersections + 1):
        intersection_name_list[intersection_id - 1] = intersection_name_map[intersection_id]
    bar_graph_data['intersection_name'] = intersection_name_list

    # set performance_metric_list and push to bar_graph_data
    for control_method, config_paths in path_info.items():
        if isinstance(config_paths, dict):
            for wild_card_value, config_path in config_paths.items():           
                performance_metric_list = [0] * num_intersections
                for intersection_dir_path in config_path.glob('intersection_*'):
                    intersection_id = int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1))
                    with open(intersection_dir_path / f"{performance_metric}.csv", 'r', encoding='utf-8') as f:
                        performance_metric_df = pd.read_csv(f)
                    
                    if performance_metric == 'average_queue':
                        performance_metric_list[intersection_id - 1] = float(performance_metric_df['queue_length'].mean())
                    elif performance_metric == 'max_queue':
                        performance_metric_list[intersection_id - 1] = float(performance_metric_df['queue_length'].mean())
                    elif performance_metric == 'average_delay':
                        performance_metric_list[intersection_id - 1] = float(performance_metric_df['delay'].mean())
                    elif performance_metric == 'max_delay':
                        performance_metric_list[intersection_id - 1] = float(performance_metric_df['delay'].mean())
                    elif performance_metric == 'calc_time':
                        performance_metric_list[intersection_id - 1] = float(performance_metric_df['calculation_time'].mean())
                    else:
                        raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                if config_yaml['figure']['wild_card_type'] == 'num_phases':
                    bar_graph_data[f"{wild_card_value}-phase MPC"] = performance_metric_list
                else:
                    raise NotImplementedError(f"Not supported wild card type: {config_yaml['figure']['wild_card_type']}")
        else:
            config_path = config_paths
            if config_path is None:
                continue

            performance_metric_list = [0] * num_intersections
            for intersection_dir_path in config_path.glob('intersection_*'):
                intersection_id = int(re.match(rf"intersection_(\d+)", intersection_dir_path.name).group(1))
                with open(intersection_dir_path / f"{performance_metric}.csv", 'r', encoding='utf-8') as f:
                    performance_metric_df = pd.read_csv(f)
                
                if performance_metric == 'average_queue':
                    performance_metric_list[intersection_id - 1] = float(performance_metric_df['queue_length'].mean())
                elif performance_metric == 'max_queue':
                    performance_metric_list[intersection_id - 1] = float(performance_metric_df['queue_length'].mean())
                elif performance_metric == 'average_delay':
                    performance_metric_list[intersection_id - 1] = float(performance_metric_df['delay'].mean())
                elif performance_metric == 'max_delay':
                    performance_metric_list[intersection_id - 1] = float(performance_metric_df['delay'].mean())
                elif performance_metric == 'calc_time':
                    performance_metric_list[intersection_id - 1] = float(performance_metric_df['calculation_time'].mean())
                else:
                    raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
            
            if config_yaml['figure']['wild_card_type'] == 'num_phases':
                bar_graph_data[control_method.upper()] = performance_metric_list
    
    # convert to DataFrame
    bar_graph_df_map[keys] = pd.DataFrame(bar_graph_data)

# make figures
for keys, bar_graph_df in bar_graph_df_map.items():
    # set save_dir_path
    save_dir_path = analysis_dir_path
    save_dir_path /= keys[0]  # layout_name
    save_dir_path /= keys[1]  # inflow_name
    save_dir_path /= f"simulator_{keys[2]}"  # simulator_id
    save_dir_path /= 'performance_metric_bars'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # set figure and axis
    fig, ax = plt.subplots()

    if config_yaml['figure']['wild_card_type'] == 'num_phases':
        legend_title = 'Control Method'
        y_axis_label = performance_metric.replace('_', ' ').title()
    else:
        raise NotImplementedError(f"Not supported wild card type: {config_yaml['figure']['wild_card_type']}")
    
    plot_df = bar_graph_df.melt(
        id_vars=['intersection_name'],
        value_vars=[col for col in bar_graph_df.columns if col not in ['intersection_id', 'intersection_name']],
        var_name=legend_title,
        value_name=y_axis_label,
    )

    if config_yaml['figure']['wild_card_type'] == 'num_phases':
        method_order_list = ['SCOOT', '4-phase MPC', '8-phase MPC', '17-phase MPC']
        for method in copy.deepcopy(method_order_list):
            if method in bar_graph_df.columns:
                continue
            method_order_list.remove(method)
    else:
        raise NotImplementedError(f"Not supported wild card type: {config_yaml['figure']['wild_card_type']}")
    ax = sns.barplot(
        data=plot_df,
        x='intersection_name',
        y=performance_metric.replace('_', ' ').title(),
        hue=legend_title,
        hue_order=method_order_list,
    )

    ax.set_title('Performance Metric Bars')
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel('Intersection Name')
    ax.legend(title=legend_title, loc='best')
    
    fig.tight_layout()
    fig.savefig(save_dir_path / f"{performance_metric}_bars.png")
    






            