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

# set data_dir_path
data_dir_path = root_dir_path / 'data'

# set performance_metrics_dir_path
performance_metrics_dir_path = data_dir_path / 'performance_metrics'

# set target_dir_path_map
def checkCofigMatch(main_config, sub_config, wild_card_key_list=None, key_list=[]):
    for main_key, main_value in main_config.items():
        tmp_keys = copy.deepcopy(key_list)
        tmp_keys.append(main_key)

        sub_value = sub_config[main_key]

        if isinstance(main_value, dict):
            if checkCofigMatch(main_value, sub_value, wild_card_key_list, tmp_keys):
                continue
            else:
                return False

        if wild_card_key_list is not None and tmp_keys == wild_card_key_list:
            continue

        if main_value != sub_config[main_key]:
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
        
        # update path_info
        if control_method == 'mpc':
            path_info[control_method] = {}
        elif control_method == 'scoot':
            path_info[control_method] = None
        else:
            raise NotImplementedError(f"Not supported control method: {control_method}")
        
        control_method_dir_path = simulator_dir_path / control_method
        for config_dir_path in control_method_dir_path.glob('config_*'):
            # set main_config_yaml and sub_config_yaml
            main_config_yaml = config_yaml[control_method]
            with open(config_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                sub_config_yaml = yaml.safe_load(f)
            
            # check if config match
            if control_method == 'mpc':
                wild_card_key_list = ['mpc', 'phases', '4-road']
                match_flg = checkCofigMatch(main_config_yaml, sub_config_yaml, wild_card_key_list)
            elif control_method == 'scoot':
                match_flg = checkCofigMatch(main_config_yaml, sub_config_yaml)
            else:
                raise NotImplementedError(f"Not supported control method: {control_method}")

            # skip if it is not sub set
            if not match_flg:
                continue

            # increment match_config_count
            match_config_count += 1
            
            # push to path_info
            if control_method == 'mpc':
                # set num_phases
                num_phases = sub_config_yaml['phases']['4-road']

                path_info[control_method][num_phases] = config_dir_path
            elif control_method == 'scoot':
                path_info[control_method] = config_dir_path
            else:
                raise NotImplementedError(f"Not supported control method: {control_method}")

    # skip if no matched configuration
    if match_config_count == 0:
        continue

    target_dir_paths_map[keys] = path_info      

# set analysis_dir_path
analysis_dir_path = data_dir_path / 'analysis'

# set layout_dir_path
layout_dir_path = root_dir_path / 'layout'

# set bar_graph_df_map
metric_bar_graph_df_map = {}
for performance_metric in ['average_queue', 'max_queue', 'average_delay', 'max_delay', 'calc_time', 'speed', 'phases']:
    metric_bar_graph_df_map[performance_metric] = {}
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
            if control_method == 'mpc':
                for num_phases, config_path in config_paths.items():  
                    if not config_yaml['figure']['plot_flg']['mpc'][f"{num_phases}-phase"]:
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
                        elif performance_metric == 'speed':
                            performance_metric_list[intersection_id - 1] = float(performance_metric_df['value'].mean())
                        elif performance_metric == 'phases':
                            performance_metric_list[intersection_id - 1] = (performance_metric_df['phase'] != performance_metric_df['phase'].shift()).sum() - 1
                        else:
                            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                        
                    bar_graph_data[f"{num_phases}-phase MPC"] = performance_metric_list
                    
            elif control_method == 'scoot':
                config_path = config_paths
                if config_path is None:
                    continue

                if performance_metric == 'calc_time':
                    continue

                if not config_yaml['figure']['plot_flg']['scoot']:
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
                    elif performance_metric == 'speed':
                        performance_metric_list[intersection_id - 1] = float(performance_metric_df['value'].mean())
                    elif performance_metric == 'phases':
                        performance_metric_list[intersection_id - 1] = (performance_metric_df['phase'] != performance_metric_df['phase'].shift()).sum() - 1
                    else:
                        raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
                
                bar_graph_data[control_method.upper()] = performance_metric_list

            else: 
                raise NotImplementedError(f"Not supported control method: {control_method}")
        
        # convert to DataFrame
        metric_bar_graph_df_map[performance_metric][keys] = pd.DataFrame(bar_graph_data)

# make figures
for performance_metric, bar_graph_df_map in metric_bar_graph_df_map.items():
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

        # set figure_title    
        figure_title = f"{performance_metric.replace('_', ' ').title()} Comparison"

        # set x_axis_label
        if keys[0] == '7-4-1':
            x_axis_label = 'Turning Rate (Left, Straight, Right)'
        else:
            x_axis_label = 'Intersection Name'

        # set y_axis_label
        y_axis_label = performance_metric.replace('_', ' ').title()
        if performance_metric in ['average_queue', 'max_queue']:
            y_axis_label = f"{y_axis_label} [m]"
        elif performance_metric in ['average_delay', 'max_delay', 'calc_time']:
            y_axis_label = f"{y_axis_label} [s]" 
        elif performance_metric == 'speed':
            y_axis_label = f"{y_axis_label} [km/h]"
        elif performance_metric == 'phases':
            y_axis_label = "Number of Phase Changes"
        else:
            raise NotImplementedError(f"Not supported performance metric: {performance_metric}")
        
        # set legend_title
        legend_title = 'Control Method'
        
        # set plot_df
        plot_df = bar_graph_df.melt(
            id_vars=['intersection_name'],
            value_vars=[col for col in bar_graph_df.columns if col not in ['intersection_id', 'intersection_name']],
            var_name=legend_title,
            value_name=y_axis_label,
        )

        # set method_order_list
        method_order_list = ['SCOOT', '4-phase MPC', '8-phase MPC', '17-phase MPC']
        for method in copy.deepcopy(method_order_list):
            if method in bar_graph_df.columns:
                continue
            method_order_list.remove(method)
        
        # plot bar graph
        ax = sns.barplot(
            data=plot_df,
            x='intersection_name',
            y=y_axis_label,
            hue=legend_title,
            hue_order=method_order_list,
        )

        ax.set_title(figure_title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.legend(title=legend_title, loc='best')
        
        fig.tight_layout()
        fig.savefig(save_dir_path / f"{performance_metric}_bars.png")
        plt.close(fig)






            