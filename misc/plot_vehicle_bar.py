import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pickle

# configuration for compared data
simulation_dir_path_map = {
    'scoot': 'scoot/balanced_2222',
    'mpc': 'mpc/balanced_2222_10',
    'drl_v2': 'drl/apex/balanced_2222_wait_v2',
}
intersection_id = 1
num_roads = 4 # TODO: we only support 4-road intersections for now

# configuration for plot
bar_width = 0.2
same_fig_flg = True

bar_colors_map = {
    1: 'blue',
    2: 'orange',
    3: 'green',
    4: 'red',
}

# get root directory path
root_dir_path = (Path(__file__).parent / '..').resolve()

# load phase data
phase_file_path = root_dir_path / 'layout' / f"phases{num_roads}.csv"
if not phase_file_path.exists():
    raise FileNotFoundError(f"File not found: {phase_file_path}")

with open(phase_file_path, 'rb') as f:
    dtype_map = {}
    dtype_map['id'] = int
    for signal_group_order_id in range(1, num_roads + 1):
        dtype_map[f"signal_group{signal_group_order_id}"] = int
    dtype_map['random_prob'] = float
    phase_df = pd.read_csv(f, encoding='utf-8', dtype=dtype_map)

# make phase_routes_map
phase_routes_map = {}
num_direction = num_roads - 1
if num_roads == 4:
    for _, phase_row in phase_df.iterrows():
        tmp_directions = []
        for signal_group_order_id in range(1, num_roads + 1):
            # signal_group_id
            signal_group_id = int(phase_row[f"signal_group{signal_group_order_id}"])

            # road_order_id
            road_order_id = int((signal_group_id - 1) // num_direction + 1)
            
            # direction_order_id
            direction_order_id = int((signal_group_id - 1) % num_direction + 1)

            tmp_directions.append((
                signal_group_id,
                road_order_id,
                direction_order_id,
            ))
        phase_routes_map[int(phase_row['id'])] = tmp_directions

else:
    raise NotImplementedError(f"phase_direction_map not implemented for num_roads={num_roads}")

# get direction_name_map
if num_roads == 4:
    direction_name_map = {
        1: 'left',
        2: 'straight',
        3: 'right',
    } 
else:
    raise NotImplementedError(f"direction_name_map not implemented for num_roads={num_roads}")

# get road_name_map
if num_roads == 4:
    road_name_map = {
        1: 'north',
        2: 'east',
        3: 'south',
        4: 'west',
    }
else: 
    raise NotImplementedError(f"road_name_map not implemented for num_roads={num_roads}")


# get path objects
for simulation_name, simulation_str in simulation_dir_path_map.items():
    simulation_dir_path = root_dir_path / 'results' / 'metrics' / simulation_str
    if not simulation_dir_path.exists():
        raise FileNotFoundError(f"File not found: {simulation_dir_path}")
    simulation_dir_path_map[simulation_name] = simulation_dir_path

# get metric file paths
metric_file_path_map = {}
for simulation_name, simulation_dir_path in simulation_dir_path_map.items():
    metric_file_path = simulation_dir_path / f"metric_{intersection_id}.pkl"
    if not metric_file_path.exists():
        raise FileNotFoundError(f"File not found: {metric_file_path}")
    metric_file_path_map[simulation_name] = metric_file_path

# get phase_record_map
phase_record_map = {}
for simulation_name, metric_file_path in metric_file_path_map.items():
    with open(metric_file_path, 'rb') as f:
        metric_data = pickle.load(f)
    
    phase_record = metric_data['phase']
    phase_record_map[simulation_name] = phase_record

# initialize route_num_green_steps_map
route_num_green_steps_map = {}
for simulation_name in simulation_dir_path_map.keys():
    tmp_route_num_green_steps_map = {}
    for road_id in range(1, num_roads + 1):
        for direction_id in range(1, num_direction + 1):
            signal_group_id = (road_id - 1) * num_direction + direction_id
            tmp_route_num_green_steps_map[signal_group_id, road_id, direction_id] = 0
    route_num_green_steps_map[simulation_name] = tmp_route_num_green_steps_map

# make data
for simulation_name, phase_record in phase_record_map.items():
    for phase_id in phase_record:
        routes_list = phase_routes_map[phase_id]

        for route in routes_list:
            route_num_green_steps_map[simulation_name][route] += 1

# plot
if same_fig_flg:
    num_simulations = len(route_num_green_steps_map)
    fig, axes = plt.subplots(num_simulations, 1, figsize=(16, 8 * num_simulations))
    for ax, simulation_name in zip(axes, route_num_green_steps_map):
        tmp_route_num_green_steps_map = route_num_green_steps_map[simulation_name]

        bar_positions = []
        bar_heights = []
        bar_labels = []
        bar_colors = []
        for road_id in range(1, num_roads + 1):
            for direction_id in range(1, num_direction + 1):
                signal_group_id = (road_id - 1) * num_direction + direction_id
                num_green_steps = tmp_route_num_green_steps_map[signal_group_id, road_id, direction_id]
                
                bar_positions.append(signal_group_id)
                bar_heights.append(num_green_steps)
                bar_labels.append(f"{road_name_map[road_id]}-{direction_name_map[direction_id]}")
                bar_colors.append(bar_colors_map[direction_id])

        ax.bar(bar_positions, bar_heights, width=bar_width, color=bar_colors)
        ax.set_title(f'Number of Green Steps per Route - {simulation_name}')
        ax.set_xlabel('Route (Road-Direction)')
        ax.set_ylabel('Number of Green Steps')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(bar_labels, rotation=45)
    plt.tight_layout()
    fig.savefig(root_dir_path / 'results' / 'plots' / 'vehicle_bar_comparison.png')
else:
    raise NotImplementedError("Separate figure plotting not implemented yet.")




