import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re
import pickle
plt.rcParams['font.family'] = 'sans'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 50
plt.rcParams['axes.titlesize'] = 50
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['xtick.major.size'] = 20.0
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 12.0
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.major.size'] = 20.0
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 12.0
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["figure.edgecolor"] = "w"

# configuration
num_roads = 4

target_simulation_map = {
    # 'scoot': 'scoot/unbalanced_2222',
    # '4-phase mpc': 'mpc/balanced_4_700_2222_10',
    # '8-phase mpc': 'mpc/balanced_8_700_2222_10',
    '17-phase mpc': 'mpc/unbalanced_2222_10',
}

intersection_name_map = {
    1: '1-1-1',
    2: '3-1-1',
    3: '1-3-1',
    4: '1-1-3',
    5: '1-3-3',
    6: '3-1-3',
    7: '3-3-1',
}

root_path = Path(__file__).parent.parent.resolve()
phases_df = pd.read_csv(root_path / 'layout' / f"phases{num_roads}.csv")
phase_map = {}
for _, phase_row in phases_df.iterrows():
    phase_map[int(phase_row['id'])] = [int(phase_row[f"signal_group{order_id}"]) for order_id in range(1, num_roads + 1)]

signal_group_route_map = {}
for signal_group_id in range(1, num_roads * (num_roads -1) + 1):
    signal_group_route_map[signal_group_id] = (
        (signal_group_id-1) // (num_roads - 1) + 1, # direction
        (signal_group_id-1) % (num_roads - 1) + 1 # route
    )

simulation_green_rate_list_map = {}
num_intersections_list = []
for simulation_name, simulation_str in target_simulation_map.items():
    simulation_dir_path = root_path / 'results' / 'metrics' / simulation_str
    if not simulation_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {simulation_dir_path}")
    
    simulation_green_rate_list_map[simulation_name] = {}
    metric_file_paths = list(simulation_dir_path.glob('metric_*.pkl'))
    num_intersections_list.append(len(metric_file_paths))
    for metric_file_path in metric_file_paths:
        match_obj = re.match(rf"metric_(\d+)\.pkl", metric_file_path.name)
        intersection_id = int(match_obj.group(1))
        with open(metric_file_path, 'rb') as f:
            metric_data = pickle.load(f)
        phase_record_list = metric_data['phase']
        green_rate_list = [0 for _ in range(num_roads * (num_roads - 1))]
        for phase_id in phase_record_list:
            signal_group_list = phase_map[phase_id]
            for signal_group_id in signal_group_list:
                green_rate_list[signal_group_id - 1] += 1
        
        total_phases = len(phase_record_list)

        green_rate_list = [green_count / total_phases for green_count in green_rate_list]

        simulation_green_rate_list_map[simulation_name][intersection_id] = green_rate_list

if len(set(num_intersections_list)) != 1:
    raise ValueError("Number of intersections do not match among simulations.")
num_intersections = set(num_intersections_list).pop()
if num_intersections != len(intersection_name_map):
    raise ValueError("Number of intersections do not match with intersection_name_map.")


for intersection_id in range(1, num_intersections + 1):
    fig, ax = plt.subplots()

    intersection_name = intersection_name_map[intersection_id]

    for simulation_name in simulation_green_rate_list_map.keys():
        green_rate_list = simulation_green_rate_list_map[simulation_name][intersection_id]

        bin_centers = [signal_group_id for signal_group_id in range(1, num_roads * (num_roads - 1) + 1)]
        bin_width = 1

        ax.bar(
            bin_centers,
            green_rate_list,
            width=bin_width,
            alpha=0.5,
            label=simulation_name
        )

    ax.set_xlabel('Signal Group ID')
    ax.set_ylabel('Green Rate')
    ax.set_title(f'Intersection {intersection_name} Green Rate Comparison')
    ax.set_xticks(bin_centers)
    ax.legend()
    plt.tight_layout()

    output_dir = root_path / 'results' / 'plots' / f"green_rate_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / f"intersection_{intersection_name}_green_rate_comparison.png"
    plt.savefig(output_file_path)

    plt.close(fig)


    
print('test')