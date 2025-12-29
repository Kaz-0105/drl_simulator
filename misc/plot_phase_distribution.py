import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

# select simulation directories which you want to compare
simulation_dir_path_map = {
    'scoot': 'scoot/balanced_2222',
    'mpc': 'mpc/balanced_2222_10',
    'drl_v2': 'drl/apex/balanced_2222_wait_v2',
}

# select an intersection to compare
intersection_id = 1

# set number of phases
num_phases = 17

# configuration of plot
same_fig_flg = True
bar_width = 0.35



root_path = (Path(__file__).parent / '..').resolve()
metrics_dir_path = root_path / 'results' / 'metrics'

for simulation_name, simulation_dir_str in simulation_dir_path_map.items():
    simulation_path = metrics_dir_path / simulation_dir_str
    if not simulation_path.exists():
        raise FileNotFoundError(f"File not found: {simulation_path}")
    simulation_dir_path_map[simulation_name] = metrics_dir_path / simulation_dir_str

intersection_metric_file_path_map = {}
for simulation_name, simulation_dir_path in simulation_dir_path_map.items():
    intersection_metric_file_path = simulation_dir_path / f"metric_{intersection_id}.pkl"
    if not intersection_metric_file_path.exists():
        raise FileNotFoundError(f"File not found: {intersection_metric_file_path}")
    intersection_metric_file_path_map[simulation_name] = intersection_metric_file_path

# make data
phase_distribution_map = {}
for simulation_name, intersection_metric_file_path in intersection_metric_file_path_map.items():
    with open(intersection_metric_file_path, 'rb') as f:
        metric_data = pickle.load(f)
    
    phase_record = metric_data['phase']
    tmp_phase_distribution_map = {}
    for phase_id in phase_record:
        if phase_id not in tmp_phase_distribution_map:
            tmp_phase_distribution_map[phase_id] = 0
        
        tmp_phase_distribution_map[phase_id] += 1
    
    phase_distribution_map[simulation_name] = tmp_phase_distribution_map

# plot
if same_fig_flg:
    fig, axes = plt.subplots(len(phase_distribution_map), 1, figsize=(16, 8 * len(phase_distribution_map)))
    for ax, simulation_name in zip(axes, phase_distribution_map):
        tmp_phase_distribution_map = phase_distribution_map[simulation_name]

        for phase_id in range(1, num_phases + 1):
            tmp_count = 0
            if phase_id in tmp_phase_distribution_map:
                tmp_count = tmp_phase_distribution_map[phase_id]
            
            ax.bar(phase_id, tmp_count, width=bar_width, label=f'Phase {phase_id}')
        ax.set_title(f'Phase Distribution - {simulation_name}')
        ax.set_xlabel('Phase ID')
        ax.set_ylabel('Count')
    plt.tight_layout()
    fig.savefig(root_path / 'results' / 'plots' / 'phase_distribution_comparison.png')

else:
    for simulation_name in phase_distribution_map:
        fig, ax = plt.subplots(figsize=(16, 8))
        tmp_phase_distribution_map = phase_distribution_map[simulation_name]
        for phase_id in range(1, num_phases + 1):
            tmp_count = 0
            if phase_id in tmp_phase_distribution_map:
                tmp_count = tmp_phase_distribution_map[phase_id]
            
            ax.bar(phase_id, tmp_count, width=bar_width, label=f'Phase {phase_id}')

        ax.set_title(f'Phase Distribution - {simulation_name}')
        ax.set_xlabel('Phase ID')
        ax.set_ylabel('Count')
        fig.savefig(root_path / 'results' / 'plots' / f'phase_distribution_{simulation_name}.png')