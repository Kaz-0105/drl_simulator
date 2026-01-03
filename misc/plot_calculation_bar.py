from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np

# configuration 
mpc_simulation_file_map = {
    'balanced_low': 'balanced_low_2222_10',
    'balanced': 'balanced_2222_10',
}

stack_rule = 'inflow' # inflow or route_choice
sec_range = 0.1
bin_width_rate = 1.0
save_file_extension = 'eps' # png or eps

# define root_dir_path
root_dir_path = (Path(__file__).parent / '..').resolve()

# change to path objects
for simulation_name, simulation_str in mpc_simulation_file_map.items():
    simulation_dir_path = root_dir_path / 'results' / 'metrics' / 'mpc' / simulation_str
    if not simulation_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {simulation_dir_path}")
    mpc_simulation_file_map[simulation_name] = simulation_dir_path

# make optimization_iterations_map
if stack_rule == 'inflow':
    optimization_iterations_map = {}
    for simulation_name, simulation_dir_path in mpc_simulation_file_map.items():
        tmp_optimization_iterations_map = {}
        for metric_file_path in simulation_dir_path.glob('metric_*.pkl'):
            with open(metric_file_path, 'rb') as f:
                metric_data = pickle.load(f)
            
            calc_time_df = metric_data['calc_time']
            calc_time_record = calc_time_df['calculation_time'].tolist()

            for calc_time in calc_time_record:
                time_id = int(round(calc_time / sec_range))
                if time_id not in tmp_optimization_iterations_map:
                    tmp_optimization_iterations_map[time_id] = 0
                tmp_optimization_iterations_map[time_id] += 1
            
        optimization_iterations_map[simulation_name] = tmp_optimization_iterations_map
else:
    raise NotImplementedError(f"stack_rule {stack_rule} not implemented.")

# plot calculation bar figure
fig, ax = plt.subplots(figsize=(16,9))

colors_map = {}
cmap = plt.cm.Blues
colors = cmap(np.linspace(0.35, 0.85, len(mpc_simulation_file_map)))
for simulation_id, simulation_name in enumerate(mpc_simulation_file_map.keys()):
    colors_map[simulation_name] = colors[simulation_id]

bar_bottoms = {}
for simulation_name, tmp_optimization_iterations_map in optimization_iterations_map.items():
    bar_positions = [round(time_id * sec_range + sec_range / 2, 3) for time_id in sorted(tmp_optimization_iterations_map.keys())]

    bar_heights = []
    for time_id in sorted(tmp_optimization_iterations_map.keys()):
        bar_heights.append(tmp_optimization_iterations_map[time_id])
        
    tmp_bar_bottoms = []
    for time_id in sorted(tmp_optimization_iterations_map.keys()):
        if time_id in bar_bottoms:
            tmp_bar_bottoms.append(bar_bottoms[time_id])
            bar_bottoms[time_id] += tmp_optimization_iterations_map[time_id]
        else:
            tmp_bar_bottoms.append(0)
            bar_bottoms[time_id] = tmp_optimization_iterations_map[time_id]
    
    ax.bar(
        bar_positions,
        bar_heights,
        width=sec_range * bin_width_rate,
        bottom=tmp_bar_bottoms,
        label=simulation_name,
        color=colors_map[simulation_name],
    )

ax.set_xlabel('Calculation Time (s)', fontsize=20)
ax.set_ylabel('Number of Optimizations', fontsize=20)
ax.set_title(f'Calculation Time Distribution', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
ax.legend(fontsize=16)
ax.set_xlim(0, max(bar_bottoms.keys()) * sec_range + sec_range)
xticks = ax.get_xticks()
ax.set_xticks(xticks[xticks != 0])

if save_file_extension == 'png':
    fig.savefig(root_dir_path / 'results' / 'plots' / f"calculation_bar_{stack_rule}.png", format='png')
elif save_file_extension == 'eps':
    fig.savefig(root_dir_path / 'results' / 'plots' / f"calculation_bar_{stack_rule}.eps", format='eps')
else:
    raise ValueError(f"Unsupported file extension: {save_file_extension}")

print('finished plotting a calculation bar figure.')