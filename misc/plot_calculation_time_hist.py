from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np

# configuration 
mpc_simulation_file_map = {
    'balanced_500': 'balanced_500_2222_10',
    'balanced_600': 'balanced_600_2222_10',
    'balanced_700': 'balanced_700_2222_10',
    'balanced_800': 'balanced_800_2222_10',
}

stack_rule = 'simulation' # simulation or queue
num_bins = 50 # number of bins for histogram
save_file_extension = 'png' # png or eps
queue_bin_width = 5 # width of each bin for queue length histogram

# define root_dir_path
root_dir_path = (Path(__file__).parent / '..').resolve()

# change to path objects
for simulation_name, simulation_str in mpc_simulation_file_map.items():
    simulation_dir_path = root_dir_path / 'results' / 'metrics' / 'mpc' / simulation_str
    if not simulation_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {simulation_dir_path}")
    mpc_simulation_file_map[simulation_name] = simulation_dir_path


if stack_rule == 'simulation':
    # make min_calc_time and max_calc_time
    all_calc_time_list_map = {}
    for simulation_name, simulation_dir_path in mpc_simulation_file_map.items():
        all_calc_time_list = []
        for metric_file_path in simulation_dir_path.glob('metric_*.pkl'):
            with open(metric_file_path, 'rb') as f:
                metric_data = pickle.load(f)
            
            calc_time_df = metric_data['calc_time']
            calc_time_record = calc_time_df['calculation_time'].tolist()
            all_calc_time_list.extend(calc_time_record)
        
        all_calc_time_list_map[simulation_name] = all_calc_time_list

    max_calc_time = np.max([np.max(calc_time_list) for calc_time_list in all_calc_time_list_map.values()])
    
    # make histogram data for each simulation
    calc_time_hist_map = {}
    for simulation_name, all_calc_time_list in all_calc_time_list_map.items():
        hist, bin_edges = np.histogram(
            all_calc_time_list,
            bins=num_bins,
            range=(0, max_calc_time)
        )

        calc_time_hist_map[simulation_name] = hist

elif stack_rule == 'queue':
    # make queue bin edges
    avg_queue_max_list = []
    for simulation_name, simulation_dir_path in mpc_simulation_file_map.items():
        for metric_file_path in simulation_dir_path.glob('metric_*.pkl'):
            with open(metric_file_path, 'rb') as f:
                metric_data = pickle.load(f)
            
            avg_queue_df = metric_data['average_queue']
            avg_queue_record = avg_queue_df['queue_length'].tolist()
            avg_queue_max_list.append(np.max(avg_queue_record))
    
    max_queue_length = int(np.ceil(np.max(avg_queue_max_list) / queue_bin_width) * queue_bin_width)
    queue_bin_edges = list(range(0, max_queue_length + 1, queue_bin_width))

    # make calculation time list map for each queue bin
    calc_time_list_map = {}
    for avg_queue_idx in range(len(queue_bin_edges) - 1):
        calc_time_list_map[avg_queue_idx] = []

    max_calc_time = 0
    for simulation_name, simulation_dir_path in mpc_simulation_file_map.items():
        for metric_file_path in simulation_dir_path.glob('metric_*.pkl'):
            with open(metric_file_path, 'rb') as f:
                metric_data = pickle.load(f)
            
            avg_queue_df = metric_data['average_queue']
            calc_time_df = metric_data['calc_time']

            avg_queue_list = avg_queue_df['queue_length'].tolist()
            calc_time_list = calc_time_df['calculation_time'].tolist()

            for avg_queue, calc_time in zip(avg_queue_list, calc_time_list):
                avg_queue_idx = avg_queue // queue_bin_width
                calc_time_list_map[avg_queue_idx].append(calc_time)
            
            max_calc_time = max(max_calc_time, np.max(calc_time_list))

    # make histogram data for each queue bin
    calc_time_hist_map = {}
    for avg_queue_idx in calc_time_list_map.keys():
        hist, bin_edges = np.histogram(
            calc_time_list_map[avg_queue_idx],
            bins=num_bins,
            range=(0, max_calc_time),
        )
        calc_time_hist_map[avg_queue_idx] = hist

else:
    raise NotImplementedError(f"stack_rule {stack_rule} not implemented.")

# plot calculation bar figure
if stack_rule == 'simulation':
    # create figure and axis
    fig, ax = plt.subplots(figsize=(16,9))

    # color info
    colors_map = {}
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.35, 0.85, len(mpc_simulation_file_map)))
    for simulation_id, simulation_name in enumerate(mpc_simulation_file_map.keys()):
        colors_map[simulation_name] = colors[simulation_id]

    # bin centers and range
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_range = bin_edges[1] - bin_edges[0]

    bin_bottoms = np.zeros(num_bins)
    for simulation_name, calc_time_hist in calc_time_hist_map.items():
        ax.bar(
            bin_centers,
            calc_time_hist,
            width=bin_range,
            label=simulation_name,
            bottom = bin_bottoms,
            color=colors_map[simulation_name],
        )

        bin_bottoms += calc_time_hist
else:
    # create figure and axis
    fig, ax = plt.subplots(figsize=(16,9))

    # color info
    colors_map = {}
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.35, 0.85, len(calc_time_hist_map)))
    for queue_idx in calc_time_hist_map.keys():
        colors_map[queue_idx] = colors[queue_idx]

    # bin centers and range
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_range = bin_edges[1] - bin_edges[0]

    bin_bottoms = np.zeros(num_bins)
    for avg_queue_idx in sorted(calc_time_hist_map.keys()):
        calc_time_hist = calc_time_hist_map[avg_queue_idx]
        
        min_queue = queue_bin_width * avg_queue_idx
        max_queue = queue_bin_width * (avg_queue_idx + 1)

        ax.bar(
            bin_centers,
            calc_time_hist,
            width=bin_range,
            label=f"Queue [{min_queue}, {max_queue})",
            bottom = bin_bottoms,
            color=colors_map[avg_queue_idx],
        )

        bin_bottoms = bin_bottoms + calc_time_hist


ax.set_xlabel('Calculation Time (s)', fontsize=20)
ax.set_ylabel('Number of Optimizations', fontsize=20)
ax.set_title(f'Calculation Time Distribution', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
ax.legend(fontsize=16)
ax.set_xlim(bin_edges[0], bin_edges[-1])
xticks = ax.get_xticks()
ax.set_xticks(xticks[xticks != 0])



if save_file_extension == 'png':
    fig.savefig(root_dir_path / 'results' / 'plots' / f"calculation_bar_{stack_rule}.png", format='png')
elif save_file_extension == 'eps':
    fig.savefig(root_dir_path / 'results' / 'plots' / f"calculation_bar_{stack_rule}.eps", format='eps')
else:
    raise ValueError(f"Unsupported file extension: {save_file_extension}")

print('finished plotting a calculation bar figure.')