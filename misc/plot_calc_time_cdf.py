from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'sans'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 50
plt.rcParams['axes.titlesize'] = 50
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
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
mpc_simulation_file_map = {
    '500 veh/h': 'balanced_500_2222_10',
    '600 veh/h': 'balanced_600_2222_10',
    '700 veh/h': 'balanced_700_2222_10',
    '800 veh/h': 'balanced_800_2222_10',
}

stack_rule = 'simulation' # simulation or queue
num_bins = 30 # number of bins for histogram
save_file_extension = 'eps' # png or eps
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
    calc_time_cdf_map = {}
    for simulation_name, all_calc_time_list in all_calc_time_list_map.items():
        hist, bin_edges = np.histogram(
            all_calc_time_list,
            bins=num_bins,
            range=(0, max_calc_time)
        )

        cdf = [0]
        cdf.extend(np.cumsum(hist) / np.sum(hist))

        calc_time_cdf_map[simulation_name] = cdf

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

    # make cdf data for each queue bin
    calc_time_cdf_map = {}
    for avg_queue_idx in calc_time_list_map.keys():
        hist, bin_edges = np.histogram(
            calc_time_list_map[avg_queue_idx],
            bins=num_bins,
            range=(0, max_calc_time),
        )
        cdf = [0]
        cdf.extend(np.cumsum(hist) / np.sum(hist))
        calc_time_cdf_map[avg_queue_idx] = cdf

else:
    raise NotImplementedError(f"stack_rule {stack_rule} not implemented.")

# plot calculation bar figure
if stack_rule == 'simulation':
    # create figure and axis
    fig, ax = plt.subplots(figsize=(16,9))

    for simulation_name, calc_time_cdf in calc_time_cdf_map.items():
        ax.plot(
            bin_edges,
            calc_time_cdf,
            label=simulation_name,
            marker='o',
            linestyle='-',
            linewidth=4,
        )
elif stack_rule == 'queue':
    # create figure and axis
    fig, ax = plt.subplots(figsize=(16,9))

    for avg_queue_idx in sorted(calc_time_cdf_map.keys()):
        calc_time_cdf = calc_time_cdf_map[avg_queue_idx]
        
        min_queue = queue_bin_width * avg_queue_idx
        max_queue = queue_bin_width * (avg_queue_idx + 1)

        ax.plot(
            bin_edges,
            calc_time_cdf,
            label=f"Queue [{min_queue}, {max_queue})",
            marker='o',
            linestyle='-',
            linewidth=4,
        )

ax.set_xlabel('Calculation Time (s)', fontsize=24, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=24, fontweight='bold')
ax.set_title(f'Calculation Time Cumulative Distribution Function', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=24)
ax.legend(fontsize=16)
ax.set_xlim(bin_edges[0], bin_edges[-1] + ( bin_edges[1] - bin_edges[0] ))
ax.set_ylim(0, 1.1)
ax.grid(
    which='major',
    axis='both',
    linestyle='--',
    linewidth=1,
    alpha=0.5,
)
leg= ax.legend(
    loc='lower right',
    title='Vehicle Inflow Rate' if stack_rule == 'simulation' else 'Average Queue Length',
    title_fontsize=24,
    fontsize=24,
)
leg.get_title().set_fontweight('bold')
fig.tight_layout()

save_dir = root_dir_path / 'results' / 'plots' / 'calc_time_cdf'
save_dir.mkdir(parents=True, exist_ok=True)
if save_file_extension == 'png':    
    fig.savefig(save_dir / f"calc_cdf_{stack_rule}.png", format='png')
elif save_file_extension == 'eps':
    fig.savefig(save_dir / f"calc_cdf_{stack_rule}.eps", format='eps')
else:
    raise ValueError(f"Unsupported file extension: {save_file_extension}")

print('finished plotting a calculation bar figure.')