import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# get target directories
target_dir_path_map = {
    'scoot': 'scoot/unbalanced_2222',
    'mpc': 'mpc/unbalanced_2222_10',
    'drl': 'drl/apex/unbalanced_2222_wait',
}

# init path objects
num_intersections = None
for target_name in target_dir_path_map:
    target_dir_path = target_dir_path_map[target_name]
    target_dir_path = Path(__file__) / '..' / '..' / 'results' / 'metrics' / target_dir_path

    if not target_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {target_dir_path}")

    target_dir_path_map[target_name] = target_dir_path.resolve()

    if num_intersections is None:
        num_intersections = len(list(target_dir_path.glob('metric_*.pkl')))
        continue

    if num_intersections != len(list(target_dir_path.glob('metric_*.pkl'))):
        raise ValueError("The number of intersection files is inconsistent among target directories. Do experiments with the same configuration.")

    

# set label name for each intersection (metric_*.pkl)
intersection_names = {
    1: '1-1-1',
    2: '3-1-1',
    3: '1-3-1',
    4: '1-1-3',
    5: '1-3-3',
    6: '3-1-3',
    7: '3-3-1',
}

if num_intersections != len(intersection_names):
    raise ValueError("The number of intersection names does not match the number of intersection files.")

# set plot flgs for each intersection
intersection_flgs = {
    1: True,
    2: True,
    3: True,
    4: True,
    5: True,
    6: True,
    7: True,
}

if not any(intersection_flgs.values()):
    raise ValueError("At least one plot flag must be True.")

# set metric types to plot
metric_type = 'average_queue' 

# create plot directory if not exists
plot_dir = Path(__file__) / '..' / '..' / 'results' / 'plots'
plot_dir.mkdir(parents=True, exist_ok=True)

time_series_data_map = {}
for intersection_id in range(1, num_intersections + 1):
    if not intersection_flgs[intersection_id]:
        continue

    tmp_intersection_time_series_data_map = {}

    for target_name, target_dir_path in target_dir_path_map.items():
        metric_file_path = target_dir_path / f'metric_{intersection_id}.pkl'

        if not metric_file_path.exists():
            raise FileNotFoundError(f"File not found: {metric_file_path}")

        with open(metric_file_path, 'rb') as f:
            data = pickle.load(f)
        
        if metric_type not in data:
            raise ValueError(f"{metric_type} data does not exist in {metric_file_path}")

        tmp_intersection_time_series_data_map[target_name] = data[metric_type]
    
    time_series_data_map[intersection_id] = tmp_intersection_time_series_data_map


# plot time series data
for intersection_id in range(1, num_intersections + 1):
    if not intersection_flgs[intersection_id]:
        continue
    
    fig, ax = plt.subplots()

    intersection_name = intersection_names[intersection_id]
    for target_name, time_series_data in time_series_data_map[intersection_id].items():
        ax.plot(time_series_data['time'], time_series_data['queue_length'], label=target_name)
    
    ax.set_title(f'Intersection {intersection_name} - {metric_type.replace("_", " ").title()} Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{metric_type.replace("_", " ").title()}')
    ax.legend()

    fig.savefig(f"results/plots/{intersection_name}_{metric_type}_time_series.png")

    