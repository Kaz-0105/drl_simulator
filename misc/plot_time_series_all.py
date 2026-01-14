from pathlib import Path
import re
import pickle
import matplotlib.pyplot as plt

simulation_name = 'mpc/balanced_connected_2222_10'
metric = 'max_queue'
type = 'mix'  # 'mix' or 'separate'


simulation_dir_path = Path(__file__) / '..' / '..' / 'results' / 'metrics' / simulation_name
simulation_dir_path = simulation_dir_path.resolve()
if not simulation_dir_path.exists():
    raise FileNotFoundError(f"Directory not found: {simulation_dir_path}")

# load data
intersection_metric_record_map = {}
for metric_file_path in simulation_dir_path.glob('metric_*.pkl'):
    match_obj = re.match(rf"metric_(\d+)\.pkl", metric_file_path.name)
    
    with open(metric_file_path, 'rb') as f:
        metric_data = pickle.load(f)

    if metric not in metric_data:
        raise ValueError(f"Metric '{metric}' not found in file: {metric_file_path}")
    
    intersection_metric_record_map[int(match_obj.group(1))] = metric_data[metric]

if type == 'separete':
    # plot
    fig, ax = plt.subplots()
    for intersection_id, metric_record in intersection_metric_record_map.items():
        if metric in ['average_queue', 'max_queue']:
            ax.plot(
                metric_record['time'],
                metric_record['queue_length'],
                label=f'Intersection {intersection_id}'
            )
        elif metric in ['average_delay', 'max_delay']:
            ax.plot(
                metric_record['time'],
                metric_record['delay'],
                label=f'Intersection {intersection_id}'
            )
        elif metric == 'calc_time':
            ax.plot(
                metric_record['time'],
                metric_record['calculation_time'],
                label=f'Intersection {intersection_id}'
            )
        else:
            raise NotImplementedError(f"Not supported metric: {metric}")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax.set_title(f'{metric.replace("_", " ").title()} Over Time for Each Intersection')
    ax.legend()
    ax.set_xlim(0,metric_record['time'].tolist()[-1])
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[xticks != 0])

elif type == 'mix':
    # plot
    fig, ax = plt.subplots()

    average_metric_list = None
    for _, metric_record in intersection_metric_record_map.items():
        if metric in ['average_queue', 'max_queue']:
            current_metric_list = metric_record['queue_length'].to_numpy()
        elif metric in ['average_delay', 'max_delay']:
            current_metric_list = metric_record['delay'].to_numpy()
        elif metric == 'calc_time':
            current_metric_list = metric_record['calculation_time'].to_numpy()
        else:
            raise NotImplementedError(f"Not supported metric: {metric}")
        
        if average_metric_list is None:
            average_metric_list = current_metric_list
            time_list = metric_record['time'].to_numpy()
        else:
            average_metric_list += current_metric_list
    
    average_metric_list = average_metric_list / len(intersection_metric_record_map)

    ax.plot(
        time_list,
        average_metric_list,
        label=f'Average {metric.replace("_", " ").title()}'
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax.set_title(f'{metric.replace("_", " ").title()} Over Time (Average of All Intersections)')
    ax.legend()
    ax.set_xlim(0, time_list[-1])
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[xticks != 0])
    
else: 
    raise NotImplementedError(f"Not supported type: {type}")

save_file_path = Path(__file__) / '..' / '..' / 'results' / 'plots' / f"{simulation_name.replace('/', '_')}_{metric}_{type}_time_series.png"
save_file_path = save_file_path.resolve()
save_file_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_file_path)    

print('Finished.')



