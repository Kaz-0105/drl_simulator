from pathlib import Path
import re
import pickle
import matplotlib.pyplot as plt
import pandas as pd

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

def calculate_ema(data, span):
    # spanは「何日移動平均か」に相当するパラメータ
    return pd.Series(data).ewm(span=span, adjust=False).mean().values

simulation_name = 'mpc/balanced_connected_600_8_2222_10'
metric = 'average_queue'
type = 'separate'  # 'mix' or 'separate'


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

if type == 'separate':
    # plot
    fig, ax = plt.subplots()
    for intersection_id, metric_record in intersection_metric_record_map.items():
        if metric in ['average_queue', 'max_queue']:
            ax.plot(
                metric_record['time'],
                calculate_ema(metric_record['queue_length'], span=20),
                label=f'ID: {intersection_id}',
                linewidth=4,
                linestyle='-',
            )
        elif metric in ['average_delay', 'max_delay']:
            ax.plot(
                metric_record['time'],
                metric_record['delay'],
                label=f'Intersection {intersection_id}',
                linewidth=4
            )
        elif metric == 'calc_time':
            ax.plot(
                metric_record['time'],
                metric_record['calculation_time'],
                label=f'Intersection {intersection_id}',
                linewidth=4
            )
        else:
            raise NotImplementedError(f"Not supported metric: {metric}")

    ax.set_xlabel('Time (s)', fontsize=20, fontweight='bold')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=20, fontweight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} Over Time for Each Intersection', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.legend(
        title='Intersections',
        fontsize=16,
        title_fontsize=20,
    )
    ax.set_xlim(0, metric_record['time'].to_numpy()[-1])
    ax.set_ylim(0, None)

elif type == 'mix':
    # plot
    fig, ax = plt.subplots(figsize=(16,9))

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

    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=20)
    ax.set_title(f'{metric.replace("_", " ").title()} Over Time (Average of All Intersections)', fontsize=24)
    ax.legend()
    ax.set_xlim(0, time_list[-1])
    ax.set_ylim(0, max(average_metric_list)*1.1)
    fig.tight_layout()
    
else: 
    raise NotImplementedError(f"Not supported type: {type}")

save_file_path = Path(__file__) / '..' / '..' / 'results' / 'plots' / 'time_series_all' / f"{simulation_name.replace('/', '_')}_{metric}_{type}_time_series.png"
save_file_path = save_file_path.resolve()
save_file_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_file_path)    

print('Finished.')



