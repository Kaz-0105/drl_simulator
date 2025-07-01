import matplotlib.pyplot as plt
from pathlib import Path
import pickle

simulation_ids = [1, 2, 3]
simulation_names = [
    '1-1-1',
    '3-1-1',
    '1-3-1',
]
metric_flgs = {
    'max_queue': True,
    'calc_time': True,
}

fig_max_queue, ax_max_queue = plt.subplots()
fig_calc_time, ax_calc_time = plt.subplots()

for idx, simulation_id in enumerate(simulation_ids):
    metric_path = Path(f"results/metrics/metric_{simulation_id}.pkl")
    if not metric_path.exists():
        print(f"Metric file {metric_path} does not exist.")
        continue

    with open(metric_path, 'rb') as f:
        saved_data = pickle.load(f)
        if saved_data['max_queue'] is not None:
            max_queue_record = saved_data['max_queue']
            ax_max_queue.plot(max_queue_record['time'].to_numpy(), max_queue_record['queue_length'].to_numpy(), label=simulation_names[idx])
    

ax_max_queue.set_title('Max Queue Length Over Time')
ax_max_queue.set_xlabel('Time (s)')
ax_max_queue.set_ylabel('Max Queue Length (m)')
ax_max_queue.legend()
ax_max_queue.grid()

plt.show()

    