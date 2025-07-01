import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np

simulation_list = [
    [1, 2, 3, 4, 5, 6, 7],
    #[10, 11, 12, 13, 14, 15, 16],
]
column_names = ['1-1-1', '3-1-1', '1-3-1', '1-1-3', '1-3-3', '3-1-3', '3-3-1']
row_names = ['drl', 'mpc']

fig_max_queue, ax_max_queue = plt.subplots()

queue_list = []
for row_idx, row_simulation_list in enumerate(simulation_list):
    tmp_queue_list = []
    for col_idx, simulation_id in enumerate(row_simulation_list):
        metric_path = Path(f"results/metrics/metric_{simulation_id}.pkl")
        if not metric_path.exists():
            print(f"Metric file {metric_path} does not exist.")
            continue

        with open(metric_path, 'rb') as f:
            saved_data = pickle.load(f)
            if saved_data['max_queue'] is not None:
                max_queue_record = saved_data['max_queue']
                tmp_queue_list.append(max_queue_record['queue_length'].mean())
            
    queue_list.append(tmp_queue_list)

# 描画
x = np.arange(len(simulation_list[0]))
for row_idx, row_queue_list in enumerate(queue_list):
    ax_max_queue.bar(
        [x + row_idx * 0.15 for x in range(len(row_queue_list))],
        row_queue_list,
        width=0.15,
        label=row_names[row_idx]
    )

plt.xticks(x + 0.15, column_names)
ax_max_queue.set_title('Average Max Queue Length')
ax_max_queue.set_xlabel('Simulation Configuration')
ax_max_queue.set_ylabel('Average Max Queue Length (m)')
ax_max_queue.legend()
ax_max_queue.grid()

plt.show()



