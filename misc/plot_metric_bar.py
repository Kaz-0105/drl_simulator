import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np

simulation_list = [
    [1, 2, 3, 4, 5, 6, 7],
    [10, 11, 12, 13, 14, 15, 16],
]
column_names = ['1-1-1', '3-1-1', '1-3-1', '1-1-3', '1-3-3', '3-1-3', '3-3-1']
row_names = ['drl', 'mpc']

max_queue_list = []
average_queue_list = []
for row_idx, row_simulation_list in enumerate(simulation_list):
    tmp_max_queue_list = []
    tmp_average_queue_list = []
    for col_idx, simulation_id in enumerate(row_simulation_list):
        metric_path = Path(f"results/metrics/metric_{simulation_id}.pkl")
        if not metric_path.exists():
            print(f"Metric file {metric_path} does not exist.")
            continue

        with open(metric_path, 'rb') as f:
            saved_data = pickle.load(f)
            if saved_data['max_queue'] is not None:
                max_queue_record = saved_data['max_queue']
                tmp_max_queue_list.append(max_queue_record['queue_length'].mean())
            
            if saved_data['average_queue'] is not None:
                average_queue_record = saved_data['average_queue']
                tmp_average_queue_list.append(average_queue_record['queue_length'].mean())
            
    max_queue_list.append(tmp_max_queue_list)
    average_queue_list.append(tmp_average_queue_list)

fig_max_queue, ax_max_queue = plt.subplots()
fig_average_queue, ax_average_queue = plt.subplots()

# 最大キュー長について
x = np.arange(len(simulation_list[0]))
for row_idx, row_max_queue_list in enumerate(max_queue_list):
    ax_max_queue.bar(
        [x + row_idx * 0.15 for x in range(len(row_max_queue_list))],
        row_max_queue_list,
        width=0.15,
        label=row_names[row_idx]
    )

ax_max_queue.set_xticks(x + 0.15, column_names)
ax_max_queue.set_title('Average Max Queue Length')
ax_max_queue.set_xlabel('Simulation Configuration')
ax_max_queue.set_ylabel('Average Max Queue Length (m)')
ax_max_queue.legend()
ax_max_queue.grid()

# 平均キュー長について
for row_idx, row_average_queue_list in enumerate(average_queue_list):
    ax_average_queue.bar(
        [x + row_idx * 0.15 for x in range(len(row_average_queue_list))],
        row_average_queue_list,
        width=0.15,
        label=row_names[row_idx]
    )
ax_average_queue.set_xticks(x + 0.15, column_names)
ax_average_queue.set_title('Average Queue Length')
ax_average_queue.set_xlabel('Simulation Configuration')
ax_average_queue.set_ylabel('Average Queue Length (m)')
ax_average_queue.legend()
ax_average_queue.grid()

plt.show()



