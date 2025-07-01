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
calc_time_list = []
for row_idx, row_simulation_list in enumerate(simulation_list):
    tmp_max_queue_list = []
    tmp_average_queue_list = []
    tmp_calc_time_list = []
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
            
            if simulation_id == 10:
                print('test')
            if saved_data['calc_time'] is not None:
                calc_time_record = saved_data['calc_time']
                tmp_calc_time_list.append(calc_time_record['calculation_time'].mean())
            
    max_queue_list.append(tmp_max_queue_list)
    average_queue_list.append(tmp_average_queue_list)
    calc_time_list.append(tmp_calc_time_list)

fig_max_queue, ax_max_queue = plt.subplots()
fig_average_queue, ax_average_queue = plt.subplots()
fig_calc_time, ax_calc_time = plt.subplots()

# 最大キュー長について
x = np.arange(len(simulation_list[0]))
for row_idx, row_max_queue_list in enumerate(max_queue_list):
    ax_max_queue.bar(
        [x + row_idx * 0.2 for x in range(len(row_max_queue_list))],
        row_max_queue_list,
        width=0.2,
        label=row_names[row_idx]
    )

ax_max_queue.set_xticks(x + 0.1)
ax_max_queue.set_xticklabels(column_names, fontsize=14)
ax_max_queue.set_title('Average Max Queue Length', fontsize=16)
ax_max_queue.set_xlabel('Simulation Configuration', fontsize=14)
ax_max_queue.set_ylabel('Average Max Queue Length (m)', fontsize=14)
ax_max_queue.legend(fontsize=14)

# 平均キュー長について
for row_idx, row_average_queue_list in enumerate(average_queue_list):
    ax_average_queue.bar(
        [x + row_idx * 0.2 for x in range(len(row_average_queue_list))],
        row_average_queue_list,
        width=0.2,
        label=row_names[row_idx]
    )
ax_average_queue.set_xticks(x + 0.1)
ax_average_queue.set_xticklabels(column_names, fontsize=14)
ax_average_queue.set_title('Average Queue Length', fontsize=16)
ax_average_queue.set_xlabel('Simulation Configuration', fontsize=14)
ax_average_queue.set_ylabel('Average Queue Length (m)', fontsize=14)
ax_average_queue.legend(fontsize=14)

# 計算時間について
for row_idx, row_calc_time_list in enumerate(calc_time_list):
    ax_calc_time.bar(
        [x + row_idx * 0.2 for x in range(len(row_calc_time_list))],
        row_calc_time_list,
        width=0.2,
        label=row_names[row_idx]
    )
ax_calc_time.set_xticks(x + 0.1)
ax_calc_time.set_xticklabels(column_names, fontsize=14)
ax_calc_time.set_title('Calculation Time', fontsize=16)
ax_calc_time.set_xlabel('Simulation Configuration', fontsize=14)
ax_calc_time.set_ylabel('Calculation Time (s)', fontsize=14)
ax_calc_time.legend(fontsize=14)

plt.show()



