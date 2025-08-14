import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict
import pandas as pd

# scoot，mpc，drlの比較
plot_name = 'balanced'   # 'main' or 'unbalanced'

simulation_list = [
    [
        ['mpc_balanced_10_1_17'],
        ['drl_balanced_10_1'],
    ],
    [
        ['mpc_main_10_1_17'],
        ['drl_main_10_1'],
    ],
    [
        ['mpc_unbalanced_10_1_17'],
        ['drl_unbalanced_10_1'],
    ],
]

num_bars = 7
row_names = ['balanced', 'main', 'unbalanced']
column_names = ['mpc', 'drl']
simulation_names = ['1-1-1', '3-1-1', '1-3-1', '1-1-3', '1-3-3', '3-1-3', '3-3-1']

if plot_name == 'main':
    plot_name_en = 'Arterial Road Type'
elif plot_name == 'balanced':
    plot_name_en = 'Balanced Type'
else:
    plot_name_en = 'Unbalanced Type'


# scoot，mpc，drlのデータ取得
max_queues = defaultdict(lambda: defaultdict(dict))
average_queues = defaultdict(lambda: defaultdict(dict))
average_delays = defaultdict(lambda: defaultdict(dict))
calc_times = defaultdict(lambda: defaultdict(dict))

for row_idx, tmp_simulation_list in enumerate(simulation_list):
    for col_idx, tmp_simulation in enumerate(tmp_simulation_list):
        for simulation_id in range(num_bars):
            metric_path = Path(f"results/metrics/{tmp_simulation[0]}/metric_{simulation_id + 1}.pkl")
            if not metric_path.exists():
                print(f"Metric file {metric_path} does not exist.")
                continue

            with open(metric_path, 'rb') as f:
                saved_data = pickle.load(f)
                if saved_data['max_queue'] is not None:
                    max_queue_record = saved_data['max_queue']
                    max_queues[row_names[row_idx]][column_names[col_idx]][simulation_names[simulation_id]] = np.mean(max_queue_record['queue_length'])

                if saved_data['average_queue'] is not None:
                    average_queue_record = saved_data['average_queue']
                    average_queues[row_names[row_idx]][column_names[col_idx]][simulation_names[simulation_id]] = np.mean(average_queue_record['queue_length'])

                if saved_data['average_delay'] is not None:
                    average_delay_record = saved_data['average_delay']
                    average_delays[row_names[row_idx]][column_names[col_idx]][simulation_names[simulation_id]] = np.mean(average_delay_record['delay'])

                if saved_data['calc_time'] is not None:
                    calc_time_record = saved_data['calc_time']
                    calc_times[row_names[row_idx]][column_names[col_idx]][simulation_names[simulation_id]] = np.mean(calc_time_record['calculation_time'])

# グラフの作成
fig_max_queue, ax_max_queue = plt.subplots(figsize=(12, 6))
fig_average_queue, ax_average_queue = plt.subplots(figsize=(12, 6))
fig_average_delay, ax_average_delay = plt.subplots(figsize=(12, 6))
fig_calc_time, ax_calc_time = plt.subplots(figsize=(12, 6))

# 最大キュー長の棒グラフ
x = np.arange(len(simulation_names))
for col_idx, col_name in enumerate(column_names):
    max_queue_values = [max_queues[plot_name][col_name][sim_name] for sim_name in simulation_names]
    ax_max_queue.bar(x + col_idx * 0.15, max_queue_values, width=0.15, label=col_name)
ax_max_queue.set_xticks(x + 0.15 * (len(column_names) - 1) / 2)
ax_max_queue.set_xticklabels(simulation_names, fontsize=20)
ax_max_queue.tick_params(axis='y', labelsize=20)
ax_max_queue.set_title(f"Max Queue Length : {plot_name}", fontsize=24)
ax_max_queue.set_xlabel('Simulation Configuration', fontsize=20)
ax_max_queue.set_ylabel('Max Queue Length (m)', fontsize=20)
ax_max_queue.legend(title='Control Method', fontsize=20, title_fontsize=20)

# 平均キュー長の棒グラフ〉〉
for col_idx, col_name in enumerate(column_names):
    average_queue_values = [average_queues[plot_name][col_name][sim_name] for sim_name in simulation_names]
    ax_average_queue.bar(x + col_idx * 0.15, average_queue_values, width=0.15, label=col_name)
ax_average_queue.set_xticks(x + 0.15 * (len(column_names) - 1) / 2)
ax_average_queue.set_xticklabels(simulation_names, fontsize=20)
ax_average_queue.tick_params(axis='y', labelsize=20)
ax_average_queue.set_title(f"Average Queue Length : {plot_name_en}", fontsize=24)
ax_average_queue.set_xlabel('Turn Ratio (Left : Straight : Right)', fontsize=20)
ax_average_queue.set_ylabel('Average Queue Length (m)', fontsize=20)
ax_average_queue.legend(title='Control Method', fontsize=20, title_fontsize=20)
# 平均遅延時間の棒グラフ
for col_idx, col_name in enumerate(column_names):
    average_delay_values = [average_delays[plot_name][col_name][sim_name] for sim_name in simulation_names]
    ax_average_delay.bar(x + col_idx * 0.15, average_delay_values, width=0.15, label=col_name)
ax_average_delay.set_xticks(x + 0.15 * (len(column_names) - 1) / 2)
ax_average_delay.set_xticklabels(simulation_names, fontsize=20)
ax_average_delay.tick_params(axis='y', labelsize=20)
ax_average_delay.set_title(f"Average Delay Time : {plot_name_en}", fontsize=24)
ax_average_delay.set_xlabel('Turn Ratio (Left : Straight : Right)', fontsize=20)
ax_average_delay.set_ylabel('Average Delay Time (s)', fontsize=20)
ax_average_delay.legend(title='Control Method', fontsize=20, title_fontsize=20)


# 計算時間の棒グラフ
for col_idx, col_name in enumerate(column_names):
    if col_name == 'scoot':
        continue

    calc_time_values = [calc_times[plot_name][col_name][sim_name] for sim_name in simulation_names]
    ax_calc_time.bar(x + col_idx * 0.15, calc_time_values, width=0.15, label=col_name)

ax_calc_time.set_xticks(x + 0.15 * (len(column_names) - 1) / 2)
ax_calc_time.set_xticklabels(simulation_names, fontsize=20)
ax_calc_time.tick_params(axis='y', labelsize=20)
ax_calc_time.set_title(f"Calculation Time : {plot_name_en}", fontsize=24)
ax_calc_time.set_xlabel('Turn Ratio (Left : Straight : Right)', fontsize=20)
ax_calc_time.set_ylabel('Calculation Time (s)', fontsize=20)
ax_calc_time.legend(title='Control Method', fontsize=20, title_fontsize=20)

plt.show()


