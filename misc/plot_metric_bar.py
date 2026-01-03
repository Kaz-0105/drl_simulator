import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np

# データの場所とラベルを設定
simulation_list = [
    ['scoot/main-minor_2222'],
    ['drl/apex/main-minor_2222_wait'],
    ['drl/apex/main-minor_2222_wait_v2'],
]
row_names = ['scoot', 'drl_wait', 'drl_wait_v2']

if len(simulation_list) != len(row_names):
    raise ValueError("The length of simulation_list and row_names must be the same.")

# 交通流パターンの数とラベルを設定
demand_type_names = [
    '1-1-1', 
    '3-1-1', 
    '1-3-1', 
    '1-1-3', 
    '1-3-3', 
    '3-1-3', 
    '3-3-1',
]
num_demand_types = len(demand_type_names)

# average_queueのy軸の最大値をmax_queueのy軸の最大値に合わせるかどうか
y_axis_flg = False

# MPC及びDRLのデータ取得
max_queue_list = []
average_queue_list = []
max_delay_list = []
average_delay_list = []
calc_time_list = []
for simulation_dir_wrapper in simulation_list:
    # リストのラッピングを外す
    simulation_dir = simulation_dir_wrapper[0]

    # 各シミュレーションの結果を保存するリストの初期化
    tmp_max_queue_list = []
    tmp_average_queue_list = []
    tmp_max_delay_list = []
    tmp_average_delay_list = []
    tmp_calc_time_list = []
    
    # 各交通パターンの結果を取得
    for demand_id in range(1, num_demand_types + 1):
        # 結果の保存されているファイルを取得
        metric_path = Path(f"results/metrics/{simulation_dir}/metric_{demand_id}.pkl")

        # ない時はエラー表示
        if not metric_path.exists():
            raise FileNotFoundError(f"File not found: {metric_path}")
        
        # pickleファイルを読み込み
        with open(metric_path, 'rb') as f:
            saved_data = pickle.load(f)
            if saved_data['max_queue'] is not None:
                max_queue_record = saved_data['max_queue']
                tmp_max_queue_list.append(max_queue_record['queue_length'].max())
            
            if saved_data['average_queue'] is not None:
                average_queue_record = saved_data['average_queue']
                tmp_average_queue_list.append(average_queue_record['queue_length'].mean())
            
            if saved_data['max_delay'] is not None:
                max_delay_record = saved_data['max_delay']
                tmp_max_delay_list.append(max_delay_record['delay'].max())
                            
            if saved_data['average_delay'] is not None:
                average_delay_record = saved_data['average_delay']
                tmp_average_delay_list.append(average_delay_record['delay'].mean())
            
            if saved_data['calc_time'] is not None:
                calc_time_record = saved_data['calc_time']
                tmp_calc_time_list.append(calc_time_record['calculation_time'].mean())

    # 各シミュレーションの結果をプッシュ       
    max_queue_list.append(tmp_max_queue_list)
    average_queue_list.append(tmp_average_queue_list)
    max_delay_list.append(tmp_max_delay_list)
    average_delay_list.append(tmp_average_delay_list)
    calc_time_list.append(tmp_calc_time_list)

# グラフの描画
fig_max_queue, ax_max_queue = plt.subplots(figsize=(16,9))
fig_average_queue, ax_average_queue = plt.subplots(figsize=(16,9))
fig_max_delay, ax_max_delay = plt.subplots(figsize=(16,9))
fig_average_delay, ax_average_delay = plt.subplots(figsize=(16,9))
fig_calc_time, ax_calc_time = plt.subplots(figsize=(16,9))

# 最大キュー長について
x = np.arange(len(demand_type_names))
for row_idx, row_max_queue_list in enumerate(max_queue_list):
    ax_max_queue.bar(
        [x + row_idx * 0.2 for x in range(len(row_max_queue_list))],
        row_max_queue_list,
        width=0.2,
        label=row_names[row_idx]
    )

ax_max_queue.set_xticks(x + 0.1)
ax_max_queue.set_xticklabels(demand_type_names, fontsize=14)
ax_max_queue.set_title('Max Queue Length', fontsize=16)
ax_max_queue.set_xlabel('Simulation Configuration', fontsize=14)
ax_max_queue.set_ylabel('Max Queue Length (m)', fontsize=14)
ax_max_queue.legend(fontsize=14)
fig_max_queue.savefig(Path(__file__).parent / '..' / 'results' / 'plots' / 'max_queue_comparison.png')

# 平均キュー長について
for row_idx, row_average_queue_list in enumerate(average_queue_list):
    ax_average_queue.bar(
        [x + row_idx * 0.2 for x in range(len(row_average_queue_list))],
        row_average_queue_list,
        width=0.2,
        label=row_names[row_idx]
    )
ax_average_queue.set_xticks(x + 0.1)
ax_average_queue.set_xticklabels(demand_type_names, fontsize=14)
ax_average_queue.set_title('Average Queue Length', fontsize=16)
ax_average_queue.set_xlabel('Simulation Configuration', fontsize=14)
ax_average_queue.set_ylabel('Average Queue Length (m)', fontsize=14)
ax_average_queue.legend(fontsize=14)

if y_axis_flg:
    # キューの最大値と平均値のy軸のスケールをそろえる
    max_y_value = max(
        max(max(row) for row in max_queue_list),
        max(max(row) for row in average_queue_list)
    )
    y_limit = (0, max_y_value * 1.1)  # 10%の余裕を持たせる
    ax_max_queue.set_ylim(y_limit)
    ax_average_queue.set_ylim(y_limit)
else:
    # max_queueについて
    max_y_value = max(max(row) for row in max_queue_list)
    y_limit = (0, max_y_value * 1.1)
    ax_max_queue.set_ylim(y_limit)

    # average_queueについて
    max_y_value = max(max(row) for row in average_queue_list)
    y_limit = (0, max_y_value * 1.1)
    ax_average_queue.set_ylim(y_limit)

fig_average_queue.savefig(Path(__file__).parent / '..' / 'results' / 'plots' / 'average_queue_comparison.png')  

# 遅れ時間について
for row_idx, row_max_delay_list in enumerate(max_delay_list):
    ax_max_delay.bar(
        [x + row_idx * 0.2 for x in range(len(row_max_delay_list))],
        row_max_delay_list,
        width=0.2,
        label=row_names[row_idx]
    )
ax_max_delay.set_xticks(x + 0.1)
ax_max_delay.set_xticklabels(demand_type_names, fontsize=14)
ax_max_delay.set_title('Max Delay Time', fontsize=16)
ax_max_delay.set_xlabel('Simulation Configuration', fontsize=14)
ax_max_delay.set_ylabel('Max Delay Time (s)', fontsize=14)
ax_max_delay.legend(fontsize=14)
fig_max_delay.savefig(Path(__file__).parent / '..' / 'results' / 'plots' / 'max_delay_comparison.png')

# 平均遅れ時間について
for row_idx, row_average_delay_list in enumerate(average_delay_list):
    ax_average_delay.bar(
        [x + row_idx * 0.2 for x in range(len(row_average_delay_list))],
        row_average_delay_list,
        width=0.2,
        label=row_names[row_idx]
    )
ax_average_delay.set_xticks(x + 0.1)
ax_average_delay.set_xticklabels(demand_type_names, fontsize=14) 
ax_average_delay.set_title('Average Delay Time', fontsize=16)
ax_average_delay.set_xlabel('Simulation Configuration', fontsize=14)
ax_average_delay.set_ylabel('Average Delay Time (s)', fontsize=14)
ax_average_delay.legend(fontsize=14)
fig_average_delay.savefig(Path(__file__).parent / '..' / 'results' / 'plots' / 'average_delay_comparison.png')

# 計算時間について
for row_idx, row_calc_time_list in enumerate(calc_time_list):
    ax_calc_time.bar(
        [x + row_idx * 0.2 for x in range(len(row_calc_time_list))],
        row_calc_time_list,
        width=0.2,
        label=row_names[row_idx]
    )
ax_calc_time.set_xticks(x + 0.1)
ax_calc_time.set_xticklabels(demand_type_names, fontsize=14)
ax_calc_time.set_title('Calculation Time', fontsize=16)
ax_calc_time.set_xlabel('Simulation Configuration', fontsize=14)
ax_calc_time.set_ylabel('Calculation Time (s)', fontsize=14)
ax_calc_time.legend(fontsize=14)
fig_calc_time.savefig(Path(__file__).parent / '..' / 'results' / 'plots' / 'calculation_time_comparison.png')



