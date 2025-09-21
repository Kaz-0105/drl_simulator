from pathlib import Path
from collections import defaultdict
import pickle
import statistics
import matplotlib.pyplot as plt

# 比較する重みのリスト
weights = [
    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [5, 7, 10, 12, 15, 20],
]

# 流入量のタイプ
inflow_types = [
    'balanced',
    'unbalanced',
]

# 道路の構成
road_layout = '2222'

# weightsとinflow_typeの長さが同じであることを確認
if len(weights) != len(inflow_types):
    raise ValueError("The length of weights and inflow_type must be the same.")

# 重みを昇順にソート
weights.sort()

# mpcのデータを取得
metrics = defaultdict(dict)
for inflow_idx, tmp_weights in enumerate(weights):
    # inflow_typeを取得
    inflow_type = inflow_types[inflow_idx]

    for weight in tmp_weights:
        # 結果の保存されているディレクトリを取得
        simulation_dir = Path.cwd() / 'results' / 'metrics' / 'mpc' / f"{inflow_type}_{road_layout}_{weight}"

        # tmp_weight_metricsの初期化
        tmp_weight_metrics = defaultdict(list)

        # demand_typeを走査
        demand_type_id = 0
        while True:
            # demand_type_idをインクリメントしファイルパスを取得
            demand_type_id += 1
            tmp_file_path = simulation_dir / f"metric_{demand_type_id}.pkl"
            
            # ファイルが存在しなければ終了
            if not tmp_file_path.exists():
                break

            # pickleファイルを読み込み
            with open(tmp_file_path, 'rb') as f:
                saved_data = pickle.load(f)
                
                if saved_data['max_queue'] is not None:
                    max_queue_record = saved_data['max_queue']
                    tmp_weight_metrics['max_queue'].append(max_queue_record['queue_length'].max())
                
                if saved_data['average_queue'] is not None:
                    average_queue_record = saved_data['average_queue']
                    tmp_weight_metrics['average_queue'].append(average_queue_record['queue_length'].mean())
                
                if saved_data['max_delay'] is not None:
                    max_delay_record = saved_data['max_delay']
                    tmp_weight_metrics['max_delay'].append(max_delay_record['delay'].max())
                
                if saved_data['average_delay'] is not None:
                    average_delay_record = saved_data['average_delay']
                    tmp_weight_metrics['average_delay'].append(average_delay_record['delay'].mean())
                
                if saved_data['calc_time'] is not None:
                    calc_time_record = saved_data['calc_time']
                    tmp_weight_metrics['calculation_time'].append(calc_time_record['calculation_time'].mean())
        
        # 複数のdemand_typeの結果を1つに集約
        if 'max_queue' in tmp_weight_metrics:
            tmp_weight_metrics['max_queue'] = max(tmp_weight_metrics['max_queue'])
        
        if 'average_queue' in tmp_weight_metrics:
            tmp_weight_metrics['average_queue'] = statistics.mean(tmp_weight_metrics['average_queue'])

        if 'max_delay' in tmp_weight_metrics:
            tmp_weight_metrics['max_delay'] = max(tmp_weight_metrics['max_delay'])
        
        if 'average_delay' in tmp_weight_metrics:
            tmp_weight_metrics['average_delay'] = statistics.mean(tmp_weight_metrics['average_delay'])

        if 'calculation_time' in tmp_weight_metrics:
            tmp_weight_metrics['calculation_time'] = statistics.mean(tmp_weight_metrics['calculation_time'])

        # metricsに追加
        metrics[inflow_type][weight] = tmp_weight_metrics

# グラフの描画
fig_max_queue, ax_max_queue = plt.subplots()
for inflow_type in inflow_types:
    x_vals = list(metrics[inflow_type].keys())
    y_vals = [metrics[inflow_type][weight]['max_queue'] for weight in x_vals]
    ax_max_queue.plot(x_vals, y_vals, marker='o', label=inflow_type)
ax_max_queue.set_xlabel('Weight')   
ax_max_queue.set_ylabel('Max Queue Length')
ax_max_queue.set_title('Max Queue Length vs Weight')
ax_max_queue.legend(title='Inflow Type')

fig_average_queue, ax_average_queue = plt.subplots()
for inflow_type in inflow_types:
    x_vals = list(metrics[inflow_type].keys())
    y_vals = [metrics[inflow_type][weight]['average_queue'] for weight in x_vals]
    ax_average_queue.plot(x_vals, y_vals, marker='o', label=inflow_type)
ax_average_queue.set_xlabel('Weight')
ax_average_queue.set_ylabel('Average Queue Length')
ax_average_queue.set_title('Average Queue Length vs Weight')
ax_average_queue.legend(title='Inflow Type')

fig_max_delay, ax_max_delay = plt.subplots()
for inflow_type in inflow_types:
    x_vals = list(metrics[inflow_type].keys())
    y_vals = [metrics[inflow_type][weight]['max_delay'] for weight in x_vals]
    ax_max_delay.plot(x_vals, y_vals, marker='o', label=inflow_type)
ax_max_delay.set_xlabel('Weight')
ax_max_delay.set_ylabel('Max Delay Time')
ax_max_delay.set_title('Max Delay Time vs Weight')
ax_max_delay.legend(title='Inflow Type')

fig_average_delay, ax_average_delay = plt.subplots()
for inflow_type in inflow_types:
    x_vals = list(metrics[inflow_type].keys())
    y_vals = [metrics[inflow_type][weight]['average_delay'] for weight in x_vals]
    ax_average_delay.plot(x_vals, y_vals, marker='o', label=inflow_type)
ax_average_delay.set_xlabel('Weight')
ax_average_delay.set_ylabel('Average Delay Time')
ax_average_delay.set_title('Average Delay Time vs Weight')

fig_calc_time, ax_calc_time = plt.subplots()
for inflow_type in inflow_types:
    x_vals = list(metrics[inflow_type].keys())
    y_vals = [metrics[inflow_type][weight]['calculation_time'] for weight in x_vals]
    ax_calc_time.plot(x_vals, y_vals, marker='o', label=inflow_type)
ax_calc_time.set_xlabel('Weight')
ax_calc_time.set_ylabel('Calculation Time')
ax_calc_time.set_title('Calculation Time vs Weight')
ax_calc_time.legend(title='Inflow Type')

plt.show()





    
            




