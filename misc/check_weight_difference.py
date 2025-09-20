from pathlib import Path
from collections import defaultdict
import pickle
import statistics
import matplotlib.pyplot as plt

# 比較する重みのリスト
weights = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20]

# 流入量のタイプ
inflow_type = 'balanced'

# 道路の構成
road_layout = '2222'

# 重みを昇順にソート
weights.sort()

# mpcのデータを取得
metrics = {}
for weight in weights:
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
    metrics[weight] = tmp_weight_metrics

# グラフの描画
fig_max_queue, ax_max_queue = plt.subplots()
x_vals = list(metrics.keys())
y_vals = [metrics[w]['max_queue'] for w in x_vals]
ax_max_queue.plot(x_vals, y_vals, marker='o')
ax_max_queue.set_xlabel('Weight')
ax_max_queue.set_ylabel('Max Queue Length')
ax_max_queue.set_title('Max Queue Length vs Weight')

fig_average_queue, ax_average_queue = plt.subplots()
x_vals = list(metrics.keys())
y_vals = [metrics[w]['average_queue'] for w in x_vals]
ax_average_queue.plot(x_vals, y_vals, marker='o')
ax_average_queue.set_xlabel('Weight')
ax_average_queue.set_ylabel('Average Queue Length')
ax_average_queue.set_title('Average Queue Length vs Weight')

plt.show()





    
            




