from pathlib import Path
from collections import defaultdict
import pickle
import statistics
import matplotlib.pyplot as plt
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

# set root_dir_path
root_dir_path = (Path(__file__).parent / '..').resolve()

# 出力に関する設定
config = {
    'road_layout': '2222',   # '2222' = 各道路の車線が1車線＋分岐車線の合計2車線，'3333' = 各道路の車線が2車線＋分岐車線の合計3車線
    'figure_flgs': {
        'max_queue': True,
        'average_queue': True,
        'max_delay': True,
        'average_delay': True,
        'calculation_time': True,
    },
    'inflow_types': {
        'balanced_500': True,
        'balanced_600': True,
        'balanced_700' : True,
        'balanced_800' : True,
        'unbalanced' : False,
        'main-minor' : False,
    },
    'compare_to' : 'all' # 1. 'all': 全てのdemand_typeを1つのグラフで比較, 2. 'each': 各demand_typeごとに比較（scootとdrlの結果も基準線として表示）
}

# 測定した重みについて（実験をおこなったら追加）
weights = {
    '2222' : {
        'balanced_500': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'balanced_600': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'balanced_700' : [0, 2, 4, 6, 8, 10, 12, 14, 18, 20],
        'balanced_800' : [6, 8, 10, 12, 14, 16, 18, 20],
        'unbalanced' : [6, 8, 10, 12, 14, 16, 20],
        'main-minor' : [8, 10, 12, 14, 16],
    },
    '3333' : {
        'balanced_low': [],
        'balanced' : [],
        'unbalanced' : [],
        'main-minor' : [],
    }
}

# 重みを昇順にソート
for inflow_type, tmp_weights in weights[config['road_layout']].items():
    tmp_weights.sort()

# mpcのデータを取得
metrics = defaultdict(dict)
for inflow_type, tmp_weights in weights[config['road_layout']].items():
    for weight in tmp_weights:
        # 結果の保存されているディレクトリを取得
        simulation_dir = Path.cwd() / 'results' / 'metrics' / 'mpc' / f"{inflow_type}_{config['road_layout']}_{weight}"
        if not simulation_dir.exists():
            raise FileNotFoundError(f"Directory not found: {simulation_dir}")

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
                
                if saved_data['max_queue'] is not None and config['figure_flgs']['max_queue']:
                    max_queue_record = saved_data['max_queue']
                    tmp_weight_metrics['max_queue'].append(max_queue_record['queue_length'].max())

                if saved_data['average_queue'] is not None and config['figure_flgs']['average_queue']:
                    average_queue_record = saved_data['average_queue']
                    tmp_weight_metrics['average_queue'].append(average_queue_record['queue_length'].mean())

                if saved_data['max_delay'] is not None and config['figure_flgs']['max_delay']:
                    max_delay_record = saved_data['max_delay']
                    tmp_weight_metrics['max_delay'].append(max_delay_record['delay'].max())
                
                if saved_data['average_delay'] is not None and config['figure_flgs']['average_delay']:
                    average_delay_record = saved_data['average_delay']
                    tmp_weight_metrics['average_delay'].append(average_delay_record['delay'].mean())

                if saved_data['calc_time'] is not None and config['figure_flgs']['calculation_time']:
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

if config['compare_to'] == 'all':
    # グラフの描画
    figs = {}
    axes = {}
    for metric_name, metric_flg in config['figure_flgs'].items():
        if not metric_flg:
            continue

        figs[metric_name], axes[metric_name] = plt.subplots()
        for inflow_type in metrics.keys():
            if not config['inflow_types'][inflow_type]:
                continue

            x_vals = list(metrics[inflow_type].keys())
            y_vals = [metrics[inflow_type][weight][metric_name] for weight in x_vals]
            axes[metric_name].plot(
                x_vals, 
                y_vals, 
                marker='o', 
                label=inflow_type,
                linewidth=4
            )
        
        axes[metric_name].set_xlabel('Weight', fontsize=20, fontweight='bold')
        axes[metric_name].set_ylabel(metric_name.replace('_', ' ').title(), fontsize=20, fontweight='bold')
        axes[metric_name].tick_params(axis='both', which='major', labelsize=20)
        axes[metric_name].tick_params(axis='both', which='minor', labelsize=20)
        axes[metric_name].set_title(f'{metric_name.replace("_", " ").title()} vs Weight', fontsize=24, fontweight='bold')
        axes[metric_name].grid(
            which='major',
            axis='both',
            linestyle='--',
            linewidth=1,
            alpha=0.5,
        )
        axes[metric_name].legend(
            title='Inflow Type',
            fontsize=20,
            title_fontsize=20,
        )

        figs[metric_name].tight_layout()

        save_dir_path = root_dir_path / 'results' / 'plots' / 'weight_comparison'
        save_dir_path.mkdir(parents=True, exist_ok=True)
        save_file_path = save_dir_path / f"weight_comparison_{metric_name}.png"
        figs[metric_name].savefig(save_file_path)

elif config['compare_to'] == 'each':
    # scootとdrlのデータを取得
    compare_metrics = defaultdict(dict)

    for method in ['scoot', 'drl']:
        for inflow_type in weights[config['road_layout']].keys():
            # inflow_typeが有効でなければスキップ
            if not config['inflow_types'][inflow_type]:
                continue

            # 結果の保存されているディレクトリを取得
            simulation_dir = Path.cwd() / 'results' / 'metrics' / method / f"{inflow_type}_{config['road_layout']}"
            if not simulation_dir.exists():
                continue

            # inflow_typeごとの測定値を格納する辞書を初期化
            tmp_metrics = defaultdict(list)
            
            # demand_typeを走査
            demand_type_id = 0
            while True:
                demand_type_id += 1
                tmp_file_path = simulation_dir / f"metric_{demand_type_id}.pkl"

                if not tmp_file_path.exists():
                    break

                with open(tmp_file_path, 'rb') as f:
                    saved_data = pickle.load(f)

                    if saved_data['max_queue'] is not None and config['figure_flgs']['max_queue']:
                        max_queue_record = saved_data['max_queue']
                        tmp_metrics['max_queue'].append(max_queue_record['queue_length'].max())

                    if saved_data['average_queue'] is not None and config['figure_flgs']['average_queue']:
                        average_queue_record = saved_data['average_queue']
                        tmp_metrics['average_queue'].append(average_queue_record['queue_length'].mean())

                    if saved_data['max_delay'] is not None and config['figure_flgs']['max_delay']:
                        max_delay_record = saved_data['max_delay']
                        tmp_metrics['max_delay'].append(max_delay_record['delay'].max())

                    if saved_data['average_delay'] is not None and config['figure_flgs']['average_delay']:
                        average_delay_record = saved_data['average_delay']
                        tmp_metrics['average_delay'].append(average_delay_record['delay'].mean())

                    if saved_data['calc_time'] is not None and config['figure_flgs']['calculation_time']:
                        calc_time_record = saved_data['calc_time']
                        tmp_metrics['calculation_time'].append(calc_time_record['calculation_time'].mean())

            # 旋回率の異なるシミュレーションの評価値を1つに集約
            if 'max_queue' in tmp_metrics:
                tmp_metrics['max_queue'] = max(tmp_metrics['max_queue'])
            if 'average_queue' in tmp_metrics:
                tmp_metrics['average_queue'] = statistics.mean(tmp_metrics['average_queue'])
            if 'max_delay' in tmp_metrics:
                tmp_metrics['max_delay'] = max(tmp_metrics['max_delay'])
            if 'average_delay' in tmp_metrics:
                tmp_metrics['average_delay'] = statistics.mean(tmp_metrics['average_delay'])
            if 'calculation_time' in tmp_metrics:
                tmp_metrics['calculation_time'] = statistics.mean(tmp_metrics['calculation_time'])

            # tmp_metricsをcompare_metricsにプッシュ
            compare_metrics[method][inflow_type] = tmp_metrics

    # グラフの描画
    figs = {}
    axes = {}
    for metric_name, metric_flg in config['figure_flgs'].items():
        if not metric_flg:
            continue
        
        figs[metric_name] = {}
        axes[metric_name] = {}

        for inflow_type in metrics.keys():
            if not config['inflow_types'][inflow_type]:
                continue

            figs[metric_name][inflow_type], axes[metric_name][inflow_type] = plt.subplots()

            x_vals = list(metrics[inflow_type].keys())
            y_vals = [metrics[inflow_type][weight][metric_name] for weight in x_vals] 
            axes[metric_name][inflow_type].plot(x_vals, y_vals, marker='o', label='mpc')

            if 'scoot' in compare_metrics and inflow_type in compare_metrics['scoot'] and metric_name != 'calculation_time':
                scoot_val = compare_metrics['scoot'][inflow_type][metric_name]
                axes[metric_name][inflow_type].axhline(y=scoot_val, color='r', linestyle='--', label='scoot')
            
            if 'drl' in compare_metrics and inflow_type in compare_metrics['drl']:
                drl_val = compare_metrics['drl'][inflow_type][metric_name]
                axes[metric_name][inflow_type].axhline(y=drl_val, color='g', linestyle='--', label='drl')
            
            axes[metric_name][inflow_type].set_xlabel('Weight')
            axes[metric_name][inflow_type].set_ylabel(metric_name.replace('_', ' ').title())
            axes[metric_name][inflow_type].set_title(f'{metric_name.replace("_", " ").title()} vs Weight ({inflow_type})')
            axes[metric_name][inflow_type].legend(title='Method')

            figs[metric_name][inflow_type].tight_layout()
            figs[metric_name][inflow_type].savefig(root_dir_path / 'results' / 'plots' / f'weight_comparison_{metric_name}_{inflow_type}.png')

print('Finished')





    
            




