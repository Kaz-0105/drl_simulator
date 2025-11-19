from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# 見たい結果を選択
drl_method = 'apex'
model_ids = [9]

# 指数移動平均を使うか
ema_flg = True

# ファイル名を指定してデータを読み込む
session_paths = {}
for model_id in model_ids:
    tmp_session_path = Path('results/session/drl') / drl_method / f'session_{model_id}' / 'session.pkl'
    if not tmp_session_path.exists():
        raise FileNotFoundError(f"Session file {tmp_session_path} does not exist.")
    session_paths[model_id] = tmp_session_path

session_data = {}
for model_id, session_path in session_paths.items():
    with open(session_path, 'rb') as f:
        session_data[model_id] = pickle.load(f)
        session_data[model_id]['episode_record'] = list(range(1, len(session_data[model_id]['total_reward_record']) + 1))

# episodeごとのtotal_rewardの推移を描画
alpha = 0.1 if ema_flg else 1.0
ema = 0
for model_id in model_ids:
    total_reward_record = []
    for idx, total_reward in enumerate(session_data[model_id]['total_reward_record']):
        ema = alpha * total_reward + (1 - alpha) * ema if idx > 0 else total_reward
        total_reward_record.append(ema)
    
    simulation_time_record = session_data[model_id]['simulation_time_record']
    for idx in range(len(total_reward_record)):
        total_reward_record[idx] *= 500 / simulation_time_record[idx]
    session_data[model_id]['total_reward_record'] = total_reward_record


fig_total_reward, ax_total_reward = plt.subplots()
for model_id in model_ids:
    ax_total_reward.plot(session_data[model_id]['episode_record'], session_data[model_id]['total_reward_record'], linewidth=4, label=f'Model {model_id}')
ax_total_reward.set_xlabel('Episode', fontsize=20)
ax_total_reward.set_ylabel('Total Reward', fontsize=20)
ax_total_reward.set_title('Total Reward Record (EMA)' if ema_flg else 'Total Reward Record', fontsize=24)
ax_total_reward.tick_params(axis='both', which='major', labelsize=20)
ax_total_reward.tick_params(axis='both', which='minor', labelsize=20)
ax_total_reward.legend(fontsize=16)

# epsilonの推移を描画
fig_epsilon, ax_epsilon = plt.subplots()
for model_id in model_ids:
    ax_epsilon.plot(session_data[model_id]['episode_record'], session_data[model_id]['epsilon_record'], linewidth=4, label=f'Model {model_id}')
ax_epsilon.set_xlabel('Episode', fontsize=20)
ax_epsilon.set_ylabel('Epsilon', fontsize=20)
ax_epsilon.set_title('Epsilon Record', fontsize=24)
ax_epsilon.tick_params(axis='both', which='major', labelsize=20)
ax_epsilon.tick_params(axis='both', which='minor', labelsize=20)
ax_epsilon.legend(fontsize=16)

# num_epochsの推移を描画
fig_num_epochs, ax_num_epochs = plt.subplots()
for model_id in model_ids:
    ax_num_epochs.plot(session_data[model_id]['episode_record'], session_data[model_id]['num_epochs_record'], linewidth=4, label=f'Model {model_id}')
ax_num_epochs.set_xlabel('Episode', fontsize=20)
ax_num_epochs.set_ylabel('Num Epochs', fontsize=20)
ax_num_epochs.set_title('Num Epochs Record', fontsize=24)
ax_num_epochs.tick_params(axis='both', which='major', labelsize=20)
ax_num_epochs.tick_params(axis='both', which='minor', labelsize=20)
ax_num_epochs.legend(fontsize=16)

# update_intervalの推移を描画
fig_update_interval, ax_update_interval = plt.subplots()
for model_id in model_ids:
    ax_update_interval.plot(session_data[model_id]['episode_record'], session_data[model_id]['update_interval_record'], linewidth=4, label=f'Model {model_id}')
ax_update_interval.set_xlabel('Episode', fontsize=20)
ax_update_interval.set_ylabel('Update Interval', fontsize=20)
ax_update_interval.set_title('Update Interval Record', fontsize=24)
ax_update_interval.tick_params(axis='both', which='major', labelsize=20)
ax_update_interval.tick_params(axis='both', which='minor', labelsize=20)
ax_update_interval.legend(fontsize=16)

# グラフを表示
plt.show()
