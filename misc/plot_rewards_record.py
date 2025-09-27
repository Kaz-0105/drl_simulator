from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# 見たい結果を選択
network_id = 1
roads_str = '2222'
num_vehicles = 5

# 指数移動平均のパラメータ設定
alpha = 0.1

# ファイル名を指定してデータを読み込む
session_path = Path(f"results/session_{network_id}_{roads_str}_{num_vehicles}.pkl")
if not session_path.exists():
    raise FileNotFoundError(f"Session file {session_path} does not exist.")

with open(session_path, 'rb') as f:
    session_data = pickle.load(f)
    total_reward_record = session_data['total_reward_record']
    epsilon_record = session_data['epsilon_record']
    num_epochs_record = session_data['num_epochs_record']
    update_interval_record = session_data['update_interval_record']

# episodeごとのtotal_rewardの推移を描画
fig_total_reward, ax_total_reward = plt.subplots()
ema_rewards = []
ema = 0
for idx, total_reward in enumerate(total_reward_record):
    ema = alpha * total_reward + (1 - alpha) * ema if idx > 0 else total_reward
    ema_rewards.append(ema)

ax_total_reward.plot(ema_rewards, linewidth=4)
ax_total_reward.set_xlabel('Episode', fontsize=20)
ax_total_reward.set_ylabel('Total Reward', fontsize=20)
ax_total_reward.set_title('Total Reward over Episodes (EMA)' if alpha != 1 else 'Total Reward over Episodes', fontsize=24)
ax_total_reward.tick_params(axis='both', which='major', labelsize=20)
ax_total_reward.tick_params(axis='both', which='minor', labelsize=20)

# epsilonの推移を描画
fig_epsilon, ax_epsilon = plt.subplots()
ax_epsilon.plot(epsilon_record, linewidth=4)
ax_epsilon.set_xlabel('Episode', fontsize=20)
ax_epsilon.set_ylabel('Epsilon', fontsize=20)
ax_epsilon.set_title('Epsilon over Episodes', fontsize=24)
ax_epsilon.tick_params(axis='both', which='major', labelsize=20)
ax_epsilon.tick_params(axis='both', which='minor', labelsize=20)

# num_epochsの推移を描画
fig_num_epochs, ax_num_epochs = plt.subplots()
ax_num_epochs.plot(num_epochs_record, linewidth=4)
ax_num_epochs.set_xlabel('Episode', fontsize=20)
ax_num_epochs.set_ylabel('Num Epochs', fontsize=20)
ax_num_epochs.set_title('Num Epochs over Episodes', fontsize=24)
ax_num_epochs.tick_params(axis='both', which='major', labelsize=20)
ax_num_epochs.tick_params(axis='both', which='minor', labelsize=20)

# update_intervalの推移を描画
fig_update_interval, ax_update_interval = plt.subplots()
ax_update_interval.plot(update_interval_record, linewidth=4)
ax_update_interval.set_xlabel('Episode', fontsize=20)
ax_update_interval.set_ylabel('Update Interval', fontsize=20)
ax_update_interval.set_title('Update Interval over Episodes', fontsize=24)
ax_update_interval.tick_params(axis='both', which='major', labelsize=20)
ax_update_interval.tick_params(axis='both', which='minor', labelsize=20)

# グラフを表示
plt.show()
