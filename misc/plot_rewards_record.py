from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# みたい結果を選択
network_id = 1
roads_str = '2222'
num_vehicles = 5

# ファイル名を指定してデータを読み込む
session_path = Path(f"results/session_{network_id}_{roads_str}_{num_vehicles}.pkl")
if not session_path.exists():
    raise FileNotFoundError(f"Session file {session_path} does not exist.")

with open(session_path, 'rb') as f:
    session_data = pickle.load(f)
    total_reward_record = session_data['total_reward_record']

# EMAの設定
alpha = 0.5

# EMAを計算
ema_rewards = []
ema = 0
for idx, total_reward in enumerate(total_reward_record):
    ema = alpha * total_reward + (1 - alpha) * ema if idx > 0 else total_reward
    ema_rewards.append(ema)

# 描画
plt.plot(ema_rewards, linewidth=2)
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Total Reward', fontsize=14)
plt.title('Total Rewards over Episodes', fontsize=16)
plt.grid()

plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()
