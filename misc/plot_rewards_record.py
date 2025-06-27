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

# 描画
plt.plot(total_reward_record, linewidth=2)
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.title('Rewards Over Episodes', fontsize=16)
plt.grid()

plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()
