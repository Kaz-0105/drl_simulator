import numpy as np
import matplotlib.pyplot as plt

# みたい結果を選択
network_id = 1
roads_str = '2222'
num_vehicles = 10

# ファイル名を指定してデータを読み込む
rewards_record = np.load(f"results/rewards_record_{network_id}_{roads_str}_{num_vehicles}.npy", allow_pickle=True).tolist()

# 描画
plt.plot(rewards_record, linewidth=2)
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.title('Rewards Over Episodes', fontsize=16)
plt.grid()

plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()
