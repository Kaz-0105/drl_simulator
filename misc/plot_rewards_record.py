import numpy as np
import matplotlib.pyplot as plt

rewards_record = np.load('results/rewards_record1_3333_10.npy', allow_pickle=True).tolist()

plt.plot(rewards_record, linewidth=2)
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.title('Rewards Over Episodes', fontsize=16)
plt.grid()

# 線を太くする
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()

print('test')



