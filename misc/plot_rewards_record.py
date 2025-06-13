import numpy as np
import matplotlib.pyplot as plt

rewards_record = np.load('results/rewards_record1_3333_10.npy', allow_pickle=True).tolist()

plt.plot(rewards_record)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards Over Episodes')
plt.grid()
plt.show()

print('test')



