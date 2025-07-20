from libs.common import Common
import numpy as np
import random
import math

class SumTree (Common):
    def __init__(self, capacity, initial_priority):
        # 継承
        super().__init__()
        
        # ツリーに格納するデータ数の最大値を設定
        self.capacity = capacity
        self.num_leaves = 2**math.ceil(math.log2(capacity)) # 完全二分木のサイズに調整

        # ツリーのノードの重みを保存する配列を定義
        self.tree = np.zeros(2 * self.num_leaves - 1, dtype=np.float32)

        # 経験データ自体を格納する配列を定義
        self.data = [None] * self.capacity

        # 初期優先度を初期化
        self.initial_priority = initial_priority

        # 現在のデータ数を初期化
        self.current_size = 0

        # 次に経験を格納する位置
        self.next_data_idx = 0
        return

    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change

        if parent != 0: 
            self._propagate(parent, change)
    
    def _retrieve(self, tree_idx, random_value):
        left_child = 2 * tree_idx + 1
        right_child = left_child + 1

        if left_child >= len(self.tree):
            return tree_idx

        if random_value <= self.tree[left_child]:
            return self._retrieve(left_child, random_value)
        else:
            return self._retrieve(right_child, random_value - self.tree[left_child])
    
    def add(self, tmp_data, priority = None):
        # 優先度が指定されていない場合は直近のデータの平均をつかう
        if priority is None:
            priority = self.initial_priority

        # 優先度を更新するツリーのインデックスを計算
        tree_idx = self.next_data_idx + self.num_leaves - 1

        # 差分を計算後に葉ノードの値を更新
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # 親ノードの値を順に更新
        self._propagate(tree_idx, change.item())

        # データを保存
        self.data[self.next_data_idx] = tmp_data

        # 現在のデータ数を更新
        if self.current_size < self.capacity:
            self.current_size += 1

        # 次にデータを格納する位置を更新
        self.next_data_idx += 1
        self.next_data_idx %= self.capacity
        return
    
    def sample(self, size):
        # サンプリングサイズが現在のサイズより大きい場合は，全データを返す
        if self.current_size < size:
            data = self.data[:self.current_size]
            data_indices = list(range(self.current_size))
            return data, data_indices

        # ランダムに0-total_priorityの範囲でサンプリング
        sample_values = np.random.uniform(0, self.total_priority, size)

        # 対応するデータのツリーのインデックスを取得
        tree_indices = [self._retrieve(0, sample_value) for sample_value in sample_values]

        # データのインデックスを取得
        data_indices = []
        for tree_idx in tree_indices:
            data_idx = tree_idx - self.num_leaves + 1
            if data_idx + 1 > self.current_size:
                data_idx = random.randint(0, self.current_size - 1)
            data_indices.append(data_idx)

        # ツリーのインデックスからデータを取得
        data = []
        for data_idx in data_indices:
            data.append(self.data[data_idx])


        return data, data_indices
    
    def update_priority(self, data_indices, new_priorities):
        for data_idx, new_priority in zip(data_indices, list(new_priorities)):
            # validation
            if data_idx < 0 or data_idx >= self.current_size:
                continue
            
            # ツリーのインデックスを計算
            tree_idx = data_idx + self.num_leaves - 1

            # 差分を計算後に葉ノードの値を更新
            change = new_priority.item() - self.tree[tree_idx].item()
            self.tree[tree_idx] = new_priority.item()

            # 親ノードの値を順に更新
            self._propagate(tree_idx, change)

    def resetPriority(self):
        for data_idx in range(self.current_size):
            tree_idx = data_idx + self.num_leaves - 1
            change = self.initial_priority - self.tree[tree_idx]
            self.tree[tree_idx] = self.initial_priority

            self._propagate(tree_idx, change.item())
        
        return
    
    @property
    def total_priority(self):
        return self.tree[0]




            

        


    



