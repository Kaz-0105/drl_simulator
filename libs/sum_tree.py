import numpy as np
import math


class SumTree:
    def __init__(self, capacity):
        # ツリーに格納するデータ数の最大値を設定
        self.capacity = capacity
        self.actual_capacity = 2**math.ceil(math.log2(capacity)) # 完全二分木のサイズに調整

        # ツリーのノードの重みを保存する配列を定義
        self.tree = np.zeros(2 * self.actual_capacity - 1, dtype=np.float32)

        # 経験データ自体を格納する配列を定義
        self.data = np.zeros(self.capacity, dtype=object)

        # 次に経験を格納する位置
        self.next_data_idx = 0

        # 現在格納されているデータの数
        self.current_size = 0

        # 初期優先度で参照する過去のデータの数
        self.initial_priority_data_count = 20

    def load(self, file_path):
        # ファイルからツリーとデータを読み込む
        loaded_data = np.load(file_path, allow_pickle=True)
        self.tree = loaded_data['tree']
        self.data = loaded_data['data']
        self.next_data_idx = loaded_data['next_data_idx'].item()
        self.current_size = loaded_data['current_size'].item()
    
    def save(self, file_path):
        # ツリーとデータをファイルに保存
        np.savez(file_path, tree=self.tree, data=self.data, next_data_idx=self.next_data_idx, current_size=self.current_size)

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
        
    @property
    def total_priority(self):
        return self.tree[0]
    
    def add(self, data, priority = None):
        # 優先度が指定されていない場合は直近のデータの平均をつかう
        if priority is None:
            priority = self.initial_priority

        # 優先度を更新するツリーのインデックスを計算
        tree_idx = self.next_data_idx + self.actual_capacity - 1

        # 差分を計算後に葉ノードの値を更新
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # 親ノードの値を順に更新
        self._propagate(tree_idx, change)

        # データを保存
        self.data[self.next_data_idx] = data

        if self.current_size < self.capacity:
            # バッファに空きがある場合はサイズと次のデータ格納位置をインクリメント
            self.current_size += 1
            self.next_data_idx = (self.next_data_idx + 1) % self.capacity
        else:
            # バッファが満タンの場合は，循環ができるようにする
            self.next_data_idx = (self.next_data_idx + 1) % self.capacity
    
    def sample(self, batch_size):
        # ランダムに0-total_priorityの範囲でサンプリング
        sample_values = np.random.uniform(0, self.total_priority, batch_size)

        # 対応するデータのツリーのインデックスを取得
        tree_indices = [self._retrieve(0, sample_value) for sample_value in sample_values]

        # データのインデックスを取得
        data_indices = [tree_idx - self.actual_capacity + 1 for tree_idx in tree_indices]

        # ツリーのインデックスからデータを取得
        valid_data = []
        valid_data_indices = []
        for data_idx in data_indices:
            if data_idx < self.current_size:
                valid_data.append(self.data[data_idx])
                valid_data_indices.append(data_idx)

        return valid_data, valid_data_indices
    
    def update_priority(self, data_indices, new_priorities):
        for data_idx, new_priority in zip(data_indices, list(new_priorities)):
            # validation
            if data_idx < 0 or data_idx >= self.current_size:
                continue
            
            # ツリーのインデックスを計算
            tree_idx = data_idx + self.actual_capacity - 1

            # 差分を計算後に葉ノードの値を更新
            change = new_priority - self.tree[tree_idx]
            self.tree[tree_idx] = new_priority

            # 親ノードの値を順に更新
            self._propagate(tree_idx, change)
        
    @property
    def initial_priority(self):
        # 最新の20個のデータの優先度を平均して初期優先度とする
        if self.current_size == 0:
            return 1
        
        if self.current_size < self.initial_priority_data_count:
            return np.mean(self.tree[self.actual_capacity - 1: self.actual_capacity - 1 + self.current_size])
        else:
            data_indices = []
            for idx in range(1, self.initial_priority_data_count + 1):
                data_indices.append((self.next_data_idx - idx) % self.capacity)
            
            tree_indices = np.array(data_indices) + self.actual_capacity - 1

            return np.mean(self.tree[tree_indices])




            

        


    



