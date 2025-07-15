from libs.container import Container
from libs.object import Object
from objects.replay_buffer import ReplayBuffer
from objects.intersections import Intersections
from objects.local_agents import LocalAgents
from neural_networks.q_net_1 import QNet1

from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import math
import numpy as np

class MasterAgents(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトを取得
        self.config = network.config
        self.executor = network.executor
        self.shared_resources = network.shared_resources

        # 上位の紐づくオブジェクトを取得
        self.network = network

        # 要素オブジェクトを初期化
        self._makeElements()
    
    def _makeElements(self):
        # intersectionsオブジェクトを取得
        intersections = self.network.intersections
        self.intersections_map = {}
        for intersection in intersections.getAll():
            # 車線数のリストを取得
            num_lanes_turple = intersection.getNumLanesTurple()

            if num_lanes_turple not in self.intersections_map:
                # 車線数のリストをキーにしてMasterAgentオブジェクトを初期化
                self.intersections_map[num_lanes_turple] = []
            
            self.intersections_map[num_lanes_turple].append(intersection)
        
        for num_lanes_turple in self.intersections_map.keys():
            # master_agentオブジェクトを初期化
            self.add(MasterAgent(self, num_lanes_turple))

    def saveLearningData(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.saveLearningData)
        
        self.executor.wait()
    
    def train(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.train)
        
        self.executor.wait()
    
    def updateSessionData(self):
        # トータルの報酬のレコードを更新
        for master_agent in self.getAll():
            self.executor.submit(master_agent.updateSessionData)
        self.executor.wait()

        # 結果を表示
        for master_agent_id in self.getKeys(container_flg=True, sorted_flg=True):
            master_agent = self[master_agent_id]
            master_agent.showTotalReward()
        return
    
    def saveModel(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.saveModel)
        
        self.executor.wait()
        return
    
    def saveSession(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.saveSession)
        
        self.executor.wait()
        return

class MasterAgent(Object):
    def __init__(self, master_agents, num_lanes_turple):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = master_agents.config
        self.executor = master_agents.executor
        self.shared_resources = master_agents.shared_resources

        # 上位オブジェクトを取得
        self.master_agents = master_agents
        self.network = master_agents.network

        # IDを設定
        self.id = self.master_agents.count() + 1

        # intersectionsオブジェクトと紐づける
        self._makeIntersectionConnections(num_lanes_turple)

        # 車線数の情報を取得
        self._makeNumLanesMap(num_lanes_turple)

        # 強化学習で共通のパラメータを取得
        self._getDrlParameters()

        # Apexのパラメータを取得
        self._getApeXParameters()

        # 保存先のパスを定義
        self._makePaths()

        # モデルを初期化
        self._makeModel()
        
        # 前回までのセッションを読み込む
        self._loadSession()
        self._loadModel()
        
        # LocalAgentオブジェクトを初期化
        self.local_agents = LocalAgents(self)

        # 最適化手法と評価関数を定義
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return

    def _makeIntersectionConnections(self, num_lanes_turple):
        # intersection_listを取得
        intersection_list = self.master_agents.intersections_map[num_lanes_turple]

        # intersectionsオブジェクトを初期化
        self.intersections = Intersections(self)

        # intersectionオブジェクトと紐づける
        for intersection in intersection_list:
            self.intersections.add(intersection)
            intersection.set('master_agent', self)

    def _makeNumLanesMap(self, num_lanes_turple):
        # 車線数のリストを少し整形
        num_lanes_map = {}
        for num_lanes in num_lanes_turple:
            num_lanes_map[len(num_lanes_map) + 1] = num_lanes

        self.num_lanes_map = num_lanes_map

    def _getDrlParameters(self):
        drl_info = self.config.get('drl_info')
        self.drl_method = drl_info['method']
        self.num_vehicles = drl_info['num_vehicles']
        self.network_id = drl_info['network_id']
        self.bc_flg = drl_info['bc_flg']
        return
    
    def _getApeXParameters(self):
        apex_info = self.config.get('apex_info')
        self.td_steps = apex_info['td_steps']
        self.update_interval = apex_info['update_interval']
        self.weight_decay = apex_info['weight_decay']
        self.gamma = apex_info['gamma']
        self.learning_rate = apex_info['learning_rate']
        self.num_epochs = apex_info['num_epochs']
        self.epsilon = apex_info['epsilon']
        
        return
    
    def _makePaths(self):
        # 車線情報を文字列に変換
        lanes_str = ''
        for num_lanes in self.num_lanes_map.values():
            lanes_str += str(num_lanes)

        self.path_map = {
            'model': Path(f"models/q_net_{self.network_id}_{lanes_str}_{self.num_vehicles}.pth"),
            'target_model': Path(f"models/target_q_net_{self.network_id}_{lanes_str}_{self.num_vehicles}.pth"),
            'session': Path(f"results/session_{self.network_id}_{lanes_str}_{self.num_vehicles}.pkl"),
        }

        # リプレーバッファーについて
        buffer_size = self.config.get('apex_info')['buffer']['size']
        self.path_map['replay_buffer'] = {
            'tree': Path(f"buffers/replay_buffer_tree_{self.network_id}_{lanes_str}_{self.num_vehicles}.pkl"),
            'data': [Path(f"buffers/replay_buffer_data_{self.network_id}_{lanes_str}_{self.num_vehicles}_{idx + 1}.pkl") for idx in range(math.ceil(buffer_size / 1000))],
        }

        if self.bc_flg:
            self.path_map['bc_model'] = Path(f"models/bc_q_net_{self.network_id}_{lanes_str}_{self.num_vehicles}.pth")

        return
        
    def _makeModel(self):
        # モデルを初期化（学習用にセット）
        if self.network_id == 1:
            self.model = QNet1(self.config, self.num_vehicles, self.num_lanes_map)
        self.model.train()

        # ターゲットモデルを初期化（学習用と同期，推論用にセット）
        if self.network_id == 1:
            self.target_model = QNet1(self.config, self.num_vehicles, self.num_lanes_map)
        self.target_model.eval()
        return

    def _loadSession(self):
        # リプレイバッファーを取得
        self.replay_buffer = ReplayBuffer(self)

        # update_countとtotal_reward_recordを取得
        self.update_count = 0
        self.total_reward_record = []
        self.update_interval_record = []
        self.num_data_for_learning_record = []
        self.batch_record = {
            'number': [],
            'size': [],
        }
        self.num_epochs_record = []
        self.learning_rate_record = []
        self.weight_decay_record = []
        self.epsilon_record = []

        if self.path_map['session'].exists():
            with self.path_map['session'].open('rb') as f:
                loaded_data = pickle.load(f)
                self.update_count = loaded_data['update_count'] 
                self.total_reward_record = loaded_data['total_reward_record']
                self.update_interval_record = loaded_data['update_interval_record']
                self.num_data_for_learning_record = loaded_data['num_data_for_learning_record']
                self.batch_record = loaded_data['batch_record']
                self.num_epochs_record = loaded_data['num_epochs_record']
                self.learning_rate_record = loaded_data['learning_rate_record']
                self.weight_decay_record = loaded_data['weight_decay_record']
                self.epsilon_record = loaded_data['epsilon_record']
        return

    def _loadModel(self):
        # メインのモデルを読み込む
        if self.path_map['model'].exists():
            self.model.load_state_dict(torch.load(self.path_map['model']))
        elif self.bc_flg and self.path_map['bc_model'].exists():
            self.model.load_state_dict(torch.load(self.path_map['bc_model']))
        
        # ターゲットモデルを読み込む
        if not self.path_map['target_model'].exists() or self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.target_model.load_state_dict(torch.load(self.path_map['target_model']))
        return
    
    def saveLearningData(self):
        # ローカルエージェントを走査
        self.buffer_change_flg = False
        for local_agent in self.local_agents.getAll():
            # 学習データを取得
            learning_data = local_agent.get('learning_data')
            
            # 学習データがない場合はスキップ
            if not learning_data:
                continue

            # バッファーにデータを保存
            self.replay_buffer.push(learning_data)

            # データをクリア
            local_agent.set('learning_data', [])

            if not self.buffer_change_flg:
                # バッファーのサイズが変化した場合はフラグを立てる
                self.buffer_change_flg = True
        return
    
    def train(self):
        if not self.buffer_change_flg:
            # バッファーのサイズが変化していない場合は学習しない
            return
        
        if not self.replay_buffer.get('should_learn_flg'):
            return
        
        # バッファーのサイズが十分でない場合は学習しない
        if self.replay_buffer.get('current_size') < self.replay_buffer.get('batch_size') * self.replay_buffer.get('num_batches'):
            return

        # バッファーからデータを取得
        batch_data = self.replay_buffer.sample()
        for epoch in range(self.num_epochs):
            losses = []
            for data, data_indices in batch_data:
                # 勾配を初期化
                self.optimizer.zero_grad()
                
                if self.network_id == 1:
                    # とった行動をテンソルに変換
                    actions = torch.tensor([tmp_data[1] - 1 for tmp_data in data], dtype=torch.int64).unsqueeze(1)

                    # 状態を配列にする
                    states = [tmp_data[0] for tmp_data in data]

                    # 勾配をトラッキングするように設定
                    self.model.set('requires_grad', True)

                    # Q値を計算し，選ばれた行動のQ値を取得
                    q_values = self.model(states).gather(1, actions)

                    # メインモデルを評価モードに設定
                    self.model.eval()

                    # TDターゲットを計算するアルゴリズムここから
                    with torch.no_grad():
                        # 次の状態を配列にする
                        states_next = [tmp_data[3] for tmp_data in data]

                        # 勾配をトラッキングしないように設定
                        self.model.set('requires_grad', False)

                        # 次の状態のメインモデルのQ値の最大値を与える行動を取得
                        max_actions = torch.argmax(self.model(states_next), dim=1).unsqueeze(1)

                        # ターゲットモデルのQ値を取得
                        target_q_values = self.target_model(states_next).gather(1, max_actions)

                        # 累積報酬をテンソルに変換（multi step bootstrap を実装している）
                        cumurative_rewards = torch.tensor([tmp_data[2] for tmp_data in data], dtype=torch.float32).unsqueeze(1)

                        # 終了フラグをテンソルに変換
                        dones = torch.tensor([tmp_data[4] for tmp_data in data], dtype=torch.float32).unsqueeze(1)

                        # TDターゲットを計算
                        td_targets = cumurative_rewards + (1 - dones) * (self.gamma ** self.td_steps) * target_q_values

                    # メインモデルを学習モードに戻す
                    self.model.train()

                # 損失を計算
                loss = self.criterion(q_values, td_targets)
                losses.append(loss.item())

                # 勾配を計算
                loss.backward()

                # 勾配爆発を防ぐために勾配をクリッピング
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                # パラメータを更新
                self.optimizer.step()
                
                # 優先度を計算しバッファーを更新
                if epoch == self.num_epochs - 1:
                    priorities = torch.abs(q_values - td_targets).detach().numpy()
                    self.replay_buffer.update(data_indices, priorities)

                # 更新カウントを増やす（更新のインターバルを超えたらターゲットモデルを更新，10回ごとに更新回数を表示）
                self.update_count += 1
                if self.update_count >= self.update_interval:
                    self.target_model.load_state_dict(self.model.state_dict())
                    self.update_count = 0

            # 更新情報を表示
            self._showUpdateInfo(epoch, losses)
        return
    def _showUpdateInfo(self, epoch, losses):
        # 更新情報を表示
        losses = np.array(losses)
        print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Update count[{self.update_count}/ {self.update_interval}]")
        print(f"Average Loss: {np.mean(losses):.2f}, Min Loss: {np.min(losses):.2f}, Max Loss: {np.max(losses):.2f}, Std Loss: {np.std(losses):.2f}")

        # 10回ごとに更新情報を表示（それ以外はスキップ）
        if self.update_count % 1000 != 0:
            return 

        # 勾配消失・爆発の確認
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.norm().item():.3f}")
        return
            
    

    def saveModel(self):
        # モデルを保存
        torch.save(self.model.state_dict(), self.path_map['model'])
        torch.save(self.target_model.state_dict(), self.path_map['target_model'])    
        return
    
    def saveSession(self):
        # バッファーを保存
        self.replay_buffer.save()

        # その他のセッション情報を保存
        with self.path_map['session'].open('wb') as f:
            session_data = {
                'update_count': self.update_count,
                'total_reward_record': self.total_reward_record,
                'update_interval_record': self.update_interval_record,
                'num_data_for_learning_record': self.num_data_for_learning_record,
                'batch_record': self.batch_record,
                'num_epochs_record': self.num_epochs_record,
                'learning_rate_record': self.learning_rate_record,
                'weight_decay_record': self.weight_decay_record,
                'epsilon_record': self.epsilon_record,
            }
            pickle.dump(session_data, f)
        
        # 何回目のエピソードかを表示
        print(f"Master Agent {self.id}: Total Number of Episodes = {len(self.total_reward_record)}")
        return

    def updateSessionData(self):
        # トータルの報酬を更新
        sum_total_reward = 0
        for local_agent in self.local_agents.getAll():
            total_reward = local_agent.get('total_reward')
            sum_total_reward += total_reward
        
        avg_total_reward = sum_total_reward / self.local_agents.count()
        self.total_reward_record.append(avg_total_reward)

        # update_intervalの更新
        self.update_interval_record.append(self.update_interval)

        # num_data_for_learningの更新
        self.num_data_for_learning_record.append(self.replay_buffer.get('num_data_for_learning'))

        # batchの更新
        self.batch_record['number'].append(self.replay_buffer.get('num_batches'))
        self.batch_record['size'].append(self.replay_buffer.get('batch_size'))

        # num_epochsの更新
        self.num_epochs_record.append(self.num_epochs)

        # learning_rateの更新
        self.learning_rate_record.append(self.learning_rate)

        # weight_decayの更新
        self.weight_decay_record.append(self.weight_decay)

        # epsilonの更新
        self.epsilon_record.append(self.epsilon)

        return
    
    def showTotalReward(self):
        for local_agent_id in self.local_agents.getKeys(container_flg=True, sorted_flg=True):
            local_agent = self.local_agents[local_agent_id]
            print(f"Local Agent {local_agent_id}: Total Reward = {local_agent.get('total_reward'):.1f}")
        
        print(f"Master Agent {self.id}: Average Total Reward = {self.total_reward_record[-1]:.1f}")
        return
    
    @property
    def replay_buffer_path_map(self):
        return self.path_map['replay_buffer']




    

            
    




