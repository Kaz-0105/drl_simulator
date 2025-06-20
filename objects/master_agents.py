from libs.container import Container
from libs.object import Object
from libs.replay_buffer import ReplayBuffer
from objects.intersections import Intersections
from objects.local_agents import LocalAgents
from neural_networks.q_net import QNet
from neural_networks.q_net2 import QNet2
from neural_networks.q_net3 import QNet3

from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

class MasterAgents(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理の実行オブジェクトを取得
        self.config = network.config
        self.executor = network.executor

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
    
    def updateLocalAgents(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.updateLocalAgents)
        
        self.executor.wait()
    
    def updateRewardsRecord(self):
        for master_agent in self.getAll():
            master_agent.updateRewardsRecord()
    
    def saveSession(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.saveSession)
        
        self.executor.wait()

class MasterAgent(Object):
    def __init__(self, master_agents, num_lanes_turple):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = master_agents.config
        self.executor = master_agents.executor

        # 上位オブジェクトを取得
        self.master_agents = master_agents

        # IDを設定
        self.id = self.master_agents.count() + 1

        # intersectionsオブジェクトと紐づける
        self._makeIntersectionConnections(num_lanes_turple)

        # 車線数の情報と自動車台数の情報を取得
        self._makeNumLanesMap(num_lanes_turple)
        drl_info = self.config.get('drl_info')
        self.drl_method = drl_info['method']
        self.num_vehicles = drl_info['num_vehicles']

        # 強化学習関連のハイパーパラメータを取得
        apex_info = self.config.get('apex_info')
        self.network_id = apex_info['network_id']
        self.update_interval = apex_info['update_interval']
        self.gamma = apex_info['gamma']
        self.learning_rate = apex_info['learning_rate']
        self.td_steps = apex_info['td_steps']

        # 保存先のパスを定義
        self._makeSavePaths()
        
        # 前回のシミュレーション終了時点の更新回数を読み込む
        self._restoreSession()

        # 使用する強化学習の手法で分岐
        self._makeModel()

        # 最適化手法と評価関数を定義
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 経験再生用のバッファを初期化
        self.replay_buffer = ReplayBuffer(self)
        
        # LocalAgentオブジェクトを初期化
        self.local_agents = LocalAgents(self)

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

    def _makeSavePaths(self):
        # 車線情報を文字列に変換
        num_lanes_str = ''
        for num_lanes in self.num_lanes_map.values():
            num_lanes_str += str(num_lanes)
        
        # 自動車台数の情報を文字列に変換
        num_vehs_str = str(self.num_vehicles)

        # モデルの保存先を定義
        self.model_path = Path('models/q_net_' + str(self.network_id) + '_' + num_lanes_str + '_' + num_vehs_str + '.pth')
        self.target_model_path = Path('models/target_q_net_' + str(self.network_id) + '_' + num_lanes_str + '_' + num_vehs_str + '.pth')
        self.update_count_path = Path('results/update_count_' + str(self.network_id) + '_' + num_lanes_str + '_' + num_vehs_str + '.npy')
        self.rewards_record_path = Path('results/rewards_record_' + str(self.network_id) + '_' + num_lanes_str + '_' + num_vehs_str + '.npy')
        
    def _makeModel(self):
        if self.config.get('drl_info')['method'] =='apex':
            # モデルを初期化（学習用にセット）
            if self.network_id == 1:
                self.model = QNet(self.config, self.num_vehicles, self.num_lanes_map)
            elif self.network_id == 2:
                self.model = QNet2(self.config, self.num_vehicles, self.num_lanes_map)
            elif self.network_id == 3:
                self.model = QNet3(self.config, self.num_lanes_map)
            self.model.train()

            # ターゲットモデルを初期化（学習用と同期，推論用にセット）
            if self.network_id == 1:
                self.target_model = QNet(self.config, self.num_vehicles, self.num_lanes_map)
            elif self.network_id == 2:
                self.target_model = QNet2(self.config, self.num_vehicles, self.num_lanes_map)
            elif self.network_id == 3:
                self.target_model = QNet3(self.config, self.num_lanes_map)

            # 過去に学習済みの場合はそれを読み込む
            self._loadModel()
    
            self.target_model.eval()

    def _loadModel(self):
        # メインのモデルを読み込む
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path))
        
        # ターゲットモデルを読み込む
        if not self.target_model_path.exists() or self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.target_model.load_state_dict(torch.load(self.target_model_path))
    
    def _restoreSession(self):
        # update_countを初期化
        self.update_count = 0

        # 存在する場合は読み込む
        if self.update_count_path.exists():
            self.update_count = np.load(self.update_count_path, allow_pickle=True).item()
        
        # エピソードごとの累積報酬のデータを初期化
        self.rewards_record = []

        # 存在する場合は読み込む
        if self.rewards_record_path.exists():
            self.rewards_record = np.load(self.rewards_record_path, allow_pickle=True).tolist()

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
    
    def train(self):
        if not self.buffer_change_flg:
            # バッファーのサイズが変化していない場合は学習しない
            return
        
        # バッファーのサイズが十分でない場合は学習しない
        if self.replay_buffer.get('current_size') < self.replay_buffer.get('batch_size'):
            return

        # バッファーからデータを取得
        batch_data, batch_data_indices = self.replay_buffer.sample()

        # 勾配を初期化
        self.optimizer.zero_grad()
        
        if self.drl_method == 'apex' and self.network_id == 1:
            # とった行動をテンソルに変換
            actions = torch.tensor([tmp_data[1] - 1 for tmp_data in batch_data], dtype=torch.int64).unsqueeze(1)

            # 状態を配列にする
            states = [tmp_data[0] for tmp_data in batch_data]

            # 勾配をトラッキングするように設定
            self.model.set('requires_grad', True)

            # Q値を計算し，選ばれた行動のQ値を取得
            q_values = self.model(states).gather(1, actions)

            # メインモデルを評価モードに設定
            self.model.eval()

            # TDターゲットを計算するアルゴリズムここから
            with torch.no_grad():
                # 次の状態を配列にする
                states_next = [tmp_data[3] for tmp_data in batch_data]

                # 勾配をトラッキングしないように設定
                self.model.set('requires_grad', False)

                # 次の状態のメインモデルのQ値の最大値を与える行動を取得
                max_actions = torch.argmax(self.model(states_next), dim=1).unsqueeze(1)

                # ターゲットモデルのQ値を取得
                target_q_values = self.target_model(states_next).gather(1, max_actions)

                # 累積報酬をテンソルに変換（multi step bootstrap を実装している）
                cumurative_rewards = torch.tensor([tmp_data[2] for tmp_data in batch_data], dtype=torch.float32).unsqueeze(1)

                # 終了フラグをテンソルに変換
                dones = torch.tensor([tmp_data[4] for tmp_data in batch_data], dtype=torch.float32).unsqueeze(1)

                # TDターゲットを計算
                td_targets = cumurative_rewards + (1 - dones) * (self.gamma ** self.td_steps) * target_q_values

            # メインモデルを学習モードに戻す
            self.model.train()

        elif self.drl_method == 'apex' and self.network_id == 3:
            # Q値を計算
            actions = torch.tensor([tmp_data[1] - 1 for tmp_data in batch_data], dtype=torch.int64).unsqueeze(1)
            states = torch.stack([tmp_data[0] for tmp_data in batch_data]).squeeze(1)
            states.requires_grad_(True)
            q_values_all = self.model(states)
            q_values = q_values_all.gather(1, actions) 
            
            # TDターゲットを計算（Double DQNの実装）
            self.model.eval()
            with torch.no_grad():
                states_next = torch.stack([tmp_data[3] for tmp_data in batch_data]).squeeze(1)
                states_next.requires_grad_(False)
                max_actions = torch.argmax(self.model(states_next), dim=1)
                target_q_values_all = self.target_model(states_next)
                target_q_values = target_q_values_all.gather(1, max_actions.unsqueeze(1))
                dones = torch.tensor([tmp_data[4] for tmp_data in batch_data], dtype=torch.float32).unsqueeze(1)
                td_targets = (1 - dones) * (self.gamma ** self.td_steps) * target_q_values
                td_targets += torch.tensor([tmp_data[2] for tmp_data in batch_data], dtype=torch.float32).unsqueeze(1)           
            self.model.train()

        # 損失を計算
        loss = self.criterion(q_values, td_targets)

        # 勾配を計算
        loss.backward()

        # 勾配爆発を防ぐために勾配をクリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        # パラメータを更新
        self.optimizer.step()

        # 更新回数をインクリメント
        self.update_count = (self.update_count + 1) % self.update_interval

        # ターゲットモデルの更新
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print('The target model syncronized to the main model.')
        
        # 優先度を計算しバッファーを更新
        priorities = torch.abs(q_values - td_targets).detach().numpy()
        self.replay_buffer.update(batch_data_indices, priorities)

        # 更新情報を表示
        self._showUpdateInfo(q_values, td_targets, loss)
    
    def _showUpdateInfo(self, q_values, td_targets, loss):
        # 現在の更新回数を表示
        print(f'Update count: {self.update_count}')

        # Q値が発散していないか確認
        print(f"Q-values: min = {q_values.min().item():.3f}, max = {q_values.max().item():.3f}")
        print(f"TD-targets: min = {td_targets.min().item():.3f}, max = {td_targets.max().item():.3f}")
        print(f"Loss: {loss.item():.3f}")

        # 10回ごとに更新情報を表示（それ以外はスキップ）
        if self.update_count % 10 != 0:
            return 

        # 勾配消失・爆発の確認
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.norm().item():.5f}")
            
    def updateLocalAgents(self):
        drl_info = self.config.get('drl_info')
        if drl_info['method'] == 'apex':
            # 同期のタイミングではないときはスキップ
            if self.update_count != 0:
                return
            
            # ローカルエージェントを走査
            for local_agent in self.local_agents.getAll():
                local_agent.model.load_state_dict(self.model.state_dict())
    
    def saveSession(self):
        # モデルを保存
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(self.target_model.state_dict(), self.target_model_path)

        # バッファーを保存
        self.replay_buffer.save()

        # update_countを保存
        np.save(self.update_count_path, np.array(self.update_count, dtype=np.int64))

        # エピソードごとの累積報酬を保存
        np.save(self.rewards_record_path, np.array(self.rewards_record, dtype=object))

    def updateRewardsRecord(self):
        rewards_list = []
        for local_agent in self.local_agents.getAll():
            total_rewards = local_agent.get('total_rewards')
            rewards_list.append(local_agent.get('total_rewards'))
            print(f"Local agent {self.id}-{local_agent.get('id')} total rewards: {total_rewards}")
        
        average_rewards = round(sum(rewards_list) / len(rewards_list), 2)
        self.rewards_record.append(average_rewards)
        print(f"Master agent {self.id} average rewards: {average_rewards:.2f}")





            
    




