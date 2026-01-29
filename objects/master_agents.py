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
import pandas as pd

class MasterAgents(Container):
    def __init__(self, network, device):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = network.config
        self.executor = network.executor

        # 引継ぎデータ格納用のオブジェクトを取得
        self.shared_resources = network.shared_resources

        # デバイスを設定
        self.device = device

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
    
    def save(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.save)
        self.executor.wait()
        return

class MasterAgent(Object):
    def __init__(self, master_agents, num_lanes_turple):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = master_agents.config
        self.executor = master_agents.executor

        # 引継ぎデータ格納用のオブジェクトを取得
        self.shared_resources = master_agents.shared_resources

        # デバイスを設定
        self.device = master_agents.device

        # 上位オブジェクトを取得
        self.master_agents = master_agents
        self.network = master_agents.network

        # IDを設定
        self.id = self.master_agents.count() + 1

        # num_roadsを設定
        self.num_roads = len(num_lanes_turple)

        # intersectionsオブジェクトと紐づける
        self._makeIntersectionConnections(num_lanes_turple)

        # 車線数の情報を取得
        self._makeNumLanesMap(num_lanes_turple)
        
        # symmetry_phase_map, random_phase_probsを作成
        self._makeSymmetryPhaseMap()
        self._makeRandomPhaseProbs()

        # パラメータを取得
        self._getParams()

        # 保存先のパスを定義
        self._makePaths()

        # モデルを初期化
        self._initSession()
        self._initModel()
        self.replay_buffer = ReplayBuffer(self)
        
        # model, session, replay_bufferのデータをロード
        self._load()

        # epsilonの初期化
        self._makeEpsilon()
        
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
        return
    
    # 対称性のあるフェーズの組み合わせをマッピングするメソッド
    def _makeSymmetryPhaseMap(self):
        self.symmetry_phase_map = {}
        symmetry_phase_tags = self.config.get('symmetry_phase_tags')
        if self.num_roads == 4:
            tmp_tags = symmetry_phase_tags[self.num_roads]
            for _, tmp_tag in tmp_tags.iterrows():
                phase_id = tmp_tag['phase_id']
                symmetry_phase_id = tmp_tag['symmetry_phase_id']
                symmetry_type = tmp_tag['type']

                if phase_id not in self.symmetry_phase_map:
                    self.symmetry_phase_map[phase_id] = {}

                self.symmetry_phase_map[phase_id][symmetry_type] = symmetry_phase_id
        else:
            # 後々定義
            raise ValueError(f"Symmetry phase map is not defined for {self.num_roads} roads.")
        
        return

    # ランダム行動時の行動の確率分布を作成するメソッド
    def _makeRandomPhaseProbs(self):
        num_roads_phases_map = self.config.get('num_roads_phases_map')
        phases = num_roads_phases_map[self.num_roads]
        self.random_phase_probs = {}
    
        for _, row in phases.iterrows():
            self.random_phase_probs[int(row['id'])] = float(row['random_prob'])
        
        # 確率の合計が1になるように正規化
        total_prob = sum(self.random_phase_probs.values())
        for phase_id in self.random_phase_probs:
            self.random_phase_probs[phase_id] /= total_prob
        return

    def _getParams(self):
        simulator_info = self.config.get('simulator_info')
        self.num_simulations = simulator_info['num_simulations']

        drl_info = self.config.get('drl_info')
        self.drl_method = drl_info['method']
        self.duration_steps = drl_info['duration_steps']
        self.network_id = drl_info['network_id']
        self.reward_id = drl_info['reward_id']
        self.done_reward = drl_info['done_reward']
        self.data_augmentation_flg = drl_info['data_augmentation_flg']
        self.num_vehicles = drl_info['num_vehicles']
        self.bc_flg = drl_info['bc_flg']
        self.learning_flg = drl_info['learning_flg']
        self.state_id = drl_info['state_id']
        self.save_interval = drl_info['save_interval']
        self.stop_flg = drl_info['stop']['flg']
        self.stop_type = drl_info['stop']['type']

        if self.stop_type == 'episode':
            self.stop_episode = drl_info['stop']['episode']
        elif self.stop_type == 'interval':
            self.stop_interval = drl_info['stop']['interval']
        else:
            raise ValueError(f"Invalid stop type: {self.stop_type}")
        
        apex_info = self.config.get('apex_info')
        self.td_steps = apex_info['td_steps']
        self.update_interval = apex_info['update_interval']
        self.buffer_size = apex_info['buffer']['size']
        self.weight_decay = apex_info['weight_decay']
        self.gamma = apex_info['gamma']
        self.learning_rate = apex_info['learning_rate']
        self.num_epochs = apex_info['num_epochs']
        return

    def _makePaths(self):
        self.path_map = {}

        num_lanes_str = ''
        for num_lanes in self.num_lanes_map.values():
            num_lanes_str += str(num_lanes)

        # modelについて
        model_dir = Path("models") / self.drl_method
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=False)

        csv_path = model_dir / "model.csv"
        if not csv_path.exists():
            model_df = pd.DataFrame(columns=['id', 'network_id', 'reward_id', 'num_lanes', 'num_vehicles', 'duration_steps', 'buffer_size', 'state_id', 'data_augmentation_flg', 'gamma', 'done_reward'])
        else:
            model_df = pd.read_csv(csv_path, dtype={'num_lanes': int}, index_col=False)

        filtered_model_df = model_df[
            (model_df['network_id'] == self.network_id) & 
            (model_df['reward_id'] == self.reward_id) & 
            (model_df['num_lanes'] == int(num_lanes_str)) & 
            (model_df['num_vehicles'] == self.num_vehicles) & 
            (model_df['duration_steps'] == self.duration_steps) & 
            (model_df['buffer_size'] == self.buffer_size) & 
            (model_df['state_id'] == self.state_id) &
            (model_df['data_augmentation_flg'] == int(self.data_augmentation_flg)) &
            (model_df['gamma'] == self.gamma) &
            (model_df['done_reward'] == self.done_reward)
        ]
        exist_flg = not filtered_model_df.empty

        if exist_flg:
            self.model_id = filtered_model_df['id'].values[0]
        else:
            new_row = {
                'id' : model_df['id'].max() + 1 if not model_df['id'].empty else 1,
                'network_id' : self.network_id,
                'reward_id' : self.reward_id,
                'num_lanes' : int(num_lanes_str),
                'num_vehicles' : self.num_vehicles,
                'duration_steps' : self.duration_steps,
                'buffer_size' : self.buffer_size,
                'state_id' : self.state_id,
                'data_augmentation_flg' : int(self.data_augmentation_flg),
                'gamma': self.gamma,
                'done_reward': self.done_reward,
            }
            model_df = pd.concat([model_df, pd.DataFrame([new_row])], ignore_index=True)
            self.model_id = new_row['id']
        
        model_df.to_csv(csv_path, index=False)

        tmp_model_dir = model_dir / f"model_{self.model_id}"
        if not tmp_model_dir.exists():
            tmp_model_dir.mkdir(parents=True, exist_ok=False)

        self.path_map['model'] = tmp_model_dir / 'q_net.pth'
        self.path_map['target_model'] = tmp_model_dir / 'target_q_net.pth'

        # sessionについて
        session_dir = Path("results") / 'session' / 'drl' / self.drl_method
        if not session_dir.exists():
            session_dir.mkdir(parents=True, exist_ok=False)
        
        tmp_session_dir = session_dir / f"session_{self.model_id}"
        if not tmp_session_dir.exists():
            tmp_session_dir.mkdir(parents=True, exist_ok=False)
        self.path_map['session'] = tmp_session_dir / 'session.pkl'

        # replay_bufferについて
        buffer_dir = Path("buffers") / self.drl_method
        if not buffer_dir.exists():
            buffer_dir.mkdir(parents=True, exist_ok=False)
        
        tmp_buffer_dir = buffer_dir / f"buffer_{self.model_id}"
        if not tmp_buffer_dir.exists():
            tmp_buffer_dir.mkdir(parents=True, exist_ok=False)

        self.path_map['replay_buffer'] = {
            'tree': tmp_buffer_dir / f"replay_buffer_tree.pkl",
            'data': [tmp_buffer_dir / f"replay_buffer_data_{idx + 1}.pkl" for idx in range(math.ceil(self.buffer_size / 1000))]
        }

        if self.bc_flg:
            # 直す必要あり
            self.path_map['bc_model'] = Path(f"models/bc_q_net_{self.network_id}_{num_lanes_str}_{self.num_vehicles}.pth")
        return
    
    def _initSession(self):
        # sessionを初期化
        self.update_count = 0
        self.total_reward_record = []
        self.update_interval_record = []
        self.num_new_data_record = []
        self.batch_record = {
            'number': [],
            'size': [],
        }
        self.num_epochs_record = []
        self.learning_rate_record = []
        self.weight_decay_record = []
        self.epsilon_record = []
        self.simulation_time_record = []
        self.random_phase_probs_record = {phase_id: [] for phase_id in self.random_phase_probs.keys()}

        self.episode = 1
        return
        
    def _initModel(self):
        # modelを初期化
        if self.network_id == 1:
            self.model = QNet1(self.config, self.device, self.num_vehicles, self.num_lanes_map)
        self.model.train()
        self.model.to(self.device)

        if self.network_id == 1:
            self.target_model = QNet1(self.config, self.device, self.num_vehicles, self.num_lanes_map)
        self.target_model.eval()
        self.target_model.to(self.device)
        return

    def _load(self):
        # sessionを読み込む
        session_data = None
        if self.simulation_count == 1 and self.path_map['session'].exists():
            with self.path_map['session'].open('rb') as f:
                session_data = pickle.load(f)
        elif self.shared_resources.has('session_data'):
            session_data = self.shared_resources.get('session_data')

        if session_data is not None:
            self.update_count = session_data['update_count']
            self.total_reward_record = session_data['total_reward_record']
            self.update_interval_record = session_data['update_interval_record']
            self.num_new_data_record = session_data['num_new_data_record']
            self.batch_record = session_data['batch_record']
            self.num_epochs_record = session_data['num_epochs_record']
            self.learning_rate_record = session_data['learning_rate_record']
            self.weight_decay_record = session_data['weight_decay_record']
            self.epsilon_record = session_data['epsilon_record']
            self.simulation_time_record = session_data['simulation_time_record']
            self.random_phase_probs_record = session_data['random_phase_probs_record']

        self.episode = len(self.total_reward_record) + 1

        # modelを読み込む
        if self.simulation_count > 1:
            self.model = self.shared_resources.get('model')
        elif self.path_map['model'].exists():
            self.model.load_state_dict(torch.load(self.path_map['model']))
        elif self.bc_flg and self.path_map['bc_model'].exists():
            self.model.load_state_dict(torch.load(self.path_map['bc_model']))
        
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        elif self.simulation_count > 1:
            self.target_model = self.shared_resources.get('target_model')
        elif self.path_map['target_model'].exists():
            self.target_model.load_state_dict(torch.load(self.path_map['target_model']))
        else:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # replay_bufferを読み込む
        self.replay_buffer.load()
        return
    
    def _makeEpsilon(self):
        apex_info = self.config.get('apex_info')
        self.epsilon_schedule_flg = apex_info['epsilon']['schedule_flg']

        if not self.epsilon_schedule_flg:
            self.epsilon = apex_info['epsilon']['value']
            return
        
        # epsilonのスケジュールを取得
        epsilon_schedule = self.config.get('epsilon_schedule')
        schedule_interval = len(epsilon_schedule) 

        self.epsilon = epsilon_schedule['epsilon'].iloc[(self.episode - 1) % schedule_interval]
        return
    
    def saveLearningData(self):
        # バッファーのサイズが変化したかどうかをフラグで管理
        self.replay_buffer.set('change_flg', False)
        
        # ローカルエージェントを走査
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

            # バッファーのサイズが変化したことをフラグで管理
            self.replay_buffer.set('change_flg', True)
        return
    
    def train(self):
        # バッファーのサイズが変化していない場合は学習しない
        if not self.learning_flg:
            return
        
        if not self.replay_buffer.get('should_learn_flg'):
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
                    actions = torch.tensor([tmp_data[1] - 1 for tmp_data in data], dtype=torch.int64).unsqueeze(1).to(self.device)

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
                        cumurative_rewards = torch.tensor([tmp_data[2] for tmp_data in data], dtype=torch.float32).unsqueeze(1).to(self.device)

                        # 終了フラグをテンソルに変換
                        done_flgs = torch.tensor([tmp_data[4] for tmp_data in data], dtype=torch.float32).unsqueeze(1).to(self.device)

                        # TDターゲットを計算
                        td_targets = cumurative_rewards + (1 - done_flgs) * (self.gamma ** self.td_steps) * target_q_values

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
                    priorities = torch.abs(q_values - td_targets).detach().cpu().numpy()
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
        mean_loss = np.mean(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        std_loss = np.std(losses)
        print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Update count[{self.update_count}/ {self.update_interval}]")
        print(f"Average Loss: {mean_loss:.3f}, Min Loss: {min_loss:.3f}, Max Loss: {max_loss:.3f}, Std Loss: {std_loss:.3f}")

        if epoch == 0:
            self.replay_buffer.set('initial_priority', max_loss)
            

        # 10回ごとに更新情報を表示（それ以外はスキップ）
        if self.update_count % 1000 != 0:
            return 

        # 勾配消失・爆発の確認
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.norm().item():.3f}")
        return
            
    def save(self):
        if not self.learning_flg:
            print(f"Master Agent {self.id}: This simulation is for evaluation only.")
            return
        
        if self.finish_flg or self.simulation_count % self.save_interval == 0:   
            # session情報を保存
            with self.path_map['session'].open('wb') as f:
                session_data = {
                    'update_count': self.update_count,
                    'total_reward_record': self.total_reward_record,
                    'update_interval_record': self.update_interval_record,
                    'num_new_data_record': self.num_new_data_record,
                    'batch_record': self.batch_record,
                    'num_epochs_record': self.num_epochs_record,
                    'learning_rate_record': self.learning_rate_record,
                    'weight_decay_record': self.weight_decay_record,
                    'epsilon_record': self.epsilon_record,
                    'simulation_time_record': self.simulation_time_record,
                    'random_phase_probs_record': self.random_phase_probs_record,
                }
                pickle.dump(session_data, f)
            
            # modelを保存
            torch.save(self.model.state_dict(), self.path_map['model'])
            torch.save(self.target_model.state_dict(), self.path_map['target_model']) 
        
        # bufferを保存
        self.replay_buffer.save()
        
        # shared_resourcesオブジェクトに保存
        self.shared_resources.set('model', self.model)
        self.shared_resources.set('target_model', self.target_model)
        self.shared_resources.set('session_data', {
            'update_count': self.update_count,
            'total_reward_record': self.total_reward_record,
            'update_interval_record': self.update_interval_record,
            'num_new_data_record': self.num_new_data_record,
            'batch_record': self.batch_record,
            'num_epochs_record': self.num_epochs_record,
            'learning_rate_record': self.learning_rate_record,
            'weight_decay_record': self.weight_decay_record,
            'epsilon_record': self.epsilon_record,
            'simulation_time_record': self.simulation_time_record,
            'random_phase_probs_record': self.random_phase_probs_record,
        })

        print(f"Master Agent {self.id}: Total Number of Episodes = {len(self.total_reward_record)}")    
        return

    def updateSessionData(self):
        # トータルの報酬を計算
        self._updateAverageTotalReward()

        # 学習フラグが立っていないときはスキップ
        if not self.learning_flg:
            return
        
        # データを記録
        self.total_reward_record.append(self.avg_total_reward) 
        self.update_interval_record.append(self.update_interval)
        self.num_new_data_record.append(self.replay_buffer.get('num_new_data')) 
        self.batch_record['number'].append(self.replay_buffer.get('num_batches')) 
        self.batch_record['size'].append(self.replay_buffer.get('batch_size'))
        self.num_epochs_record.append(self.num_epochs)
        self.learning_rate_record.append(self.learning_rate) 
        self.weight_decay_record.append(self.weight_decay) 
        self.epsilon_record.append(self.epsilon) 
        self.simulation_time_record.append(self.simulation_time)
        for phase_id, prob in self.random_phase_probs.items():
            self.random_phase_probs_record[phase_id].append(prob)
        return
    
    def _updateAverageTotalReward(self):
        sum_total_reward = 0
        for local_agent in self.local_agents.getAll():
            total_reward = local_agent.get('total_reward')
            sum_total_reward += total_reward
        self.avg_total_reward = sum_total_reward / self.local_agents.count()
        return
    
    def showTotalReward(self):
        for local_agent_id in self.local_agents.getKeys(container_flg=True, sorted_flg=True):
            local_agent = self.local_agents[local_agent_id]
            print(f"Local Agent {local_agent_id}: Total Reward = {local_agent.get('total_reward'):.1f}")
        
        print(f"Master Agent {self.id}: Average Total Reward = {self.avg_total_reward:.1f}")
        return
    
    @property
    def replay_buffer_path_map(self):
        return self.path_map['replay_buffer']
    
    @property
    def simulation_count(self):
        vissim = self.network.vissim
        return vissim.get('simulation_count')
    
    @property
    def finish_flg(self):
        if self.simulation_count == self.num_simulations:
            return True
        
        if not self.stop_flg:
            return False
        
        if self.stop_type == 'episode' and self.episode == self.stop_episode:
            return True

        if self.stop_type == 'interval' and (self.episode % self.stop_interval == 0):
            return True
    
        return False
        
    @property
    def simulation_time(self):
        return self.network.simulation.get('current_time')
        




    

            
    




