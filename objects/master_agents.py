from libs.container import Container
from libs.object import Object
from objects.buffers.apex_buffer import ApexBuffer
from objects.intersections import Intersections
from objects.local_agents import LocalAgents
from objects.neural_networks.apex.proto_q_net import ProtoQNet

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import json

class MasterAgents(Container):
    def __init__(self, network, device):
        super().__init__()

        self.config = network.config
        self.executor = network.executor
        self.shared_resources = network.shared_resources
        self.device = device
        self.network = network

        self._initElements()
        return

    
    def _initElements(self):
        # make master agent objects
        for intersection in self.network.intersections.getAll():
            num_roads = intersection.get('num_roads')
            num_lanes_tuple = intersection.get('num_lanes_tuple')

            if intersection.has('master_agent'):
                continue

            self.add(MasterAgent(
                master_agents=self, 
                num_roads=num_roads,
                num_lanes_tuple=num_lanes_tuple,
            ))
        
        return

    def saveLearningData(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.saveLearningData)
        
        self.executor.wait()
        return
    
    def train(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.train)
        
        self.executor.wait()
        return
    
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
    def __init__(self, master_agents, num_roads,num_lanes_tuple):
        super().__init__()

        self.config = master_agents.config
        self.executor = master_agents.executor
        self.shared_resources = master_agents.shared_resources
        self.device = master_agents.device
        self.master_agents = master_agents
        self.network = master_agents.network

        self._initProps(num_roads, num_lanes_tuple)
        self._initIntersections()
        
        # set symmetry_phase_map, random_phase_prob_map, save_dir_path_map
        self._makeSymmetryPhaseMap()
        self._makeSaveDirPathMap()

        # initialize drl objects
        self._initDrlObjects()
        
        # model, session, bufferのデータをロード
        self._load()
        
        # LocalAgentオブジェクトを初期化
        self.local_agents = LocalAgents(self)
        return
    
    @property
    def num_learning_data(self):
        num_learning_data = 0
        for local_agent in self.local_agents.getAll():
            num_learning_data += local_agent.get('num_learning_data')
        return num_learning_data
    
    @property
    def learning_data_list(self):
        learning_data_list = []
        for local_agent in self.local_agents.getAll():
            learning_data_list.extend(local_agent.get('learning_data_list'))
        return learning_data_list
    
    def _initProps(self, num_roads, num_lanes_tuple):
        # set id and num_roads
        self.id = self.master_agents.count() + 1
        self.num_roads = num_roads

        # set phases_df and num_phases
        phases_df_map = self.config.get('phases_df_map')
        self.phases_df = phases_df_map[self.num_roads]  
        self.num_phases = self.phases_df.shape[0]

        # set random_phase_prob_map
        self.random_phase_prob_map = {}
        for _, row in self.phases_df.iterrows():
            self.random_phase_prob_map[int(row['id'])] = float(row['random_prob'])
        total_prob = sum(self.random_phase_prob_map.values())
        for phase_id in self.random_phase_prob_map:
            self.random_phase_prob_map[phase_id] /= total_prob

        # set num_lanes_map
        self.num_lanes_map = {}
        for road_id in range(1, self.num_roads + 1):
            self.num_lanes_map[road_id] = num_lanes_tuple[road_id - 1]

        # set num_simulations
        simulator_info = self.config.get('simulator_info')
        self.num_simulations = simulator_info['num_simulations']

        # set drl information
        drl_info = self.config.get('drl_info')
        self.learning_flg = drl_info['learning']['flg']
        self.architecture = drl_info['architecture']['type']
        self.learning_rate = float(drl_info['learning']['learning_rate'])
        self.weight_decay = float(drl_info['learning']['weight_decay'])

        # set update_count, episode, and session_df
        self.update_count = 0
        self.episode = 1
        self.session_df = None
        self.random_phase_probs_df = None
        return
    
    def _initIntersections(self):
        self.intersections = Intersections(self)
        for intersection in self.network.intersections.getAll():
            if intersection.get('num_lanes_tuple') == tuple((self.num_lanes_map[road_id] for road_id in range(1, self.num_roads + 1))):
                self.intersections.add(intersection)
                intersection.set('master_agent', self)
        return
    
    # 対称性のあるフェーズの組み合わせをマッピングするメソッド
    def _makeSymmetryPhaseMap(self):
        self.symmetry_phase_map = {}
        symmetry_phases_df_map = self.config.get('symmetry_phases_df_map')
        if self.num_roads == 4:
            tmp_tags = symmetry_phases_df_map[self.num_roads]
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
        
    def _makeSaveDirPathMap(self):
        self.save_dir_path_map = {}

        # get drl_dir_path
        root_dir_path = self.network.get('root_dir_path')
        drl_dir_path = root_dir_path / 'data' / 'drl'
        drl_dir_path.mkdir(parents=True, exist_ok=True)
        
        # get target config_dir_path
        found_flg = False
        for config_dir_path in drl_dir_path.glob('config_*'):
            config_file_path = config_dir_path / 'config.yaml'
            if not config_file_path.exists():
                continue

            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_yaml = yaml.safe_load(f)

            if config_yaml == self.config.get('drl_info'):
                found_flg = True
                break
        
        if not found_flg:
            config_id = 1
            while True:
                config_dir_path = drl_dir_path / f"config_{config_id}"
                if not config_dir_path.exists():
                    config_dir_path.mkdir(parents=True, exist_ok=False)
                    config_file_path = config_dir_path / 'config.yaml'
                    with config_file_path.open('w', encoding='utf-8') as f:
                        yaml.dump(self.config.get('drl_info'), f)
                    break
                config_id += 1
        
        # set save_dir_path_map
        self.save_dir_path_map = {
            'model': config_dir_path / 'model',
            'optimizer': config_dir_path / 'optimizer',
            'buffer': config_dir_path / 'buffer',
            'session': config_dir_path / 'session',
        }
        for path in self.save_dir_path_map.values():
            path.mkdir(parents=True, exist_ok=True)

        return
        
    def _initDrlObjects(self):
        # initialize model and target_model
        if self.architecture == 'proto':
            self.model = ProtoQNet(self)
        else:
            raise NotImplementedError(f"Not supported architecture: {self.architecture}")
        self.model.train()
        self.model.to(self.device)

        if self.architecture == 'proto':
            self.target_model = ProtoQNet(self)
        else:
            raise NotImplementedError(f"Not supported architecture: {self.architecture}")
        
        self.target_model.eval()
        self.target_model.to(self.device)

        # initialize buffer
        self.buffer = ApexBuffer(self)

        # initialize optimizer and criterion
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        return

    def _load(self):
        # load session_info, session_df
        if self.simulation_count > 1:
            self.update_count = self.shared_resources.get('update_count')
            self.session_df = self.shared_resources.get('session_df')
            self.random_phase_probs_df = self.shared_resources.get('random_phase_probs_df')
            self.episode = self.session_df.len() + 1

        else:
            session_info_file_path = self.save_dir_path_map['session'] / 'session_info.json'
            if session_info_file_path.exists():
                with session_info_file_path.open('r', encoding='utf-8') as f:
                    session_info = json.load(f)
                
                self.update_count = session_info['update_count']

            session_df_file_path = self.save_dir_path_map['session'] / 'session_df.csv'
            if session_df_file_path.exists():
                with open(session_df_file_path, 'r', encoding='utf-8') as f:
                    self.session_df = pd.read_csv(f)
                    self.episode = len(self.session_df) + 1
            
            random_phase_probs_df_file_path = self.save_dir_path_map['session'] / 'random_phase_probs_df.csv'
            if random_phase_probs_df_file_path.exists():
                with open(random_phase_probs_df_file_path, 'r', encoding='utf-8') as f:
                    self.random_phase_probs_df = pd.read_csv(f)
                    

        # load model
        if self.simulation_count > 1:
            self.model = self.shared_resources.get('model')
        
        else:
            model_file_path = self.save_dir_path_map['model'] / 'q_net.pth'
            if model_file_path.exists():
                self.model.load_state_dict(torch.load(model_file_path))
        
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        elif self.simulation_count > 1:
            self.target_model = self.shared_resources.get('target_model')
        else:
            target_model_file_path = self.save_dir_path_map['model'] / 'target_q_net.pth'
            if target_model_file_path.exists():
                self.target_model.load_state_dict(torch.load(target_model_file_path))
            else:
                self.target_model.load_state_dict(self.model.state_dict())
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
        if self.num_learning_data == 0:
            self.buffer.set('change_flg', False)
            return
        
        self.buffer.update()
        return
    
    def train(self):
        if not self.learning_flg:
            return
        
        if not self.buffer.get('enough_new_data_flg'):
            return

        batch_data = self.buffer.sample()
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
                    self.buffer.update(data_indices, priorities)

                # 更新カウントを増やす（更新のインターバルを超えたらターゲットモデルを更新，10回ごとに更新回数を表示）
                self.update_count += 1
                if self.update_count >= self.update_interval:
                    self.target_model.load_state_dict(self.model.state_dict())
                    self.update_count = 0

            # 更新情報を表示
            self._showUpdateInfo(epoch, losses)
        return
    
    def clearLearningData(self):
        for local_agent in self.local_agents.getAll():
            learning_data_list = local_agent.get('learning_data_list')
            learning_data_list.clear()
        
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
            self.buffer.set('initial_priority', max_loss)
            

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
            return
        
        if self.finish_flg or self.simulation_count % self.save_interval == 0:
            # save session_info, session_df and random_phase_probs_df
            with open(self.save_dir_path_map['session'] / 'session_info.json', 'w', encoding='utf-8') as f:
                json.dump({'update_count': self.update_count,}, f)

            with open(self.save_dir_path_map['session'] / 'session_df.csv', 'w', encoding='utf-8') as f:
                self.session_df.to_csv(f, index=False)
            
            with open(self.save_dir_path_map['session'] / 'random_phase_probs_df.csv', 'w', encoding='utf-8') as f:
                self.random_phase_probs_df.to_csv(f, index=False)
            
            # save model and optimizer
            torch.save(self.model.state_dict(), self.save_dir_path_map['model'] / 'q_net.pth')
            torch.save(self.target_model.state_dict(), self.save_dir_path_map['model'] / 'target_q_net.pth')
            torch.save(self.optimizer.state_dict(), self.save_dir_path_map['optimizer'] / 'optimizer.pth')  
        
            # save replay buffer
            self.buffer.save()

        # set properties to shared resources
        self.shared_resources.set('model', self.model)
        self.shared_resources.set('target_model', self.target_model)
        self.shared_resources.set('update_count', self.update_count)
        self.shared_resources.set('session_df', self.session_df)
        self.shared_resources.set('random_phase_probs_df', self.random_phase_probs_df)  
        self.shared_resources.set('buffer', self.buffer)
        return

    def updateSessionData(self):
        self._updateAverageTotalReward()

        if not self.learning_flg:
            return
        
        # update session_df
        new_session_data = {
            'episode': self.episode,
            'total_reward': self.avg_total_reward,
            'update_interval': self.update_interval,
            'num_new_data': self.buffer.get('num_new_data'),
            'num_batches': self.buffer.get('num_batches'),
            'batch_size': self.buffer.get('batch_size'),
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epsilon': self.epsilon,
            'simulation_time': self.simulation_time,
        }
        new_session_df = pd.DataFrame(new_session_data, index=[0])

        if self.session_df is None:
            self.session_df = new_session_df
        else:
            self.session_df = pd.concat([self.session_df, new_session_df], ignore_index=True)

        # update random_phase_probs_df
        new_random_phase_probs_df = pd.DataFrame(self.random_phase_prob_map, index=[0])
        if self.random_phase_probs_df is None:
            self.random_phase_probs_df = new_random_phase_probs_df
        else:
            self.random_phase_probs_df = pd.concat([self.random_phase_probs_df, new_random_phase_probs_df], ignore_index=True)

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
        




    

            
    




