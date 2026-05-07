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
import random
from torch.utils.data.dataloader import default_collate
import copy

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
    
    @property
    def finish_flg(self):
        return any(agent.get('finish_flg') for agent in self.getAll())
    
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
    
    def update(self, type):
        for agent in self.getAll():
            self.executor.submit(agent.update, type)
        self.executor.wait()
        return
    
    def train(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.train)
        self.executor.wait()
        return
    
    def save(self):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.save)
        self.executor.wait()
        return

    def showInfo(self, type):
        for master_agent in self.getAll():
            self.executor.submit(master_agent.showInfo, type)
        self.executor.wait()
        return

class MasterAgent(Object):
    def __init__(self, master_agents, num_roads, num_lanes_tuple):
        super().__init__()

        self.config = master_agents.config
        self.executor = master_agents.executor
        self.shared_resources = master_agents.shared_resources
        self.device = master_agents.device
        self.master_agents = master_agents
        self.network = master_agents.network

        self._initProps(num_roads, num_lanes_tuple)
        self._initIntersections()
        
        # set symmetry_phase_map, save_dir_path_map
        self._makeSymmetryPhaseMap()
        self._makeSaveDirPathMap()

        # initialize drl objects
        self._initDrlObjects()
        
        # load model, session, and buffer
        self._load()
        
        # set local_agents
        self.local_agents = LocalAgents(self)
        return
    
    @property
    def learning_data_list(self):
        learning_data_list = []
        for local_agent in self.local_agents.getAll():
            learning_data_list.extend(local_agent.get('learning_data_list'))
        return learning_data_list
    
    @property
    def finish_flg(self):
        if self.simulation_type == 'test':
            return True
        
        elif self.simulation_type == 'train':
            if self.stop_type == 'interval':
                return self.episode % self.stop_interval == 0
            elif self.stop_type == 'episode':
                return self.episode == self.stop_episode
            else:
                raise NotImplementedError(f"Not supported stop type: {self.stop_type}")
        
        else:
            raise NotImplementedError(f"Not supported DRL type: {self.simulation_type}")
        
    @property
    def avg_total_reward(self):
        sum_total_reward = 0
        for local_agent in self.local_agents.getAll():
            total_reward = local_agent.get('total_reward')
            sum_total_reward += total_reward
        return sum_total_reward / self.local_agents.count()
    
    @property
    def simulation_id(self):
        return self.network.simulation.get('id')
        
    @property
    def simulation_time(self):
        return self.network.simulation.get('current_time')

    @property
    def seed(self):
        return self.network.simulation.get('seed')
    
    def _initProps(self, num_roads, num_lanes_tuple):
        # set id and num_roads
        self.id = self.master_agents.count() + 1
        self.num_roads = num_roads
        
        # get phases_df_map
        phases_df_map = self.config.get('phases_df_map')

        # set active_phase_list and num_max_phases
        self.active_phase_list = []
        for _, phase_row in phases_df_map[self.num_roads].iterrows():
            if phase_row['active_flg'] == 1:
                self.active_phase_list.append(int(phase_row['id']))
        self.num_max_phases = len(phases_df_map[self.num_roads])

        # set num_lanes_map
        self.num_lanes_map = {}
        for road_id in range(1, self.num_roads + 1):
            self.num_lanes_map[road_id] = num_lanes_tuple[road_id - 1]
        
        # set drl information
        drl_info = self.config.get('drl_info')
        self.simulation_type = drl_info['simulation_type']
        
        # set training information
        self.num_batches = drl_info['training']['batch']['number']
        self.batch_size = drl_info['training']['batch']['size']
        self.num_epochs = drl_info['training']['epoch']
        self.learning_rate = float(drl_info['training']['learning_rate'])
        self.weight_decay = float(drl_info['training']['weight_decay'])
        self.norm_clip = float(drl_info['training']['norm_clip'])

        # set stop information
        self.stop_type = drl_info['stop']['type']
        if self.stop_type == 'episode':
            self.stop_episode = drl_info['stop']['episode']
        elif self.stop_type == 'interval':
            self.stop_interval = drl_info['stop']['interval']
        else:
            raise NotImplementedError(f"Not supported stop type: {self.stop_type}")

        # set framework information
        self.td_steps = drl_info['framework']['apex']['td_steps']
        self.update_interval = drl_info['framework']['apex']['target_network']['update_interval']
        self.reset_flg = drl_info['framework']['apex']['buffer']['priority']['reset']['type'] is not None and drl_info['framework']['apex']['buffer']['priority']['reset']['type'] == 'target'

        # set architecture information
        self.architecture = drl_info['architecture']['type']

        # set state information
        self.state_info = copy.deepcopy(drl_info['state'])
        del self.state_info['vehicle']['number']
        
        # set reward information
        self.gamma = float(drl_info['reward']['common']['gamma'])

        # set update_count, episode, and session_df
        self.update_count = 0
        self.episode = 1
        self.session_df = None
        self.active_phases_df = None
        self.epsilon_record_df = None
        return
    
    def _initIntersections(self):
        self.intersections = Intersections(self)
        for intersection in self.network.intersections.getAll():
            if intersection.get('num_lanes_tuple') == tuple((self.num_lanes_map[road_id] for road_id in range(1, self.num_roads + 1))):
                self.intersections.add(intersection)
                intersection.set('master_agent', self)
        return
    
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
            raise NotImplementedError(f"Not supported number of roads: {self.num_roads}")
        
        return
        
    def _makeSaveDirPathMap(self):
        self.save_dir_path_map = {}

        # get drl_dir_path
        root_dir_path = self.network.get('root_dir_path')
        drl_dir_path = root_dir_path / 'data' / 'drl'
        drl_dir_path.mkdir(parents=True, exist_ok=True)
        
        # make drl_info
        drl_info = copy.deepcopy(self.config.get('drl_info'))

        # simulation_type, stop, training, and data_augmentation
        drl_info.pop('simulation_type')
        drl_info.pop('stop')
        drl_info.pop('training')
        drl_info.pop('data_augmentation')

        # framework
        saved_framework_info = drl_info['framework']
        for key in copy.deepcopy(drl_info['framework']):
            if key == 'type':
                continue
            if key == drl_info['framework']['type']:
                continue
            drl_info['framework'].pop(key)

        if drl_info['framework']['type'] == 'apex':
            drl_info['framework']['apex'].pop('local_agent')
            drl_info['framework']['apex'].pop('target_network')
            drl_info['framework']['apex']['buffer'].pop('priority')
        else:
            raise NotImplementedError(f"Not supported framework type: {saved_framework_info['type']}")
        
        # action
        drl_info.pop('action')

        # reward
        for key in copy.deepcopy(drl_info['reward']):
            if key == 'type':
                continue
            if key == 'common':
                continue
            if key == drl_info['reward']['type']:
                continue
            drl_info['reward'].pop(key)
        
        # architecture
        for key in copy.deepcopy(drl_info['architecture']):
            if key == 'type':
                continue
            if key == 'common':
                continue
            if key == drl_info['architecture']['type']:
                continue
            drl_info['architecture'].pop(key)

        for key in copy.deepcopy(drl_info['architecture']['common']['activation_function']):
            if key == 'type':
                continue
            if key == drl_info['architecture']['common']['activation_function']['type']:
                continue
            drl_info['architecture']['common']['activation_function'].pop(key)
        
        # get target config_dir_path
        found_flg = False
        for config_dir_path in drl_dir_path.glob('config_*'):
            config_file_path = config_dir_path / 'config.yaml'
            if not config_file_path.exists():
                continue

            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_yaml = yaml.safe_load(f)

            # check intersection shape
            for start_road_id in range(1, self.num_roads + 1):
                drl_info['shape'] = [self.num_lanes_map[((tmp_id - 1) // self.num_roads) + 1] for tmp_id in range(start_road_id, start_road_id + self.num_roads)]
                if config_yaml == drl_info:
                    found_flg = True
                    break
            
            # if found_flg is true, break the loop
            if found_flg:
                break
        
        if not found_flg:
            # reset shape information
            drl_info['shape'] = [self.num_lanes_map[road_id] for road_id in range(1, self.num_roads + 1)]

            # search empty config directory
            config_id = 1
            while True:
                config_dir_path = drl_dir_path / f"config_{config_id}"
                if not config_dir_path.exists():
                    config_dir_path.mkdir(parents=True, exist_ok=False)
                    config_file_path = config_dir_path / 'config.yaml'
                    with config_file_path.open('w', encoding='utf-8') as f:
                        yaml.dump(drl_info, f)
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
            self.model = ProtoQNet(self, requires_grad=True)
            self.model.showInfo('parameters')
        else:
            raise NotImplementedError(f"Not supported architecture: {self.architecture}")
        self.model.train()
        self.model.to(self.device)

        if self.architecture == 'proto':
            self.target_model = ProtoQNet(self, requires_grad=False)
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
        if self.simulation_id > 1:
            self.update_count = self.shared_resources.get('update_count')
            self.session_df = self.shared_resources.get('session_df')
            self.active_phases_df = self.shared_resources.get('active_phases_df')
            self.epsilon_record_df = self.shared_resources.get('epsilon_record_df')
            self.episode = len(self.session_df) + 1

        else:
            session_info_file_path = self.save_dir_path_map['session'] / 'session.json'
            if session_info_file_path.exists():
                with session_info_file_path.open('r', encoding='utf-8') as f:
                    session_info = json.load(f)
                
                self.update_count = session_info['target_network']['update_count']

            session_df_file_path = self.save_dir_path_map['session'] / 'session.csv'
            if session_df_file_path.exists():
                with open(session_df_file_path, 'r', encoding='utf-8', newline='') as f:
                    self.session_df = pd.read_csv(f)
                
                self.episode = len(self.session_df) + 1
            
            active_phases_df_file_path = self.save_dir_path_map['session'] / 'active_phases.csv'
            if active_phases_df_file_path.exists():
                with open(active_phases_df_file_path, 'r', encoding='utf-8', newline='') as f:
                    self.active_phases_df = pd.read_csv(f)

            epsilon_record_df_file_path = self.save_dir_path_map['session'] / 'epsilon_record.csv'
            if epsilon_record_df_file_path.exists():
                with open(epsilon_record_df_file_path, 'r', encoding='utf-8', newline='') as f:
                    self.epsilon_record_df = pd.read_csv(f)
                    
        # load model
        if self.simulation_id > 1:
            self.model = self.shared_resources.get('model')
        
        else:
            model_file_path = self.save_dir_path_map['model'] / 'q_net.pth'
            if model_file_path.exists():
                self.model.load_state_dict(torch.load(model_file_path))
        
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        elif self.simulation_id > 1:
            self.target_model = self.shared_resources.get('target_model')
        else:
            target_model_file_path = self.save_dir_path_map['model'] / 'target_q_net.pth'
            if target_model_file_path.exists():
                self.target_model.load_state_dict(torch.load(target_model_file_path))
            else:
                self.target_model.load_state_dict(self.model.state_dict())

        # load optimizer
        if self.simulation_id > 1:
            self.optimizer = self.shared_resources.get('optimizer')
        else:
            optimizer_file_path = self.save_dir_path_map['optimizer'] / 'optimizer.pth'
            if optimizer_file_path.exists():
                self.optimizer.load_state_dict(torch.load(optimizer_file_path))
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
    
    def _toDevice(self, data):
        if isinstance(data, dict):
            return {key: self._toDevice(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._toDevice(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data
        
    def _showInfo(self, type, loss_list_map = None):
        print('==============================================')
        if type == 'training':
            print('status: training results')
            print(f"master agent id: {self.id}")
            print(f"update count: {self.update_count}/{self.update_interval}")

            for epoch, avg_loss in enumerate(loss_list_map['avg'], start=1):
                print(f"epoch [{epoch}/{self.num_epochs}]: average loss = {avg_loss:.3f}")

        elif type == 'result':
            print('status: total rewards')
            print(f"master agent id: {self.id}")
            print(f"average total reward: {self.avg_total_reward:.1f}")

        else:
            raise NotImplementedError(f"Not supported info type: {type}")
        
        return
    
    def _updateSession(self):
        if self.simulation_type == 'test':
            return
        
        # update session_df
        session_row = pd.DataFrame({
            'episode': self.episode,
            'total_reward': self.avg_total_reward,
            'update_interval': self.update_interval,
            'new_data_count': self.buffer.get('new_data_count'),
            'num_batches': self.buffer.get('num_batches'),
            'batch_size': self.buffer.get('batch_size'),
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'simulation_time': self.simulation_time,
            'seed': self.seed,
        }, index=[0])
        if self.session_df is None:
            self.session_df = session_row.copy()
        else:
            self.session_df = pd.concat([self.session_df, session_row], ignore_index=True)

        # update active_phases_df
        active_phases_info = {'episode': self.episode}
        for phase_id in range(1, self.num_max_phases + 1):
            if phase_id in self.active_phase_list:
                active_phases_info[f"phase_{phase_id}"] = 1
            else:
                active_phases_info[f"phase_{phase_id}"] = 0
        active_phases_row = pd.DataFrame(active_phases_info, index=[0])

        if self.active_phases_df is None:
            self.active_phases_df = active_phases_row.copy()
        else:
            self.active_phases_df = pd.concat([self.active_phases_df, active_phases_row], ignore_index=True)

        # update epsilons_df
        epsilon_info = {'episode': self.episode}
        for agent in self.local_agents.getAll():
            epsilon_info[f"intersection_{agent.intersection.get('id')}"] = agent.get('epsilon')
        epsilon_record_row = pd.DataFrame(epsilon_info, index=[0])

        if self.epsilon_record_df is None:
            self.epsilon_record_df = epsilon_record_row.copy()
        else:
            self.epsilon_record_df = pd.concat([self.epsilon_record_df, epsilon_record_row], ignore_index=True)
        return
    
    
    def update(self, type):
        if type == 'buffer':
            self.buffer.update(learning_data_list=self.learning_data_list)

        elif type == 'session':
            self._updateSession()

        else:
            raise NotImplementedError(f"Not supported update type: {type}")
        
        return
    
    def train(self):
        if self.simulation_type == 'test':
            return
        if not self.buffer.get('change_flg'):
            return

        # sample data from buffer
        data_id_list, learning_data_list = self.buffer.sample(self.num_batches * self.batch_size)

        # if the number of data is smaller than batch size, do not train and return
        if len(data_id_list) < self.batch_size:
            return
        
        loss_list_map = {key: [] for key in ['avg', 'max', 'min']}
        for epoch in range(1, self.num_epochs + 1):
            # shuffle data_id_list
            id_list = list(range(len(data_id_list)))
            random.shuffle(id_list)
            shuffled_data_id_list = [data_id_list[i] for i in id_list]
            shuffled_learning_data_list = [learning_data_list[i] for i in id_list]
            shuffled_priority_list = []
            loss_list = []
            for batch in range(1, self.num_batches + 1):
                # when the number of data is smaller than batch size, break the loop
                if batch * self.batch_size > len(shuffled_data_id_list):
                    break

                # get mini batch data
                batch_learning_data_list = shuffled_learning_data_list[(batch - 1) * self.batch_size : batch * self.batch_size]
                batch_learning_data = default_collate(batch_learning_data_list)
                batch_learning_data = self._toDevice(batch_learning_data)

                self.optimizer.zero_grad()
                
                self.model.train()
                q_values = self.model(batch_learning_data['state']).gather(dim=1, index=(batch_learning_data['action'].long() - 1))
                
                self.model.eval()
                with torch.no_grad():
                    max_actions = self.model(batch_learning_data['next_state']).argmax(dim=1, keepdim=True)
                    target_q_values = self.target_model(batch_learning_data['next_state']).gather(dim=1, index=max_actions)
                    td_targets = batch_learning_data['cumulative_reward'] + (1 - batch_learning_data['done_flg']) * (self.gamma ** self.td_steps) * target_q_values
                self.model.train()

                loss = self.criterion(q_values, td_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.norm_clip)
                self.optimizer.step()
                loss_list.append(loss.item())

                # synchronize target model if update interval is reached
                self.update_count += 1
                if self.update_count >= self.update_interval:
                    self.target_model.load_state_dict(self.model.state_dict())
                    self.update_count = 0

                    if self.reset_flg:
                        self.buffer.reset('priority')

                if self.update_count % 100 == 0:
                    self.model.showInfo('gradient')
                
                if epoch == self.num_epochs:
                    shuffled_priority_list.extend(torch.abs(q_values - td_targets).squeeze().detach().cpu().numpy().tolist())

            # update avg_loss_list
            loss_list_map['avg'].append(np.mean(loss_list).item())
            loss_list_map['max'].append(np.max(loss_list).item())
            loss_list_map['min'].append(np.min(loss_list).item())
        
        # update priorities and initial_priority
        zipped_priority_data = sorted(zip(shuffled_data_id_list, shuffled_priority_list))
        _, priority_tuple = zip(*zipped_priority_data)
        self.buffer.update(data_id_list=data_id_list, priority_list=list(priority_tuple))
        self.buffer.set('initial_priority', max(loss_list_map['max']))
        
        # show training results and gradients
        self._showInfo('training', loss_list_map)

        # reset new_data_count
        self.buffer.reset('new_data_count')
        return
    
    def clearLearningData(self):
        for local_agent in self.local_agents.getAll():
            learning_data_list = local_agent.get('learning_data_list')
            learning_data_list.clear()
        
        return
            
    def save(self):
        if self.simulation_type == 'test':
            return
        
        # save session information and tree data
        with open(self.save_dir_path_map['session'] / 'session.json', 'w', encoding='utf-8') as f:
            json.dump({
                'episode': self.episode,
                'target_network': {
                    'update_count': self.update_count,
                },
                'buffer': {
                    'new_data_count': self.buffer.get('new_data_count'),
                    'next_data_id': self.buffer.get('next_data_id'),
                    'current_size': self.buffer.get('current_size'),
                }
            }, f)
        
        self.buffer.save()
        
        # save session_df, phase_probs_df, and epsilon_record_df
        with open(self.save_dir_path_map['session'] / 'session.csv', 'w', encoding='utf-8', newline='') as f:
            self.session_df.to_csv(f, index=False)
        
        with open(self.save_dir_path_map['session'] / 'active_phases.csv', 'w', encoding='utf-8', newline='') as f:
            self.active_phases_df.to_csv(f, index=False)
        
        with open(self.save_dir_path_map['session'] / 'epsilon_record.csv', 'w', encoding='utf-8', newline='') as f:
            self.epsilon_record_df.to_csv(f, index=False)

        torch.save(self.model.state_dict(), self.save_dir_path_map['model'] / 'q_net.pth')
        torch.save(self.target_model.state_dict(), self.save_dir_path_map['model'] / 'target_q_net.pth')
        torch.save(self.optimizer.state_dict(), self.save_dir_path_map['optimizer'] / 'optimizer.pth')

        # set properties to shared resources
        self.shared_resources.set('model', self.model)
        self.shared_resources.set('target_model', self.target_model)
        self.shared_resources.set('optimizer', self.optimizer)
        self.shared_resources.set('update_count', self.update_count)
        self.shared_resources.set('session_df', self.session_df)
        self.shared_resources.set('active_phases_df', self.active_phases_df)  
        self.shared_resources.set('epsilon_record_df', self.epsilon_record_df)
        self.shared_resources.set('buffer', self.buffer)
        return
    
    def showInfo(self, type):
        self._showInfo(type)
        return
        




    

            
    




