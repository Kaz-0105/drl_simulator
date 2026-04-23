from libs.common import Common
from config.config import Config
from libs.executor import Executor
from libs.shared_resource import SharedResources
from objects.network import Network
from objects.simulation import Simulation

import win32com.client
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
import time
import torch


class Vissim(Common):
    def __init__(self, root_dir_path):
        super().__init__()

        # set root_dir_path
        self.root_dir_path = root_dir_path

        # initialize config, executor, and shared_resources
        self.config = Config(self)
        self.executor = Executor(self)
        self.shared_resources = SharedResources(self)

        # init properties
        self._initProps()

        # init simulation and network
        self.simulation = Simulation(self)
        self.network = Network(self)

        # set config change handler
        if self.config_change_flg:
            self.config_change_handler = ConfigChangeHandler(self)
        
        # set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return

    def _initProps(self):
        # set layout file path map
        simulator_info = self.config.get('simulator_info')
        self.layout_file_path_map = {
            'inpx': self.root_dir_path / 'layout' / simulator_info['layout_name'] / 'network.inpx',
            'layx': self.root_dir_path / 'layout' / simulator_info['layout_name'] / 'network.layx'
        }
        self.layout_dir_path = self.root_dir_path / 'layout' / simulator_info['layout_name']

        # set change_handle_flg
        self.config_change_flg = simulator_info['config_change']['flg']

        # set control method
        self.control_method = simulator_info['control_method']

        # set simulation count
        self.simulation_id = 1
        return

    def _activate(self):
        # update config if needed
        if self.config.get('update_flg'):
            self.config.update()

        # set com object
        while True:
            try:
                self.com = win32com.client.Dispatch('Vissim.Vissim')
                break
            except Exception as e:
                self._showInfo('fail_com_connection')
                time.sleep(1)
        
        # load network and layout
        self.com.LoadNet(self.layout_file_path_map['inpx'])
        self.com.LoadLayout(self.layout_file_path_map['layx'])

        # quick mode
        self.com.Graphics.SetAttValue('QuickMode', True)

        # activate simulation and network
        self.simulation.activate(self.simulation_id)
        self.network.activate()
        return

    def _deactivate(self):
        self.com.Exit()
        if self.config_change_flg:
            self.config_change_handler.stop()
        return
    
    def _showInfo(self, type):
        if type == 'fail_com_connection':
            print('status: fail to connect to vissim com interface. retrying...')
        else:
            raise NotImplementedError(f"Not supported info type: {type}")
        return
    
    def run(self):
        while True:
            self._activate()
            self.simulation.run()
            self._deactivate()

            if self.simulation.get('finish_flg'):
                break

            self.simulation_id += 1
        
        self.config.stopObserver()
        self.executor.shutdown()
        return

class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, vissim):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = vissim.config
        self.vissim = vissim

        # 制御方法を取得
        simulator_info = self.config.get('simulator_info')
        self.control_method = simulator_info['control_method']

        # observerを初期化
        self._initObserver()
        return
    
    def _initObserver(self):
        self.observer = Observer()
        self.observer.schedule(self, path='layout', recursive=True)
        self.observer.start()
        return

    def on_modified(self, event):
        if event.src_path.endswith('phases3.csv'):
            # 実装したときに追加する
            pass
        elif event.src_path.endswith('phases4.csv'):
            # DRLでない場合は何もしない
            if self.control_method != 'drl':
                return
            
            # phases4.csvの内容を読み込み、local_agentsのrandom_phase_probsを更新
            phases = pd.read_csv(event.src_path, index_col=False)
            random_phase_probs = {}
            for _, row in phases.iterrows():
                random_phase_probs[int(row['id'])] = float(row['random_prob'])
            
            # 確率の合計を1に正規化
            sum_probs = sum(random_phase_probs.values())
            for key in random_phase_probs.keys():
                random_phase_probs[key] /= sum_probs
            
            network = self.vissim.network
            for local_agent in network.local_agents.getAll():
                # 十字路でない場合はスキップ
                if local_agent.get('num_roads') != 4:
                    continue

                # local_agentのrandom_phase_probsを更新
                local_agent.set('random_phase_probs', random_phase_probs)
            
            # configオブジェクトの更新
            num_roads_phases_map = self.config.get('num_roads_phases_map')
            num_roads_phases_map[4] = phases

        elif event.src_path.endswith('phases5.csv'):
            # 実装したときに修正する
            pass
            
        elif event.src_path.endswith('config.yaml'):
            # 設定ファイルが変更された場合、設定を再読み込み
            self.config.readConfigFile()
            self.config.reshapeDrlInfo()
        
        elif event.src_path.endswith('epsilon_schedule.csv'):
            epsilon_schedule = pd.read_csv(event.src_path, index_col=False)
            self.config.set('epsilon_schedule', epsilon_schedule)
        return

    def stop(self):
        self.observer.stop()
        self.observer.join()
        return

        
        
