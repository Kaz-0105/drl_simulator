from libs.common import Common
from objects.network import Network
from objects.simulation import Simulation

import win32com.client
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
from datetime import datetime
import shutil
from pathlib import Path

class Vissim(Common):
    def __init__(self, config, executor, shared_resources, simulation_count):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期オブジェクトを設定
        self.config = config
        self.executor = executor
        self.shared_resources = shared_resources

        # シミュレーションのカウントを設定
        self.simulation_count = simulation_count

        # VissimのCOMオブジェクトを取得
        self._getVissimCom()

        # 下位のオブジェクトを初期化
        self.simulation = Simulation(self)
        self.network = Network(self)

        # 設定ファイルの変更を監視するためのイベントハンドラを設定
        self.config_change_handler = ConfigChangeHandler(self)
    
    def _getVissimCom(self):
        simulator_info = self.config.get('simulator_info')
        network_name = simulator_info['network_name']
        self.com = win32com.client.Dispatch('Vissim.Vissim')
        
        self.com.LoadNet(os.getcwd() + '\\layout\\' + network_name + '\\network.inpx')
        self.com.LoadLayout(os.getcwd() + '\\layout\\' + network_name + '\\network.layx')
        return
    
    def run(self):
        self.simulation.run()
    
    def exit(self):
        self.com.Exit()
        self.config_change_handler.stop()
        return
    
    def backup(self):
        # バックアップする必要がないときはスキップ
        if not self.backup_flg:
            return
        
        src_dirs = [Path('buffers'), Path('models')]
        backup_root = Path('backup')

        for src_dir in src_dirs:
            # バックアップ先ディレクトリに日時付きのフォルダを作る
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = backup_root / f"{src_dir.name}_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=False)

            # ファイルをコピー
            for item in src_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, backup_dir / item.name)

            # 5個以上のバックアップがある場合は古いものを削除
            backup_dirs = sorted(backup_root.glob(f"{src_dir.name}_*"), key=lambda x: x.name)
            if len(backup_dirs) > 5:
                for old_backup in backup_dirs[:-5]:
                    shutil.rmtree(old_backup)

            print(f"Backup of {src_dir} completed to {backup_dir}")
    
    @property
    def backup_flg(self):
        simulator_info = self.config.get('simulator_info')
        if not simulator_info['backup']['flg']:
            return False
        
        if self.simulation_count % simulator_info['backup']['interval'] != 0:
            return False
        
        if simulator_info['control_method'] != 'drl':
            return False
        
        return True

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
    
    def _initObserver(self):
        self.observer = Observer()
        self.observer.schedule(self, path='layout', recursive=True)
        self.observer.start()

    def on_modified(self, event):
        if event.src_path.endswith('phases3.csv'):
            # 実装したときに追加する
            pass
        elif event.src_path.endswith('phases4.csv'):
            # DRLでない場合は何もしない
            if self.control_method != 'drl':
                return
            
            # phases4.csvの内容を読み込み、local_agentsのrandom_phase_probsを更新
            phases = pd.read_csv(event.src_path)
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
        
        return

    def stop(self):
        self.observer.stop()
        self.observer.join()
        return

        
        
