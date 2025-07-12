from libs.common import Common
from objects.network import Network
from objects.simulation import Simulation

import win32com.client
import os

class Vissim(Common):
    def __init__(self, config, executor):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config
        
        # 非同期処理オブジェクトを初期化
        self.executor = executor

        # VissimのCOMオブジェクトを取得
        self._getVissimCom()

        # 下位のオブジェクトを初期化
        self.simulation = Simulation(self)
        self.network = Network(self)
    
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
        return

        
        
