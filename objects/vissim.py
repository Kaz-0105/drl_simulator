from libs.common import Common
import win32com.client
import os
from objects.network import Network
from objects.simulation import Simulation
from libs.executor import Executor

class Vissim(Common):
    def __init__(self, config):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = config
        
        # 非同期処理オブジェクトを初期化
        self.executor = Executor(self)

        # VissimのCOMオブジェクトを取得
        self.getVissimCom()

        # 下位のオブジェクトを初期化
        self.simulation = Simulation(self)
        self.network = Network(self)
    
    def getVissimCom(self):
        simulator_info = self.config.get('simulator_info')
        network_name = simulator_info['network_name']
        self.com = win32com.client.Dispatch('Vissim.Vissim')
        
        self.com.LoadNet(os.getcwd() + '\\layout\\' + network_name + '\\network.inpx')
        self.com.LoadLayout(os.getcwd() + '\\layout\\' + network_name + '\\network.layx')
    
    def run(self):
        self.simulation.run()
    
    def exit(self):
        self.com.Exit()

        
        
