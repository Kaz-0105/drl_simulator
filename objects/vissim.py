from libs.common import Common
import win32com.client
import os
from objects.network import Network
from objects.simulation import Simulation
from libs.executor import Executor

class Vissim(Common):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.executor = Executor(self.config.get('max_workers'))

        self.getVissimCom()

        self.simulation = Simulation(self)
        self.network = Network(self)

        print('test')
    
    def getVissimCom(self):
        network_name = self.config.network_name
        self.com = win32com.client.Dispatch('Vissim.Vissim')
        
        self.com.LoadNet(os.getcwd() + '\\layout\\' + network_name + '\\network.inpx')
        self.com.LoadLayout(os.getcwd() + '\\layout\\' + network_name + '\\network.layx')
    
    def run(self):
        self.simulation.run()
    
    def exit(self):
        self.com.Exit()

        
        
