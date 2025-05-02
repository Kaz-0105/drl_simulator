from config.config import Config
from libs.object import Object
import win32com.client
import os
from objects.network import Network

class Vissim(Object):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.getVissimCom()

        self.network = Network(self)
    
    def getVissimCom(self):
        network_name = self.config.network_name
        self.com = win32com.client.Dispatch('Vissim.Vissim')
        
        self.com.LoadNet(os.getcwd() + '\\layout\\' + network_name + '\\network.inpx')
        self.com.LoadLayout(os.getcwd() + '\\layout\\' + network_name + '\\network.layx')

        
        
