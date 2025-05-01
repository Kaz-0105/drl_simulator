from config.config import Config
from libs.object import Object
import win32com.client
import os

class Vissim(Object):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.getVissimCom()
    
    def getVissimCom(self):
        network_name = self.config.network_name
        self.com = win32com.client.Dispatch('Vissim.Vissim')
        
        self.com.LoadNet(os.getcwd() + '\\layout\\' + network_name + '\\network.inpx')
        self.com.LoadLayout(os.getcwd() + '\\layout\\' + network_name + '\\network.layx')

        
        
