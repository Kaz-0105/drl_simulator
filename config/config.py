import yaml
from libs.object import Object 

class Config(Object):
    def __init__(self):
        super().__init__()

        self.readConfigFile()
    
    def readConfigFile(self):
        with open('layout/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

            self.layout = data['simulator']['layout']
            self.simulation_time = data['simulator']['simulation_time']



