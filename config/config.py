import yaml
import pandas as pd
from libs.common import Common 

class Config(Common):
    def __init__(self):
        super().__init__()

        self.readConfigFile()

        self.roads = pd.read_csv('layout/' + self.network_name + '/roads.csv')
        self.road_link_tags = pd.read_csv('layout/' + self.network_name + '/road_link_tags.csv')
        self.intersections = pd.read_csv('layout/' + self.network_name + '/intersections.csv')
        self.intersection_road_tags = pd.read_csv('layout/' + self.network_name + '/intersection_road_tags.csv')
        self.link_input_tags = pd.read_csv('layout/' + self.network_name + '/link_input_tags.csv')
        self.intersection_turn_ratio_tags = pd.read_csv('layout/' + self.network_name + '/intersection_turn_ratio_tags.csv')

        self.getNumRoadTurnRatioMap()
    
    def readConfigFile(self):
        with open('layout/config.yaml', 'r') as file:
            data = yaml.safe_load(file)
            self.network_name = data['simulator']['network_name']
            self.simulation_time = data['simulator']['simulation_time']

    def getNumRoadTurnRatioMap(self):
        self.num_road_turn_ratio_map = {}
        for num_roads in [3, 4, 5]:
            self.num_road_turn_ratio_map[num_roads] = pd.read_csv('layout/turn_ratio_templates' + str(num_roads) + '.csv')
        
            



