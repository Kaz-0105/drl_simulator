import yaml
import pandas as pd
from libs.object import Object 

class Config(Object):
    def __init__(self):
        super().__init__()

        self.readConfigFile()
        self.readRoads()
        self.readIntersections()
    
    def readConfigFile(self):
        with open('layout/config.yaml', 'r') as file:
            data = yaml.safe_load(file)
            self.network_name = data['simulator']['network_name']
            self.simulation_time = data['simulator']['simulation_time']

    def readRoads(self):
        self.roads_df = pd.read_csv('layout/' + self.network_name + '/roads.csv')
        self.road_link_tags_df = pd.read_csv('layout/' + self.network_name + '/road_link_tags.csv')

    def readIntersections(self):
        self.intersections_df = pd.read_csv('layout/' + self.network_name + '/intersections.csv')
        self.intersection_road_tags_df = pd.read_csv('layout/' + self.network_name + '/intersection_road_tags.csv')
            



