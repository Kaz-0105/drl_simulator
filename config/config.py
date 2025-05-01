import yaml
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
        with open('layout/' + self.network_name + '/roads.yaml', 'r') as file:
            data = yaml.safe_load(file)
            self.roads = data['roads']
            self.road_link_tags = data['road_link_tags']

    def readIntersections(self):
        with open('layout/' + self.network_name + '/intersections.yaml', 'r') as file:
            data = yaml.safe_load(file)
            self.intersections = data['intersections']
            self.intersection_road_tags = data['intersection_road_tags']
            



