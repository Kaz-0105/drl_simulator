import yaml
from libs.object import Object 

class Config(Object):
    def __init__(self):
        super().__init__()

        self.readConfigFile()
        self.readRoads()
    
    def readConfigFile(self):
        with open('layout/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

            self.network_name = data['simulator']['network_name']
            self.simulation_time = data['simulator']['simulation_time']

    def readRoads(self):
        with open('layout/' + self.network_name + '/roads.yaml', 'r') as file:
            self.roads = yaml.safe_load(file)

            for road_id in range(len(self.roads)):
                road = self.roads[road_id]
                intersection_ids = road['intersection_ids']
                road.pop('intersection_ids')
                road['intersections'] = {'input': intersection_ids[0], 'output': intersection_ids[1]}

            print(self.roads)
            



