from libs.container import Container
from libs.object import Object

class Roads(Container): 
    def __init__(self, network):
        super().__init__()
        self.config = network.config
        self.network = network

        self.makeElements()
        print('test')
    
    def makeElements(self):
        roads_df = self.config.get('roads_df')
        for index, road_df in roads_df.iterrows():
            self.add(Road(road_df, self))

class Road(Object):
    def __init__(self, road_df, roads):
        super().__init__()
        self.config = roads.config
        self.roads = roads
        self.id = int(road_df['id'])
        self.max_speed = int(road_df['max_speed'])
