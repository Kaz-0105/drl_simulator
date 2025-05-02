from libs.container import Container
from libs.object import Object
from objects.roads import Roads

class Intersections(Container):
    def __init__(self, network):
        super().__init__()
        self.config = network.config
        self.network = network

        self.makeElements()

    def makeElements(self):
        intersections_df = self.config.get('intersections_df')
        for _, intersection_df in intersections_df.iterrows():
            self.add(Intersection(intersection_df, self))

class Intersection(Object):
    def __init__(self, intersection_df, intersections):
        super().__init__()
        self.config = intersections.config
        self.intersections = intersections
        self.id = int(intersection_df['id'])
        self.num_roads = int(intersection_df['num_roads'])

        self.connectRoads()
    
    def connectRoads(self):
        self.input_roads = Roads(self, {'type': 'input'})
        self.output_roads = Roads(self, {'type': 'output'})
    
    def getNetwork(self):
        return self.intersections.network

