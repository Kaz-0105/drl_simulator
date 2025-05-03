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
        intersections = self.config.get('intersections')
        for _, intersection in intersections.iterrows():
            self.add(Intersection(intersection, self))

class Intersection(Object):
    def __init__(self, intersection, intersections):
        super().__init__()
        self.config = intersections.config
        self.intersections = intersections
        self.id = int(intersection['id'])
        self.num_roads = int(intersection['num_roads'])

        self.connectRoads()
    
    def connectRoads(self):
        self.input_roads = Roads(self, {'type': 'input'})
        self.output_roads = Roads(self, {'type': 'output'})
    
    def getNetwork(self):
        return self.intersections.network

