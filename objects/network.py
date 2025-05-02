from libs.object import Object
from objects.roads import Roads
from objects.intersections import Intersections
from objects.links import Links

class Network(Object):
    def __init__(self, vissim):
        super().__init__()
        self.config = vissim.config
        self.vissim = vissim

        self.com = self.vissim.com.Net

        self.roads = Roads(self)
        self.intersections = Intersections(self)
        self.links = Links(self)
