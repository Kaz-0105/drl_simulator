from libs.object import Object
from objects.roads import Roads

class Network(Object):
    def __init__(self, vissim):
        super().__init__()
        self.config = vissim.config
        self.vissim = vissim

        self.roads = Roads(self)
