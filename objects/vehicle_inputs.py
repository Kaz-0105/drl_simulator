from libs.container import Container
from libs.object import Object

class VehicleInputs(Container):
    def __init__(self, network):
        super().__init__()
        self.config = network.config
        self.network = network
        self.com = self.network.com.VehicleInputs

        self.makeElements()

    def makeElements(self):
        for vehicle_input_com in self.com.GetAll():
            self.add(VehicleInput(vehicle_input_com, self))


class VehicleInput(Object):
    def __init__(self, com, vehicle_inputs):
        super().__init__()
        self.config = vehicle_inputs.config
        self.vehicle_inputs = vehicle_inputs
        self.com = com

        self.id = self.com.AttValue('No')

        self.connectLink()
    
    def connectLink(self):
        network = self.vehicle_inputs.network
        links = network.links

        link = links[int(self.com.AttValue('Link'))]
        self.link = link
        link.set('vehicle_input', self)
