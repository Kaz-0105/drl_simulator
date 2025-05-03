from libs.object import Object
from objects.roads import Roads
from objects.intersections import Intersections
from objects.links import Links
from objects.vehicle_inputs import VehicleInputs

class Network(Object):
    def __init__(self, vissim):
        super().__init__()
        self.config = vissim.config
        self.vissim = vissim

        self.com = self.vissim.com.Net

        self.roads = Roads(self)
        self.intersections = Intersections(self)
        self.links = Links(self)
        self.vehicle_inputs = VehicleInputs(self)

        self.setParametersToVissim()

    def setParametersToVissim(self):
        for vehicle_input in self.vehicle_inputs.getAll():
            input_volume = vehicle_input.link.get('input_volume')
            vehicle_input.com.SetAttValue('Volume(1)', input_volume)

