from libs.container import Container
from libs.object import Object

class TravelTimeMeasurements(Container):
    def __init__(self, upper_object):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.com = self.network.com.VehicleTravelTimeMeasurements

            self._initElements()

        elif upper_object.__class__.__name__ == 'VehicleRoutingDecision':
            self.vehicle_routing_decision = upper_object
            self.network = upper_object.network
        
        elif upper_object.__class__.__name__ == 'Link':
            self.link = upper_object
            self.network = upper_object.network
    
    def _initElements(self):
        for travel_time_measurement_com in self.com.GetAll():
            self.add(TravelTimeMeasurement(travel_time_measurement_com, self))
        return

class TravelTimeMeasurement(Object):
    def __init__(self, com, travel_time_measurements):
        super().__init__()

        self.config = travel_time_measurements.config
        self.executor = travel_time_measurements.executor
        self.travel_time_measurements = travel_time_measurements
        self.network = travel_time_measurements.network

        self.com = com

        self._initProps()
        self._connectObjects()
        return
    
    def _initProps(self):
        # set id
        self.id = self.com.AttValue('No')

        # set route_id (VehicleRoute._connectObjects())
        self.route_id = None
        return
    
    def _connectObjects(self):
        # set link
        self.start_link = self.network.links[int(self.com.StartLink.AttValue('No'))]
        self.end_link = self.network.links[int(self.com.EndLink.AttValue('No'))]

        if self.end_link.travel_time_measurements is None:
            self.end_link.travel_time_measurements = TravelTimeMeasurements(self.end_link)
        
        self.start_link.travel_time_measurements.add(self)
        self.end_link.travel_time_measurements.add(self)

        # set road
        self.road = self.start_link.road
        self.road.travel_time_measurements.add(self)

        # set vehicle_route and vehicle_routing_decision
        self.vehicle_route = self.end_link.vehicle_route
        self.vehicle_route.travel_time_measurement = self
        self.route_id = self.vehicle_route.get('route_id')
        self.vehicle_routing_decision = self.vehicle_route.vehicle_routing_decision
        self.vehicle_routing_decision.travel_time_measurements.add(self)
        return