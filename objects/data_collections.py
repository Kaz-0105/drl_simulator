from libs.container import Container
from libs.object import Object

import pandas as pd

class DataCollectionPoints(Container):
    def __init__(self, upper_object):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.com = self.network.com.DataCollectionPoints

            self._initElements(upper_object)
        
        elif upper_object.__class__.__name__ == 'Link':
            self.link = upper_object
        
        elif upper_object.__class__.__name__ == 'DataCollectionMeasurement':
            self.data_collection_measurement = upper_object
            self.network = self.data_collection_measurement.network

            self._initElements(upper_object)
        
        return
    
    def _initElements(self, upper_object):
        if upper_object.__class__.__name__ == 'Network':
            for data_collection_point_com in self.com.GetAll():
                self.add(DataCollectionPoint(data_collection_point_com, self))

        elif upper_object.__class__.__name__ == 'DataCollectionMeasurement':
            for data_collection_point_com in self.data_collection_measurement.com.DataCollectionPoints.GetAll():
                data_collection_point = self.network.data_collection_points[data_collection_point_com.AttValue('No')]
                self.add(data_collection_point)
                data_collection_point.data_collection_measurements.add(self.data_collection_measurement)

        else:
            raise NotImplementedError(f"Not supported upper_object class: {upper_object.__class__.__name__}")
        
        return

class DataCollectionPoint(Object):
    def __init__(self, com, data_collection_points):
        super().__init__()

        self.config = data_collection_points.config
        self.executor = data_collection_points.executor
        self.data_collection_points = data_collection_points
        self.network = data_collection_points.network

        self.com = com

        self._initProps()
        self._connectObjects()
        return
    
    @property
    def flow_rate(self):
        if self.data_collection_measurement is None:
            raise Exception(f"No single type data collection measurement found for DataCollectionPoint {self.id}, so flow rate is not available")
        
        return self.data_collection_measurement.get('flow_rate')
    
    def _initProps(self):
        self.id = self.com.AttValue('No')

        # set type (DataCollectionPoint._connectObjects())
        self.type = None 

        # set route_id (DataCollectionPoint._connectObjects())
        self.route_id = None
        return
    
    def _connectObjects(self):
        # set link and lane
        self.link = self.network.links[self.com.Lane.Link.AttValue('No')]
        self.link.data_collection_points.add(self)
        self.lane = self.link.lanes[self.com.Lane.AttValue('Index')]
        self.lane.data_collection_point = self

        # set data_collection_measurements (DataCollectionMeasurement._connectObjects())
        self.data_collection_measurements = DataCollectionMeasurements(self)
        self.data_collection_measurement = None

        # set type (intersection, input, output)
        if self.link.get('type') == 'connector':
            if self.link.intersection is not None:
                self.type = 'intersection'
            else:
                raise NotImplementedError(f"Unsupported connector link for DataCollectionPoint {self.id} on Link {self.link.get('id')}: intersection is None")
        else:
            if self.link.road.input_intersection is not None:
                self.type = 'output'
            elif self.link.road.output_intersection is not None:
                self.type = 'input'
            else:
                raise NotImplementedError(f"Unsupported road connection for DataCollectionPoint {self.id} on Link {self.link.get('id')}: input_intersection is None and output_intersection is None")
        
        # set vehicle_route, signal_head and route_id
        if self.type == 'intersection':
            self.vehicle_route = self.link.vehicle_route
            self.signal_head = self.link.signal_head
            self.route_id = self.vehicle_route.get('route_id')

        # set road
        if self.type == 'intersection':
            self.road = self.link.from_link.road
            self.road.data_collection_points.add(self)
        elif self.type in ['input', 'output']:
            self.road = self.link.road
            self.road.data_collection_points.add(self)
        else:
            raise NotImplementedError(f"Not supported data collection point type: {self.type}")
        return

    def getFlowRate(self, duration_step=None):
        if self.data_collection_measurement is None:
            raise Exception(f"No single type data collection measurement found for DataCollectionPoint {self.id}, so flow rate is not available")
        
        return self.data_collection_measurement.getFlowRate(duration_step=duration_step)
    
class DataCollectionMeasurements(Container):
    def __init__(self, upper_object):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.com = self.network.com.DataCollectionMeasurements

            self._initElements()
        
        elif upper_object.__class__.__name__ == 'DataCollectionPoint':
            self.data_collection_point = upper_object
        
        else: 
            raise NotImplementedError(f"Not supported upper_object class: {upper_object.__class__.__name__}")
        
        return

    def _initElements(self):
        for data_collection_measurement_com in self.com.GetAll():
            self.add(DataCollectionMeasurement(data_collection_measurement_com, self))
    
    def update(self):
        measurement_ids = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('No')]
        veh_nums = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('Vehs(Current, Last, All)')]

        for index, measurement_id in enumerate(measurement_ids):
            measurement = self[measurement_id]
            self.executor.submit(measurement.update, veh_nums[index])
    
class DataCollectionMeasurement(Object):
    DURATION = 60 # seconds
    def __init__(self, com, data_collection_measurements):
        super().__init__()

        self.config = data_collection_measurements.config
        self.executor = data_collection_measurements.executor
        self.data_collection_measurements = data_collection_measurements
        self.network = data_collection_measurements.network

        self.com = com

        self._initProps()
        self._connectObjects()
        return
        
    @property
    def current_time(self):
        return self.network.simulation.get('current_time')
    
    @property
    def time_step(self):
        return self.network.simulation.get('time_step')
    
    @property
    def flow_rate(self):
        if self.type != 'single':
            raise Exception(f"Flow rate is only available for single type data collection measurement, but the type of DataCollectionMeasurement {self.id} is {self.type}")
        
        if len(self.num_vehs_record) > self.duration_step:
            num_vehs_sum = self.num_vehs_record['num_vehs'][-self.duration_step:].sum()
            return num_vehs_sum / (self.duration_step * self.time_step) # [veh/second]
        else:
            num_vehs_sum = self.num_vehs_record['num_vehs'].sum()
            return num_vehs_sum / (len(self.num_vehs_record) * self.time_step) # [veh/second]
        

    def _initProps(self):
        self.id = self.com.AttValue('No')
        self.duration_step = int(DataCollectionMeasurement.DURATION / self.time_step)
        self.type = None 

        self.current_num_vehs = 0
        self.num_vehs_record = pd.DataFrame(columns=['time', 'num_vehs'])
        return
    
    def _connectObjects(self):
        self.data_collection_points = DataCollectionPoints(self)
        self.type = 'single' if self.data_collection_points.count() == 1 else 'multiple'

        if self.type == 'single':
            self.data_collection_point = self.data_collection_points.getAll()[0]
            self.data_collection_point.data_collection_measurement = self
        return
    
    def update(self, num_vehs):
        self.current_num_vehs = 0 if num_vehs is None else num_vehs
        self.num_vehs_record.loc[len(self.num_vehs_record)] = [self.current_time, self.current_num_vehs]
        return

    def getFlowRate(self, duration_step=None):
        if self.type != 'single':
            raise Exception(f"Flow rate is only available for single type data collection measurement, but the type of DataCollectionMeasurement {self.id} is {self.type}")
        
        if duration_step is None:
            duration_step = self.duration_step
        
        if len(self.num_vehs_record) > duration_step:
            num_vehs_sum = self.num_vehs_record['num_vehs'][-duration_step:].sum()
            return num_vehs_sum / (duration_step * self.time_step) # [veh/second]
        else:
            num_vehs_sum = self.num_vehs_record['num_vehs'].sum()
            return num_vehs_sum / (len(self.num_vehs_record) * self.time_step) # [veh/second]


    
