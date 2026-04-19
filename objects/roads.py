from libs.container import Container
from libs.object import Object
from objects.links import Links
from objects.signal_controllers import SignalGroups
from objects.queue_counters import QueueCounters
from objects.delay_measurements import DelayMeasurements
from objects.data_collections import DataCollectionPoints

from pandas import DataFrame
import pandas as pd


class Roads(Container): 
    def __init__(self, upper_object, type=None):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self._initElements()

        elif upper_object.__class__.__name__ == 'Intersection':
            self.intersection = upper_object
            self.type = type
            self._initElements()
        
        else:
            raise NotImplementedError(f"Not supported upper_object: {upper_object.__class__.__name__}")
        
        return
    
    def _initElements(self):
        if self.has('network'):
            roads_df = self.config.get('roads_df')
            for _, road_row in roads_df.iterrows():
                self.add(Road(road_row, self))

        elif self.has('intersection'):
            # get intersection_road_tags_df
            tags_df = self.config.get('intersection_road_tags_df')
            target_tags_df = tags_df[
                (tags_df['intersection_id'] == self.intersection.get('id')) & 
                (tags_df['type'] == self.type)
            ]

            # connect intersection and roads
            roads = self.intersection.network.roads
            for _, tag_row in target_tags_df.iterrows():
                road = roads[tag_row['road_id']]
                self.add(road, tag_row['order_id'])

                if self.type == 'input':
                    road.set('output_intersection', self.intersection)
                elif self.type == 'output':
                    road.set('input_intersection', self.intersection)
                else:
                    raise NotImplementedError(f"Not supported road type: {self.type}")

            # validation of the number of roads
            if self.count() != self.intersection.get('num_roads'):
                raise Exception(f"Intersection {self.intersection.get('id')} has {self.intersection.get('num_roads')} roads, but roads object has {self.count()} {self.type} roads.")
            
            # when type is output, return here
            if self.type == 'output':
                return
            
            # set turn_ratios for each road
            tags_df = self.config.get('intersection_turn_ratio_tags_df')
            target_tags_df = tags_df[tags_df['intersection_id'] == self.intersection.get('id')]
            turn_ratio_df = self.config.get('turn_ratio_df_map')[self.count()]
    
            for _, tag_row in target_tags_df.iterrows():
                road_order_id = tag_row['road_order_id']
                turn_ratio_template_id = tag_row['turn_ratio_template_id']
                turn_ratio_record = turn_ratio_df[turn_ratio_df['id'] == turn_ratio_template_id]
                turn_ratios = {}
                for order_id in range(1, self.count()):
                    turn_ratios[order_id] = turn_ratio_record[f"ratio{order_id}"].to_numpy()[0]
                
                road = self[road_order_id]
                road.set('turn_ratios', turn_ratios)

            return 
        
    def update(self):
        for road in self.getAll():
            road.update()
        self.executor.wait()        
        return

    def syncDataFrame(self):
        for road in self.getAll():
            self.executor.submit(road.syncDataFrame)
        
        self.executor.wait()
        return

class Road(Object):
    def __init__(self, road_row, roads):
        super().__init__()

        # set objects
        self.config = roads.config
        self.executor = roads.executor
        self.roads = roads
        self.network = roads.network

        self._initProps(road_row)

        # initialize links, queue_counters, delay_measurements, signal_groups, data_collection_points, and vehicle_routing_decision objects
        self.links = Links(self)
        self.queue_counters = QueueCounters(self)
        self.delay_measurements = DelayMeasurements(self)
        self.signal_groups = SignalGroups(self)
        self.data_collection_points = DataCollectionPoints(self)
        self.vehicle_routing_decision = None
        return

    def _initProps(self, road_row):
        # set id, max_speed, type
        self.id = int(road_row['id'])
        self.max_speed = int(road_row['max_speed'])
        self.type = road_row['type']
    
        # initialize route_signal_group_map
        self.route_signal_group_map = {}

        # initialize records
        self.speed_record_list = []
        self.speed_record_df = None
        self.num_vehs_record_list = []
        self.num_vehs_record_df = None

        if self.network.get('control_method') == 'scoot':
            self.effective_storage_length_map = None
        return

    def update(self):
        self.links.update()
        self.executor.submit(self.summarizeData)
        return
    
    def summarizeData(self):
        # initialize vehicles_df
        self.vehicles_df = None
        
        for link in self.links.getAll():
            # get vehicles_df
            vehicles_df = link.get('vehicles_df').copy()

            # skip if vehicles_df is empty
            if vehicles_df.empty:
                continue
                
            # update vehicles_df
            self.vehicles_df = vehicles_df if self.vehicles_df is None else pd.concat([self.vehicles_df, vehicles_df], ignore_index=True)
        
        # sort
        if self.vehicles_df is not None:
            self.vehicles_df = self.vehicles_df.sort_values(by='position', ascending=False)
            self.vehicles_df = self.vehicles_df.reset_index(drop=True)

        # initialize vehicles_df if it is None
        if self.vehicles_df is None:
            self.vehicles_df = DataFrame(columns=['id', 'position', 'in_queue', 'speed', 'lane_id', 'link_id', 'road_id', 'route_id'])

        # if the road is not input road, skip the rest of the process    
        if not self.has('output_intersection'):
            return
        
        # calculate route_num_vehs_map
        self.route_num_vehs_map = {route_id: 0 for route_id in range(self.output_intersection.get('num_roads'))}    
        for _, vehicle_row in self.vehicles_df.iterrows():
            self.route_num_vehs_map[vehicle_row['route_id']] += 1

        # update speed_record_list
        self.speed_record_list.append({
            'time': int(self.network.get('current_time')),
            'value': self.vehicles_df['speed'].mean() if not self.vehicles_df.empty else self.max_speed,
        })

        # update num_vehs_record_list
        self.num_vehs_record_list.append({
            'time': int(self.network.get('current_time')),
            'value': self.vehicles_df.shape[0],
        })
        return

    def syncDataFrame(self):
        self.speed_record_df = DataFrame(self.speed_record_list)
        self.num_vehs_record_df = DataFrame(self.num_vehs_record_list)
        return
                
    @property
    def max_queue_length(self):
        return self.queue_counters.get('max_queue_length')

    @property
    def average_delay(self):
        delays = []
        for delay_measurement in self.delay_measurements.getAll():
            delays.append(delay_measurement.get('current_delay'))
        
        return sum(delays) / len(delays) if len(delays) > 0 else 0
    
    @property
    def num_vehicles(self):
        return self.vehicles_df.shape[0]
    
    @property
    def route_signal_color_map(self):
        route_signal_color_map = {}
        for route_id, signal_group_id in self.route_signal_group_map.items():
            signal_group = self.signal_groups[signal_group_id]
            route_signal_color_map[route_id] = signal_group.get('current_signal_color')
        
        return route_signal_color_map


    
