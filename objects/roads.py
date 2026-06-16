from libs.container import Container
from libs.object import Object
from objects.links import Links
from objects.signal_controllers import SignalGroups
from objects.queue_counters import QueueCounters
from objects.delay_measurements import DelayMeasurements
from objects.data_collections import DataCollectionPoints

import pandas as pd

from objects.travel_time_measurements import TravelTimeMeasurements


class Roads(Container): 
    def __init__(self, upper_object, type=None):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self._initElements(upper_object)

        elif upper_object.__class__.__name__ == 'Intersection':
            self.intersection = upper_object
            self.network = self.intersection.network
            self.type = type
            self._initElements(upper_object)
        
        else:
            raise NotImplementedError(f"Not supported upper_object: {upper_object.__class__.__name__}")
        
        return
    
    @property
    def max_length(self):
        max_length = 0
        for road in self.getAll():
            if road.get('length') > max_length:
                max_length = road.get('length')
        
        return max_length
    
    def _initElements(self, upper_object):
        if upper_object.__class__.__name__ == 'Network':
            roads_df = self.config.get('roads_df')
            for _, road_row in roads_df.iterrows():
                self.add(Road(road_row, self))

        elif upper_object.__class__.__name__ == 'Intersection':
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
                    road.output_intersection = self.intersection
                elif self.type == 'output':
                    road.input_intersection = self.intersection
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
                road.turn_ratios = turn_ratios
            return 
        
    def update(self):
        for road in self.getAll():
            road.update()
        self.executor.wait()        
        return
    
    def sync(self, type):
        for road in self.getAll():
            self.executor.submit(road.sync, type)

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
        self._connectObjects()
        return
    
    @property
    def close_threshold(self):
        return self.max_queue_length if self.max_queue_length > self.max_speed else self.max_speed

    @property
    def max_queue_length(self):
        return self.queue_counters.get('max_queue_length')
    
    @property
    def main_queue_length(self):
        if self.main_link is None:
            raise NotImplementedError("main_queue_length is only implemented for road with main_link.")
        
        return self.main_link.get('current_queue_length')
    
    @property
    def right_queue_length(self):
        if self.right_link is None:
            raise NotImplementedError("right_queue_length is only implemented for road with right_link.")
        
        return self.right_link.get('current_queue_length')
    
    @property
    def average_delay(self):
        delays = []
        for delay_measurement in self.delay_measurements.getAll():
            delays.append(delay_measurement.get('current_delay'))
        return sum(delays) / len(delays) if len(delays) > 0 else 0
    
    @property
    def num_vehicles(self):
        return self.vehicles_df.shape[0] if self.vehicles_df is not None else 0
    
    @property
    def inflow_rate(self):
        for data_collection_point in self.data_collection_points.getAll():
            if data_collection_point.get('type') != 'input':
                continue
            return data_collection_point.get('flow_rate')
        raise Exception(f"No input type data collection point found for Road {self.id}")
    
    @property
    def route_signal_color_map(self):
        route_signal_color_map = {}
        for route_id, signal_group_id in self.route_signal_group_map.items():
            signal_group = self.signal_groups[signal_group_id]
            route_signal_color_map[route_id] = signal_group.get('current_signal_color')
        
        return route_signal_color_map

    def _initProps(self, road_row):
        # set id, max_speed, type
        self.id = int(road_row['id'])
        self.max_speed = int(road_row['max_speed'])
        self.type = road_row['type']

        # set turn_ratios (Roads._initElements())
        self.turn_ratios = None

        # set inflow_volume and length (Links._connectObjects())
        self.inflow_volume = None
        self.length = None
    
        # initialize route_signal_group_map
        self.route_signal_group_map = {}

        # initialize records
        self.speed_record_list = []
        self.speed_record_df = None
        self.num_vehs_record_list = []
        self.num_vehs_record_df = None
        return
    
    def _connectObjects(self):
        # set link (Link._connectObjects)
        self.links = Links(self)
        self.main_link = None
        self.right_link = None
        self.left_link = None
        self.right_connector = None
        self.left_connector = None

        # set intersection (Intersection._connectObjects)
        self.input_intersection = None
        self.output_intersection = None
        
        # set queue_counters (QueueCounters._connectObjects)
        self.queue_counters = QueueCounters(self)

        # set delay_measurements (DelayMeasurements._connectObjects)
        self.delay_measurements = DelayMeasurements(self)

        # set signal_groups (SignalGroups._connectObjects)
        self.signal_groups = SignalGroups(self)

        # set data_collection_points (DataCollectionPoints._connectObjects)
        self.data_collection_points = DataCollectionPoints(self)

        # set vehicle_routing_decision (VehicleRoutingDecision._connectObjects)
        self.vehicle_routing_decision = None

        # set travel_time_measurements (TravelTimeMeasurement._connectObjects)
        self.travel_time_measurements = TravelTimeMeasurements(self)
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
            self.vehicles_df = pd.DataFrame(columns=['id', 'position', 'length', 'in_queue', 'speed', 'lane_id', 'link_id', 'road_id', 'route_id'])

        # if the road is not input road, skip the rest of the process    
        if self.output_intersection is None:
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

    def sync(self, type):
        if type == 'dataframe':
            self.speed_record_df = pd.DataFrame(self.speed_record_list)
            self.num_vehs_record_df = pd.DataFrame(self.num_vehs_record_list)
        else:
            raise NotImplementedError(f"Not supported type: {type}")
        return

    
