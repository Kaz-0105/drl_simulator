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
    def __init__(self, upper_object, options = None):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self._initElements()

        elif upper_object.__class__.__name__ == 'Intersection':
            self.intersection = upper_object
            self.type = options['type']
            self._initElements()
        
        else:
            raise NotImplementedError(f"Not supported upper_object: {upper_object.__class__.__name__}")
        
        return
    
    def _initElements(self):
        if self.has('network'):
            roads = self.config.get('roads')
            for _, road_row in roads.iterrows():
                self.add(Road(road_row, self))

        elif self.has('intersection'):
            tags = self.config.get('intersection_road_tags')
            target_tags = tags[(tags['intersection_id'] == self.intersection.get('id')) & (tags['type'] == self.type)]

            network = self.intersection.network
            roads = network.roads

            for _, tag in target_tags.iterrows():
                road = roads[tag['road_id']]
                self.add(road, tag['order_id'])

                if self.type == 'input':
                    road.set('output_intersection', self.intersection)
                elif self.type == 'output':
                    road.set('input_intersection', self.intersection)
                
            if self.count() != self.intersection.get('num_roads'):
                raise Exception(f"Intersection {self.intersection.get('id')} has {self.intersection.get('num_roads')} roads, but roads object has {self.count()} {self.type} roads.")
            
            # 流出道路の場合はここで終わり
            if self.type == 'output':
                return
            
            # turn_ratioについての設定
            tags = self.config.get('intersection_turn_ratio_tags')
            target_tags = tags[tags['intersection_id'] == self.intersection.get('id')]
            turn_ratio_templates = self.config.get('num_roads_turn_ratio_map')[self.count()]
    
            for _, tag in target_tags.iterrows():
                road_order_id = tag['road_order_id']
                turn_ratio_template_id = tag['turn_ratio_template_id']
                turn_ratio_record = turn_ratio_templates[turn_ratio_templates['id'] == turn_ratio_template_id]
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
        return

    def _initProps(self, road_row):
        # set id, max_speed, type
        self.id = int(road_row['id'])
        self.max_speed = int(road_row['max_speed'])
        self.type = road_row['type']

        # initialize links
        self.links = Links(self)

        # initialize queue_counters, delay_measurements, signal_groups
        self.queue_counters = QueueCounters(self)
        self.delay_measurements = DelayMeasurements(self)
        self.signal_groups = SignalGroups(self)

        # initialize direction_signal_group_map
        self.direction_signal_group_map = {}

        # initialize data_collection_points
        self.data_collection_points = DataCollectionPoints(self)

        # initialize records
        self.speed_record_list = []
        self.speed_record_df = None
        self.num_vehs_record_list = []
        self.num_vehs_record_df = None
        return

    def initEffectiveStorageLengths(self):
        # skip if the road is not input road
        if not self.has('output_intersection'):
            return
        
        self.effective_storage_lengths = {'left': 0.0, 'straight': 0.0, 'right': 0.0}

        # 左折・直進・右折の割合を取得
        turn_ratios = {1: 0.0, 2: 0.0, 3: 0.0}
        for vehicle_route in self.vehicle_routing_decision.vehicle_routes.getAll():
            direction_id = vehicle_route.get('direction_id')
            turn_ratios[int(direction_id)] += vehicle_route.get('turn_ratio')

        # 道路タイプで分岐
        if self.type == 1:
            # 車線数：分岐前1車線，分岐後2車線
            # 進路：左車線は左折と直進，右車線は右折
            
            #　左車線について
            before_branch_length = self.right_connector.get('from_pos')
            after_branch_length = self.main_link.get('length') - before_branch_length
            self.effective_storage_lengths['left'] += (turn_ratios[1] / (turn_ratios[1] + turn_ratios[2] + turn_ratios[3])) * before_branch_length + (turn_ratios[1] / (turn_ratios[1] + turn_ratios[2])) * after_branch_length
            self.effective_storage_lengths['straight'] += (turn_ratios[2] / (turn_ratios[1] + turn_ratios[2] + turn_ratios[3])) * before_branch_length + (turn_ratios[2] / (turn_ratios[1] + turn_ratios[2])) * after_branch_length
            self.effective_storage_lengths['right'] += (turn_ratios[3] / (turn_ratios[1] + turn_ratios[2] + turn_ratios[3])) * before_branch_length

            # 右車線について
            branch_length = 0.0
            branch_length += self.right_link.get('length')
            branch_length += self.right_connector.get('length') 
            branch_length -= self.right_connector.get('to_pos')
            self.effective_storage_lengths['right'] += branch_length

        elif self.type == 2:
            # 車線数：分岐前2車線，分岐後3車線
            # 進路：左車線は左折と直進，真ん中の車線は直進，右車線は右折

            # 左車線について
            main_link_length = self.main_link.get('length')
            self.effective_storage_lengths['left'] += (turn_ratios[1] / (turn_ratios[1] + (turn_ratios[2] / 2))) *  main_link_length
            self.effective_storage_lengths['straight'] += ((turn_ratios[2] / 2) / (turn_ratios[1] + (turn_ratios[2] / 2))) *  main_link_length

            # 真ん中の車線について
            before_branch_length = self.right_connector.get('from_pos')
            after_branch_length = self.main_link.get('length') - before_branch_length
            self.effective_storage_lengths['straight'] += ((turn_ratios[2] / 2) / (turn_ratios[3] + (turn_ratios[2] / 2))) * before_branch_length + after_branch_length
            self.effective_storage_lengths['right'] += (turn_ratios[3] / (turn_ratios[3] + (turn_ratios[2] / 2))) * before_branch_length

            # 右車線について
            branch_length = 0.0
            branch_length += self.right_link.get('length')
            branch_length += self.right_connector.get('length') 
            branch_length -= self.right_connector.get('to_pos')
            self.effective_storage_lengths['right'] += branch_length

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
            self.vehicles_df = DataFrame(columns=['id', 'position', 'in_queue', 'speed', 'lane_id', 'link_id', 'road_id', 'direction_id'])

        # if the road is not input road, skip the rest of the process    
        if not self.has('output_intersection'):
            return
        
        # calculate route_num_vehs_map
        self.route_num_vehs_map = {}
        for direction_id in range(self.output_intersection.get('num_roads')):
            self.route_num_vehs_map[direction_id] = 0        

        for _, tmp_vehicles_df in self.vehicles_df.iterrows():
            direction_id = tmp_vehicles_df['direction_id']
            self.route_num_vehs_map[int(direction_id)] += 1

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
    def direction_signal_value_map(self):
        direction_signal_value_map = {}
        for direction_id, signal_group_id in self.direction_signal_group_map.items():
            signal_group = self.signal_groups[signal_group_id]
            direction_signal_value_map[direction_id] = signal_group.get('current_value')
        
        return direction_signal_value_map
    
    @property
    def vehicle_routing_decision(self):
        return self.main_link.vehicle_routing_decision


    
