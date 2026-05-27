from libs.container import Container
from libs.object import Object
from objects.data_collections import DataCollectionPoints

import pandas as pd
import re

from objects.travel_time_measurements import TravelTimeMeasurements

class Links(Container):
    def __init__(self, upper_object):
        super().__init__()
        
        # set objects
        self.config = upper_object.config
        self.executor = upper_object.executor
        
        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.com = self.network.com.Links

            self._initElements()
            self._connectObjects()

        elif upper_object.__class__.__name__ == 'Road':
            self.road = upper_object
            self.network = self.road.network
        
        elif upper_object.__class__.__name__ == 'Intersection':
            self.intersection = upper_object
            self.network = self.intersection.network

        elif upper_object.__class__.__name__ == 'Link':
            self.link = upper_object
            self.network = self.link.network

        elif upper_object.__class__.__name__ == 'Lane':
            self.lane = upper_object
            self.network = self.lane.network
        
        elif upper_object.__class__.__name__ == 'DelayMeasurement':
            self.delay_measurement = upper_object
            self.network = self.delay_measurement.network
        
        else:
            raise NotImplementedError(f"Not supported upper object: {upper_object.__class__.__name__}")
        
        return

    def _initElements(self): 
        # get link_info_map
        link_info_map = {}
        link_input_tags_df = self.config.get('link_input_tags_df_map')[self.network.simulation.get('inflow_name')]
        for _, link_input_tag_row in link_input_tags_df.iterrows():
            link_info_map[int(link_input_tag_row['link_id'])] = {
                'input_volume': int(link_input_tag_row['input_volume'])
            }

        road_link_tags_df = self.config.get('road_link_tags_df')
        for _, road_link_tag_row in road_link_tags_df.iterrows():
            if int(road_link_tag_row['link_id']) not in link_info_map:
                link_info_map[int(road_link_tag_row['link_id'])] = {}
            link_info_map[int(road_link_tag_row['link_id'])]['type'] = road_link_tag_row['type']
        
        # make road elements
        for link_com in self.com.GetAll():
            self.add(Link(
                com=link_com, 
                links=self,
                link_info=link_info_map[int(link_com.AttValue('No'))] if int(link_com.AttValue('No')) in link_info_map else {}
            ))
        return
    
    def _connectObjects(self):
        # set link
        for link in self.getAll():
            if link.get('type') != 'connector':
                continue
            
            # set from_link and to_link
            link.from_link = self[int(link.com.AttValue('FromLink'))]
            link.from_link.to_links.add(link)
            link.to_link = self[int(link.com.AttValue('ToLink'))]
            link.to_link.from_links.add(link)

            # set to_lane and from_lane
            link.to_lane = link.to_link.lanes[link.com.ToLane.AttValue('Index')]
            link.from_lane = link.from_link.lanes[link.com.FromLane.AttValue('Index')]
            if link.to_link.get('type') in ['main', 'link']:
                link.to_lane.from_connectors.add(link)
                link.from_lane.to_connectors.add(link)
            elif link.to_link.get('type') in ['left', 'right']:
                link.to_lane.from_connector = link
                link.from_lane.to_connectors.add(link)
            else:
                raise NotImplementedError(f"Not supported link type for lane connection: {link.to_link.get('type')}")

        # set road
        road_link_tags_df = self.config.get('road_link_tags_df')
        for _, road_link_tag_row in road_link_tags_df.iterrows():
            link = self[int(road_link_tag_row['link_id'])]
            road = self.network.roads[int(road_link_tag_row['road_id'])]

            link.road = road
            road.links.add(link)
            road.set('inflow_volume', link.get('input_volume'))

            if link.get('type') == 'main':
                road.main_link = link
                road.set('length', link.get('length'))
            elif link.get('type') == 'right':
                road.right_link = link
            elif link.get('type') == 'left':
                road.left_link = link
            else:
                raise NotImplementedError(f"Not supported link type: {link.get('type')}")
                
        for link in self.getAll():
            if link.get('type') != 'connector':
                continue

            if link.to_link.get('type') in ['main', 'link']:
                link.intersection = link.to_link.road.input_intersection
                link.intersection.connectors.add(link)

            elif link.to_link.get('type') == 'right':
                link.road = link.to_link.road
                link.road.right_connector = link
                link.road.links.add(link)

            elif link.to_link.get('type') == 'left':
                link.road = link.to_link.road
                link.road.left_connector = link
                link.road.links.add(link)

            else:
                raise NotImplementedError(f"Not supported link type for road connection: {link.to_link.get('type')}")
        return
    
    def update(self):
        for link in self.getAll():
            link.update()
    
        self.executor.wait()

        for link in self.getAll():
            link.lanes.update()
        
        return
        
class Link(Object):
    def __init__(self, com, links, link_info):
        super().__init__()

        self.config = links.config
        self.executor = links.executor
        self.links = links
        self.network = self.links.network

        self.com = com

        self._initProps(link_info)

        self._connectObjects()
        return
    
    @property
    def length(self):
        return self.length_info['length']
    
    @property
    def to_pos(self):
        if self.get('type') != 'connector':
            raise NotImplementedError(f"to_pos is only implemented for connector link. current link type: {self.get('type')}")
        
        return self.length_info['to_pos']
    
    @property
    def from_pos(self):
        if self.get('type') != 'connector':
            raise NotImplementedError(f"from_pos is only implemented for connector link. current link type: {self.get('type')}")
        
        return self.length_info['from_pos'] 

    def _initProps(self, link_info):
        # set id and input_volume
        self.id = int(self.com.AttValue('No'))
        self.input_volume = link_info['input_volume'] if 'input_volume' in link_info else None

        # set type (main, left, right, link, or connector)
        if 'type' in link_info:
            self.type = link_info['type']
        elif self.com.AttValue('ToLink') is None:
            self.type = 'link'
        else:
            self.type = 'connector'

        # set length_info
        self.length_info = {}
        self.length_info['length'] = float(self.com.AttValue('Length2D'))
        if self.type == 'connector':
            self.length_info['to_pos'] = float(self.com.AttValue('ToPos'))
            self.length_info['from_pos'] = float(self.com.AttValue('FromPos'))

        return
    
    def _connectObjects(self):
        # set road (Link._connectObjects), intersection (Links._connectObjects), and travel_time_measurements (TravelTimeMeasurement._connectObjects)
        if self.type == 'connector':
            self.road = None
            self.intersection = None
            self.travel_time_measurements = None
            
        elif self.type == 'main':
            self.road = None
            self.travel_time_measurements = TravelTimeMeasurements(self)
            self.vehicle_routing_decision = None

        elif self.type in ['main', 'link', 'left', 'right']:
            self.road = None

        else:
            raise NotImplementedError(f"Not supported link type: {self.type}")

        # set lanes
        self.lanes = Lanes(self)

        # set link (Links._connectObjects())
        if self.type == 'connector':
            self.from_link = None 
            self.to_link = None
            self.to_lane = None
            self.from_lane = None
        elif self.type in ['main', 'link', 'left', 'right']:
            self.from_links = Links(self)
            self.to_links = Links(self)
            self.from_lanes = Lanes(self)
            self.to_lanes = Lanes(self)
        else:
            raise NotImplementedError(f"Not supported link type: {self.type}")
        
        if self.type == 'connector':
            # set vehicle_route (VehicleRoute._connectObjects())
            self.vehicle_route = None

            # set signal_head (SignalHead._connectObjects())
            self.signal_head = None
        
        # set data_collection_points (DataCollectionPoint._connectObjects())
        self.data_collection_points = DataCollectionPoints(self)
        return
    
    def update(self):
        self._getVehicleDataFromVissim()
        self.executor.submit(self._makeFormattedVehicleData)
        return
    
    def _getVehicleDataFromVissim(self):
        self.vehicles_df = {}

        # get vehicle id list
        self.vehicles_df['id'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('No')]

        # if there is no vehicle, stop early and return
        if len(self.vehicles_df['id']) == 0:
            return
        
        # get position, in_queue, speed, lane_id, vehicle_route, and next_link_id list
        self.vehicles_df['position'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('Pos')]
        self.vehicles_df['length'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('Length')]
        self.vehicles_df['in_queue'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('InQueue')]
        self.vehicles_df['speed'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('Speed')]
        self.vehicles_df['lane_id'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('Lane')]
        self.vehicles_df['vehicle_route'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('VehRoutSta')]
        self.vehicles_df['next_link_id'] = [int(tmp_data[1]) if tmp_data[1] != None else None for tmp_data in self.com.Vehs.GetMultiAttValues('NextLink')]
        return
    
    def _makeFormattedVehicleData(self):
        # if there is no vehicle, return empty DataFrame
        if len(self.vehicles_df['id']) == 0:
            self.vehicles_df = pd.DataFrame(columns=[
                'id', 'position', 'length', 'in_queue', 'speed', 'lane_id', 'link_id', 
                'next_link_id', 'road_id', 'route_id',
            ])
            return
        
        # get link_id list
        self.vehicles_df['link_id'] = [self.id] * len(self.vehicles_df['id'])

        # get road_id list
        road_id = None if not self.has('road') else self.road.get('id')
        self.vehicles_df['road_id'] = [road_id] * len(self.vehicles_df['id'])

        # reshape lane_id to only lane number
        self.vehicles_df['lane_id'] = [int(re.match(rf"{self.id}-(\d+)", lane_str).group(1)) for lane_str in self.vehicles_df['lane_id']]
        
        # get route_id_list
        vehicle_routing_decisions = self.links.network.vehicle_routing_decisions
        route_id_list = []
        for vehicle_route_str in self.vehicles_df['vehicle_route']:
            if vehicle_route_str is None:
                route_id_list.append(0)
                continue
            match_obj = re.match(rf"(\d+)-(\d+)", vehicle_route_str)
            vehicle_route = vehicle_routing_decisions[int(match_obj.group(1))].vehicle_routes[int(match_obj.group(2))]
            route_id_list.append(vehicle_route.get('route_id'))
        
        self.vehicles_df['route_id'] = route_id_list

        # remove vehicle_route column
        self.vehicles_df.pop('vehicle_route')
        
        # change to dataframe
        self.vehicles_df = pd.DataFrame(self.vehicles_df)
        self.vehicles_df = self.vehicles_df.sort_values(by='position', ascending=False)
        self.vehicles_df = self.vehicles_df.reset_index(drop=True)
    
        return

    @property
    def queue_length(self):
        return self.queue_counter.get('current_queue_length')

class Lanes(Container):
    def __init__(self, upper_object):
        super().__init__()

        # set objects
        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Link':
            # set link
            self.link = upper_object
            self.network = self.link.network

            # set com object
            self.com = self.link.com.Lanes

            # initialize lane objects
            self._initElements()
        
        elif upper_object.__class__.__name__ == 'MpcController':
            # set mpc_controller
            self.mpc_controller = upper_object
        
        elif upper_object.__class__.__name__ == 'BcAgent':
            # set bc_agent
            self.bc_agent = upper_object
        
        else:
            raise NotImplementedError(f"Not supported upper object: {upper_object.__class__.__name__}")

        return   
    
    def _initElements(self):
        for lane_com in self.com.GetAll():
            self.add(Lane(lane_com, self))
    
    def update(self):
        for lane in self.getAll():
            self.executor.submit(lane.update)
        return

class Lane(Object):
    def __init__(self, com, lanes):
        super().__init__()

        # set objects
        self.config = lanes.config
        self.executor = lanes.executor
        self.lanes = lanes
        self.link = lanes.link
        self.network = lanes.network

        # set com objects
        self.com = com

        self._initProps()
        self._connectObjects()
        return
    
    @property
    def length(self):
        return self.link.get('length')

    @property
    def num_vehicles(self):
        return self.vehicles_df.shape[0]

    @property
    def num_vehs_in_queue(self):
        return self.vehicles_df[self.vehicles_df['in_queue']].shape[0]
    
    def _initProps(self):
        self.id = int(self.com.AttValue('Index'))
        return
    
    def _connectObjects(self):
        if self.link.get('type') in ['main', 'link']:
            # set connector (Links._connectObjects())
            self.from_connectors = Links(self)
            self.to_connectors = Links(self)

            # set data_collection_point (DataCollectionPoint._connectObjects())
            self.data_collection_point = None

        elif self.link.get('type') in ['left', 'right']:
            # set connector (Links._connectObjects())
            self.from_connector = None
            self.to_connectors = Links(self)

        elif self.link.get('type') == 'connector':
            # set data_collection_point (DataCollectionPoint._connectObjects())
            self.data_collection_point = None

            # set signal_head (SignalHead._connectObjects())
            self.signal_head = None

            # set vehicle_route (VehicleRoute._connectObjects())
            self.vehicle_route = None

        else:
            raise NotImplementedError(f"Not supported link type for lane connection: {self.link.get('type')}")
            
        return

    def update(self):
        # get vehicles_df from link
        vehicles_df = self.link.get('vehicles_df')

        if vehicles_df.empty:
            self.vehicles_df = pd.DataFrame(columns=['id', 'position', 'length', 'in_queue', 'speed', 'lane_id', 'link_id', 'road_id', 'route_id'])
            return

        # set vehicles_df for each lane
        self.vehicles_df = vehicles_df[vehicles_df['lane_id'] == self.id].copy()
        return

    
    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        
        if self.get('id') != other.get('id'):
            return False
        
        if self.link != other.link:
            return False
        
        return True
        



        
        

        