from libs.container import Container
from libs.object import Object
from objects.data_collections import DataCollectionPoints

import pandas as pd
import re

class Links(Container):
    def __init__(self, upper_object, options = None):
        super().__init__()
        
        # set objects
        self.config = upper_object.config
        self.executor = upper_object.executor
        
        if upper_object.__class__.__name__ == 'Network':
            # set network
            self.network = upper_object

            # set com object
            self.com = self.network.com.Links

            # initialize link objects
            self._initElements()

            # set inputs
            self._initInputs()

            # connect links based on from_link and to_link information in vissim
            self._makeLinkConnections()

            # connect links and roads based on road_link_tags in config file
            self._makeRoadConnections()

        elif upper_object.__class__.__name__ == 'Link':
            # set link
            self.link = upper_object

            # set type
            self.type = options['type']

        elif upper_object.__class__.__name__ == 'Road':
            # set road
            self.road = upper_object
        
        elif upper_object.__class__.__name__ == 'TravelTimeMeasurement':
            # set travel_time_measurement
            self.travel_time_measurement = upper_object
        
        elif upper_object.__class__.__name__ == 'DelayMeasurement':
            # set delay_measurement
            self.delay_measurement = upper_object
        
        else:
            raise NotImplementedError(f"Not supported upper object: {upper_object.__class__.__name__}")
        return

    def _initElements(self):
        for link_com in self.com.GetAll():
            self.add(Link(link_com, self))
    
    def _initInputs(self):
        # get inflows_df
        inflow_name = self.network.simulation.get('inflow_name')
        link_input_tags_df_map = self.config.get('link_input_tags_df_map')
        inflows_df = link_input_tags_df_map[inflow_name]

        # set input_volume to each link object
        for _, inflow_row in inflows_df.iterrows():
            link = self[int(inflow_row['link_id'])]
            link.set('input_volume', int(inflow_row['input_volume']))

        return
    
    def _makeLinkConnections(self):
        for connector in self.getAll():
            if connector.get('type') != 'connector':
                continue

            from_link = self[int(connector.com.AttValue('FromLink'))]
            to_link = self[int(connector.com.AttValue('ToLink'))]
            
            # set from_link, to_link and to_lane for connector
            connector.from_links.add(from_link)
            connector.to_links.add(to_link)
            connector.set('from_link', from_link)
            connector.set('to_link', to_link)
            connector.set('to_lane', to_link.lanes[connector.get('to_lane_id')])
            
            # set connector for from_link and to_link
            from_link.to_links.add(connector)
            to_link.from_links.add(connector)
        return
        
    def _makeRoadConnections(self):
        tags_df = self.config.get('road_link_tags_df')
        for _, tag_row in tags_df.iterrows():
            road = self.network.roads[tag_row['road_id']]
            link = self[tag_row['link_id']]
            
            # set type and road for link
            link.set('type', tag_row['type'])
            link.set('road', road)

            # set link to road
            road.links.add(link)
            if tag_row['type'] == 'main':
                road.set('main_link', link)
                road.set('length', link.get('length'))
            elif tag_row['type'] == 'right':
                road.set('right_link', link)
            elif tag_row['type'] == 'left':
                road.set('left_link', link)
            else:
                raise NotImplementedError(f"Not supported link type: {tag_row['type']}")

            # set input_volume to road if link has input_volume
            if link.has('input_volume'):
                road.set('input_volume', link.get('input_volume'))

        for link in self.getAll():
            if link.get('type') != 'connector':
                continue

            from_link = link.from_links.getAll()[0]
            to_link = link.to_links.getAll()[0]

            if from_link.road == to_link.road:
                road = from_link.road

                # connect link and road
                link.set('road', road)
                road.links.add(link)
                
                if to_link == road.right_link:
                    road.set('right_connector', link)
                elif to_link == road.left_link:
                    road.set('left_connector', link)
                else:
                    raise ValueError(f"Something wrong in layout design: road_id = {road.get('id')}, connector_id = {link.get('id')}")

        return
    
    def update(self):
        for link in self.getAll():
            link.update()
    
        self.executor.wait()

        for link in self.getAll():
            link.lanes.update()
        
        return
        
class Link(Object):
    def __init__(self, com, links):
        super().__init__()

        # set objects
        self.config = links.config
        self.executor = links.executor
        self.links = links
        self.network = self.links.network

        # set com objects
        self.com = com

        # set properties
        self._initProps()

        # init lane objects
        self.lanes = Lanes(self)

        # initialize from_links, to_links, and data_collection_points
        self.from_links = Links(self, options={'type': 'from'})
        self.to_links = Links(self, options={'type': 'to'})
        self.data_collection_points = DataCollectionPoints(self)
        return

    def _initProps(self):
        self.id = int(self.com.AttValue('No'))
        self.type = 'link' if self.com.AttValue('ToLink') is None else 'connector'

        self.length_info = {}
        self.length_info['length'] = float(self.com.AttValue('Length2D'))
        self.length = self.length_info['length']

        if self.type == 'link':
            return

        self.to_lane_id = self.com.ToLane.AttValue('Index')
        self.length_info['to_pos'] = float(self.com.AttValue('ToPos'))
        self.length_info['from_pos'] = float(self.com.AttValue('FromPos'))
        self.to_pos = self.length_info['to_pos']
        self.from_pos = self.length_info['from_pos']
        
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
        return

    @property
    def num_vehicles(self):
        return self.vehicles_df.shape[0]

    @property
    def num_vehs_in_queue(self):
        return self.vehicles_df[self.vehicles_df['in_queue']].shape[0]
    
    def _initProps(self):
        # set id, length_info
        self.id = int(self.com.AttValue('Index'))
        self.length_info = self.link.get('length_info')

        self.length = self.length_info['length']
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
        



        
        

        