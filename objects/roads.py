from libs.container import Container
from libs.object import Object
from objects.links import Links
from objects.signal_controllers import SignalGroups
from objects.queue_counters import QueueCounters
from pandas import DataFrame
import pandas as pd


class Roads(Container): 
    def __init__(self, upper_object, options = None):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

        # 上位の紐づくオブジェクトによって分岐
        if upper_object.__class__.__name__ == 'Network':
            # 上位の紐づくオブジェクトを取得
            self.network = upper_object

            # 要素オブジェクトを初期化
            self.makeElements()

        elif upper_object.__class__.__name__ == 'Intersection':
            # 上位の紐づくオブジェクトを取得
            self.intersection = upper_object

            # タイプを設定（input/output）
            self.type = options['type']

            # 要素オブジェクトを設定
            self.makeElements()
    
    def makeElements(self):
        # 上位の紐づくオブジェクトによって分岐
        if self.has('network'):
            roads = self.config.get('roads')
            for _, road in roads.iterrows():
                self.add(Road(road, self))
        elif self.has('intersection'):
            tags = self.config.get('intersection_road_tags')
            target_tags = tags[tags['intersection_id'] == self.intersection.get('id') & (tags['type'] == self.type)]

            network = self.intersection.getNetwork()
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
            
    def updateData(self):
        for road in self.getAll():
            road.updateData()
        
        self.executor.wait()        

class Road(Object):
    def __init__(self, road, roads):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = roads.config
        self.executor = roads.executor
        self.roads = roads

        # IDを取得
        self.id = int(road['id'])

        # 法定速度を設定
        self.max_speed = int(road['max_speed'])

        # 紐づくlinkオブジェクトを格納するコンテナを初期化
        self.links = Links(self)

        # リンクのタイプを格納する辞書型配列を初期化
        self.link_types = {}

        # 紐づくSignalGroupオブジェクトを格納するコンテナを初期化
        self.signal_groups = SignalGroups(self)

        # SignalGroupオブジェクトの信号方向との対応関係を示す辞書型配列を初期化
        self.direction_signal_group_map = {}

    def addLink(self, link, link_type):
        self.links.add(link)
        self.link_types[link.get('id')] = link_type
    
    def getMainLink(self):
        for link_id, link_type in self.link_types.items():
            if link_type == 'main':
                return self.links[link_id]

    def getVehicleRoutingDecision(self):
        main_link = self.getMainLink()
        if main_link.has('vehicle_routing_decision'):
            return main_link.vehicle_routing_decision
        else:
            return None
    
    @property
    def queue_counters(self):
        return QueueCounters(self)
    
    def updateData(self):
        # 紐づくlinkオブジェクトのデータを更新
        self.links.updateData()

        # linksのデータをroadにまとめる
        self.executor.submit(self.summarizeData)
    
    def summarizeData(self):
        # 車両データを初期化
        self.vehicle_data = None
        
        for link in self.links.getAll():
            # 車両データを取得
            vehicle_data = link.get('vehicle_data')

            # 車両データが空の場合はスキップ
            if vehicle_data.shape[0] == 0:
                continue
                
            # 車両データをroadにまとめる
            if self.vehicle_data is None:
                self.vehicle_data = vehicle_data
            else:
                self.vehicle_data = pd.concat([self.vehicle_data, vehicle_data], ignore_index=True)

        
    
    
