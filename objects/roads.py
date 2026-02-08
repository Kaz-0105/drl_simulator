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
            self.makeElements()

        elif upper_object.__class__.__name__ == 'Intersection':
            self.intersection = upper_object
            self.type = options['type']
            self.makeElements()
        
        else:
            raise NotImplementedError(f"Not supported upper_object: {upper_object.__class__.__name__}")
        
        return
    
    def makeElements(self):
        if self.has('network'):
            roads = self.config.get('roads')
            for _, road in roads.iterrows():
                self.add(Road(road, self))

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

        # タイプを取得
        self.type = road['type']

        # 紐づくlinkオブジェクトを格納するコンテナを初期化
        self.links = Links(self)

        # リンクのタイプを格納する辞書型配列を初期化
        self.link_types = {}

        # queue_countersオブジェクトを初期化
        self.queue_counters = QueueCounters(self)

        # delay_measurementsオブジェクトを初期化
        self.delay_measurements = DelayMeasurements(self)

        # signal_groupsオブジェクトを初期化
        self.signal_groups = SignalGroups(self)

        # SignalGroupオブジェクトの信号方向との対応関係を示す辞書型配列を初期化
        self.direction_signal_group_map = {}

        # data_collection_pointを初期化
        self.data_collection_points = DataCollectionPoints(self)

    def addLink(self, link, link_type):
        self.links.add(link)
        self.link_types[link.get('id')] = link_type
        return

    def initEffectiveStorageLengths(self):
        # 流出道路は考える必要なし
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

    def getVehicleRoutingDecision(self):
        main_link = self.get('main_link')
        if main_link.has('vehicle_routing_decision'):
            return main_link.vehicle_routing_decision
        else:
            return None
    
    def update(self):
        # 紐づくlinkオブジェクトのデータを更新
        self.links.update()

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
        
        # 位置でソートする
        if self.vehicle_data is not None:
            self.vehicle_data.sort_values(by='position', ascending=False, inplace=True)
            self.vehicle_data.reset_index(drop=True, inplace=True)

        # 1台も車両がいないときNoneになるので、DataFrameを初期化
        if self.vehicle_data is None:
            self.vehicle_data = DataFrame(columns=['id', 'position', 'in_queue', 'speed', 'lane_id', 'link_id', 'road_id', 'direction_id', 'go_flg'])

        # 流出道路のときはここで終了    
        if not self.has('output_intersection'):
            return
        
        # route_num_vehs_mapを作成
        self.route_num_vehs_map = {}
        for direction_id in range(self.output_intersection.get('num_roads')):
            self.route_num_vehs_map[direction_id] = 0        

        for _, tmp_vehicle_data in self.vehicle_data.iterrows():
            direction_id = tmp_vehicle_data['direction_id']
            self.route_num_vehs_map[int(direction_id)] += 1
        
        return
    
    @property
    def main_link(self):
        for link in self.links.getAll():
            if link.get('type') == 'main':
                return link
        return None
    
    @property
    def right_link(self):
        for link in self.links.getAll():
            if link.get('type') == 'right':
                return link        
        return None
    
    @property
    def right_connector(self):
        for link in self.links.getAll():
            if link.get('type') == 'connector' and link.to_link.get('id') == self.right_link.get('id'):
                return link
        return None
                
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
    def length(self):
        return self.main_link.get('length')
    
    @property
    def num_vehicles(self):
        return self.vehicle_data.shape[0]

    @property
    def num_going_vehicles(self):
        return self.vehicle_data[self.vehicle_data['go_flg']].shape[0]

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


    
