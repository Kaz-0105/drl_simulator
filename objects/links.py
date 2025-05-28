from libs.container import Container
from libs.object import Object

from pandas import DataFrame

class Links(Container):
    def __init__(self, upper_object, options = None):
        # 継承
        super().__init__()
        
        # 設定オブジェクトと非同期オブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor
        
        # 上位のオブジェクトによって分岐
        if upper_object.__class__.__name__ == 'Network':
            # 上位の紐づくオブジェクトを取得
            self.network = upper_object

            # comオブジェクトを取得
            self.com = self.network.com.Links

            # 下位の紐づくオブジェクトを初期化
            self.makeElements()

            # 設定ファイルから流入量に関する情報を取得
            self.setInputs()

            # link同士を紐づける
            self.makeLinkConnections()

            # roadオブジェクトと紐づける
            self.makeRoadConnections()

        elif upper_object.__class__.__name__ == 'Link':
            # 上位の紐づくオブジェクトを取得
            self.link = upper_object

            # タイプを取得（from or to）
            self.type = options['type']

        elif upper_object.__class__.__name__ == 'Road':
            # 上位の紐づくオブジェクトを取得
            self.road = upper_object
        
        elif upper_object.__class__.__name__ == 'TravelTimeMeasurement':
            # 上位の紐づくオブジェクトを取得
            self.travel_time_measurement = upper_object
        
        elif upper_object.__class__.__name__ == 'DelayMeasurement':
            # 上位の紐づくオブジェクトを取得
            self.delay_measurement = upper_object

    def makeElements(self):
        for link_com in self.com.GetAll():
            self.add(Link(link_com, self))
    
    def setInputs(self):
        tags = self.config.get('link_input_tags')
        for _, tag in tags.iterrows():
            link = self[int(tag['link_id'])]
            link.set('input_volume', int(tag['input_volume']))
    
    def makeLinkConnections(self):
        for link in self.getAll():
            if link.type == 'link':
                continue

            from_link = self[int(link.com.AttValue('FromLink'))]
            to_link = self[int(link.com.AttValue('ToLink'))]

            link.from_links.add(from_link)
            link.to_links.add(to_link)

            from_link.to_links.add(link)
            to_link.from_links.add(link)
        
    def makeRoadConnections(self):
        tags = self.config.get('road_link_tags')
        network = self.network
        roads = network.roads

        for _, tag in tags.iterrows():
            road = roads[tag['road_id']]
            link = self[tag['link_id']]
            
            road.addLink(link, tag['type'])

            link.set('type', tag['type'])
            link.set('road', road)

        for link in self.findAll({'type': 'connector'}):
            from_link = link.from_links.getAll()[0]
            to_link = link.to_links.getAll()[0]

            if from_link.road == to_link.road:
                road = from_link.road
                
                road.addLink(link, 'connector')
                link.set('type', 'connector')
                link.set('road', from_link.road)
    
    def updateData(self):
        # 要素オブジェクトの更新
        for link in self.getAll():
            link.updateData()
        
        # 終わるまで待機
        self.executor.wait()

        # 車線ごとに車両データを分割
        for link in self.getAll():
            link.lanes.updateData()
        
class Link(Object):
    def __init__(self, com, links):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = links.config
        self.executor = links.executor
        self.links = links

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # リンクの種類を設定（リンク, コネクタ，リンクは後でRoadオブジェクトの設定ファイルからさらに分岐する）
        if self.com.AttValue('ToLink') is None:
            self.type = 'link'
        else:
            self.type = 'connector'
        
        # 紐づくリンクを格納するコンテナを初期化
        self.from_links = Links(self, {'type': 'from'})
        self.to_links = Links(self, {'type': 'to'})

        # 下位の紐づくオブジェクトを初期化
        self.lanes = Lanes(self)

        # linkの長さを取得
        self.length_info = {'length': self.com.AttValue('Length2D')}
        if self.type == 'connector':
            self.length_info['to_pos'] = self.com.AttValue('ToPos')
            self.length_info['from_pos'] = self.com.AttValue('FromPos')
    
    @property
    def queue_length(self):
        return self.queue_counter.get('current_queue_length')
    
    
    
    def updateData(self):
        # 車両データを取得
        self.getVehicleDataFromVissim()

        # 非同期処理で車両データを整形
        self.executor.submit(self.makeFormattedVehicleData)
    
    def getVehicleDataFromVissim(self):
        # Vissimから車両データを取得
        self.vehicle_data = {}
        self.vehicle_data['id'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('No')]
        self.vehicle_data['position'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('Pos')]
        self.vehicle_data['in_queue'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('InQueue')]
        self.vehicle_data['speed'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('Speed')]
        self.vehicle_data['lane_id'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('Lane')]
        self.vehicle_data['vehicle_route'] = [tmp_data[1] for tmp_data in self.com.Vehs.GetMultiAttValues('VehRoutSta')]
    
    def makeFormattedVehicleData(self):
        # 車両が存在しない場合は空のDataFrameを返す
        if len(self.vehicle_data['id']) == 0:
            column_names = ['id', 'position', 'in_queue', 'speed', 'lane_id', 'link_id', 'road_id', 'direction_id']
            self.vehicle_data = DataFrame(columns = column_names)
            return
        
        # linkのIDを追加
        self.vehicle_data['link_id'] = [self.id] * len(self.vehicle_data['id'])

        # roadのIDを追加
        if self.has('road'):
            self.vehicle_data['road_id'] = [self.road.get('id')] * len(self.vehicle_data['id'])
        else:
            self.vehicle_data['road_id'] = [None] * len(self.vehicle_data['id'])

        # positionとspeedを少数点第1位までに丸める
        self.vehicle_data['position'] = [round(position, 1) for position in self.vehicle_data['position']]
        self.vehicle_data['speed'] = [round(speed, 1) for speed in self.vehicle_data['speed']]

        # in_queueをbool型に変換
        self.vehicle_data['in_queue'] = [bool(in_queue) for in_queue in self.vehicle_data['in_queue']]

        # laneをint型に変換
        self.vehicle_data['lane_id'] = [int(lane.replace(str(self.id) + '-', '')) for lane in self.vehicle_data['lane_id']]
        
        # vehicle_routeを方向に変換
        vehicle_routing_decisions = self.links.network.vehicle_routing_decisions
        direction_ids = []
        for vehicle_route_str in self.vehicle_data['vehicle_route']:
            if vehicle_route_str is None:
                direction_ids.append(0)
            else:
                vehicle_routing_decision_id, vehicle_route_id = tuple([int(id) for id in vehicle_route_str.split('-')])
                vehicle_route = vehicle_routing_decisions[vehicle_routing_decision_id].vehicle_routes[vehicle_route_id]
                direction_ids.append(vehicle_route.get('direction_id'))
        
        self.vehicle_data['direction_id'] = direction_ids
        self.vehicle_data.pop('vehicle_route')
            
        self.vehicle_data = DataFrame(self.vehicle_data)
        return

    @property
    def length(self):
        return self.length_info['length']

class Lanes(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理用のオブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Link':
            # 上位の紐づくオブジェクトを取得
            self.link = upper_object

            # comオブジェクトを取得
            self.com = self.link.com.Lanes

            # 下位の紐づくオブジェクトを初期化
            self.makeElements()
        
        elif upper_object.__class__.__name__ == 'DRLController':
            # 上位の紐づくオブジェクトを取得
            self.drl_controller = upper_object
    
    def makeElements(self):
        for lane_com in self.com.GetAll():
            self.add(Lane(lane_com, self))
    
    def updateData(self):
        # 要素オブジェクトの更新
        for lane in self.getAll():
            # 車両データを取得（非同期処理）
            self.executor.submit(lane.updateData)

class Lane(Object):
    def __init__(self, com, lanes):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = lanes.config
        self.executor = lanes.executor
        self.lanes = lanes

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = int(self.com.AttValue('Index'))

    @property
    def length_info(self):
        return self.lanes.link.length_info
    
    @property
    def link(self):
        return self.lanes.link
    
    @property
    def num_vehicles(self):
        return self.vehicle_data.shape[0]

    def updateData(self):
        # 車両データを取得
        vehicle_data = self.lanes.link.get('vehicle_data')

        if vehicle_data.shape[0] == 0:
            # 車両データが存在しない場合は空のDataFrameを返す
            column_names = ['id', 'position', 'in_queue', 'speed', 'lane_id', 'link_id', 'road_id', 'direction_id']
            self.vehicle_data = DataFrame(columns = column_names)
            return

        # 車両データを取得
        self.vehicle_data = vehicle_data[vehicle_data['lane_id'] == self.id].copy()

    
    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        
        if self.get('id') != other.get('id'):
            return False
        
        if other.has('link') == False:
            return False
        
        if self.link != other.link:
            return False
        
        return True
        



        
        

        