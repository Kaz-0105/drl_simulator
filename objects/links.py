from libs.container import Container
from libs.object import Object

class Links(Container):
    def __init__(self, upper_object, options = None):
        # 継承
        super().__init__()
        
        # 設定オブジェクトを取得
        self.config = upper_object.config

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

class Link(Object):
    def __init__(self, com, links):
        # 継承
        super().__init__()
        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = links.config
        self.links = links

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # リンクの種類を設定（リンク, コネクタ）
        if self.com.AttValue('ToLink') is None:
            self.type = 'link'
        else:
            self.type = 'connector'
        
        # 紐づくリンクを格納するコンテナを初期化
        self.from_links = Links(self, {'type': 'from'})
        self.to_links = Links(self, {'type': 'to'})

        # 下位の紐づくオブジェクトを初期化
        self.lanes = Lanes(self)
    

class Lanes(Container):
    def __init__(self, link):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = link.config
        self.link = link

        # comオブジェクトを取得
        self.com = self.link.com.Lanes

        # 下位の紐づくオブジェクトを初期化
        self.makeElements()
    
    def makeElements(self):
        for lane_com in self.com.GetAll():
            self.add(Lane(lane_com, self))

class Lane(Object):
    def __init__(self, com, lanes):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = lanes.config
        self.lanes = lanes

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = int(self.com.AttValue('Index'))

        