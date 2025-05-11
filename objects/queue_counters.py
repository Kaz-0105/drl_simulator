from libs.container import Container
from libs.object import Object

class QueueCounters(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = upper_object.config

        if upper_object.__class__.__name__ == 'Network':
            # 上位の紐づくオブジェクトを取得
            self.network = upper_object

            # 対応するComオブジェクトを取得
            self.com = self.network.com.QueueCounters
            
            # 要素オブジェクトを初期化
            self.makeElements()

            # linkオブジェクトと紐づける
            self.makeLinkConnections()
        
        elif upper_object.__class__.__name__ == 'Road':
            # 上位の紐づくオブジェクトを取得
            self.road = upper_object

            # 要素オブジェクトを初期化
            self.makeElements()

    def makeElements(self):
        if self.has('network'):
            for queue_counter_com in self.com.GetAll():
                self.add(QueueCounter(queue_counter_com, self))
        
        elif self.has('road'):
            links = self.road.links
            for link in links.getAll():
                if link.has('queue_counter'):
                    self.add(link.get('queue_counter'))
    
    def makeLinkConnections(self):
        for queue_counter in self.getAll():
            link_com = queue_counter.com.Link
            link = self.network.links[link_com.AttValue('No')]

            # linkオブジェクトと紐づける
            queue_counter.set('link', link)
            link.set('queue_counter', queue_counter)

class QueueCounter(Object):
    def __init__(self, com, queue_counters):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = queue_counters.config
        self.queue_counters = queue_counters

        # 対応するComオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')