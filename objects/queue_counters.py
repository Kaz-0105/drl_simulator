from libs.container import Container
from libs.object import Object
import pandas as pd

class QueueCounters(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

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
    
    def updateData(self):
        # Comオブジェクトからデータを取得
        queue_counter_ids = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('No')]
        queue_lengths = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('QLen(Current, Last)')]

        # データを要素オブジェクトにセット（非同期処理）
        for index, queue_counter_id in enumerate(queue_counter_ids):
            queue_counter = self[queue_counter_id]
            self.executor.submit(queue_counter.updateData, queue_lengths[index])

class QueueCounter(Object):
    def __init__(self, com, queue_counters):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = queue_counters.config
        self.executor = queue_counters.executor
        self.queue_counters = queue_counters

        # 対応するComオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # current_queue_lengthを初期化
        self.current_queue_length = 0

        # queue_lengths（時系列データ）を初期化
        self.queue_lengths = None

    @property
    def network(self):
        return self.queue_counters.network
    
    @property
    def current_time(self):
        return self.network.simulation.get('current_time')

    def updateData(self, queue_length):
        # 最初はNoneが返ってくるので、0に置き換える
        if queue_length is None:
            queue_length = 0
            
        # current_queue_lengthを更新
        self.current_queue_length = round(queue_length, 1)

        # queue_lengthsを更新
        new_queue_length = pd.DataFrame({'time': [self.current_time], 'queue_length': [self.current_queue_length]})

        if self.queue_lengths is None:
            self.queue_lengths = new_queue_length
        else:
            self.queue_lengths = pd.concat([self.queue_lengths, new_queue_length], ignore_index = True)
    
    @property
    def delta_queue_length(self):
        if len(self.queue_lengths) < 2:
            return self.current_queue_length
        
        return self.queue_lengths.iloc[-1]['queue_length'] - self.queue_lengths.iloc[-2]['queue_length']

        
        
