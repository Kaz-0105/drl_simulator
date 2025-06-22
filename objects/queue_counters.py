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
        
        elif upper_object.__class__.__name__ == 'Road':
            # 上位の紐づくオブジェクトを取得
            self.road = upper_object
        return

    def makeElements(self):
        for queue_counter_com in self.com.GetAll():
            self.add(QueueCounter(queue_counter_com, self))
        return
    
    def updateData(self):
        # Comオブジェクトからデータを取得
        queue_counter_ids = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('No')]
        queue_lengths = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('QLen(Current, Last)')]

        # データを要素オブジェクトにセット（非同期処理）
        for index, queue_counter_id in enumerate(queue_counter_ids):
            queue_counter = self[queue_counter_id]
            self.executor.submit(queue_counter.updateData, queue_lengths[index])
        return
    
    @property
    def max_queue_length(self):
        max_queue_length = 0
        for queue_counter in self.getAll():
            if queue_counter.get('current_queue_length') > max_queue_length:
                max_queue_length = queue_counter.get('current_queue_length') 
        return max_queue_length

class QueueCounter(Object):
    def __init__(self, com, queue_counters):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期実行オブジェクトを取得
        self.config = queue_counters.config
        self.executor = queue_counters.executor

        # queue_countersオブジェクトを取得
        self.queue_counters = queue_counters

        # 対応するComオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = self.com.AttValue('No')

        # networkオブジェクトと紐づける
        self.network = queue_counters.network

        # linkとroadオブジェクトを紐づける
        self._makeLinkConnection()
        self._makeRoadConnection()

        # current_queue_lengthを初期化
        self.current_queue_length = 0

        # queue_length_record（時系列データ）を初期化
        self.queue_length_record = pd.DataFrame(columns=['time', 'queue_length'])
        return
    
    def _makeLinkConnection(self):
        link_com = self.com.Link
        self.link = self.network.links[link_com.AttValue('No')]
        self.link.set('queue_counter', self)
        return
    
    def _makeRoadConnection(self):
        self.road = self.link.road
        self.road.queue_counters.add(self)
        return

    def updateData(self, queue_length):    
        # current_queue_lengthを更新
        self.current_queue_length = 0.0 if queue_length is None else round(queue_length, 1)

        # queue_length_recordを更新
        self.queue_length_record.loc[len(self.queue_length_record)] = [self.current_time, self.current_queue_length]
        return

    @property
    def current_time(self):
        return self.network.simulation.get('current_time')

    @property
    def delta_queue_length(self):
        if len(self.queue_length_record) < 2:
            return self.current_queue_length
        return self.queue_length_record.iloc[-1]['queue_length'] - self.queue_length_record.iloc[-2]['queue_length']

        
        
