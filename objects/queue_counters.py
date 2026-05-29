from libs.container import Container
from libs.object import Object
import pandas as pd

class QueueCounters(Container):
    def __init__(self, upper_object):
        super().__init__()

        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.com = self.network.com.QueueCounters
            self._initElements()

        elif upper_object.__class__.__name__ == 'Road':
            self.road = upper_object
        else:
            raise NotImplementedError(f"Not supported upper_object class for QueueCounters: {upper_object.__class__.__name__}")
        
        return

    def _initElements(self):
        for queue_counter_com in self.com.GetAll():
            self.add(QueueCounter(queue_counter_com, self))
        return
    
    def update(self):
        # get data from com object
        queue_counter_ids = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('No')]
        queue_lengths = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('QLen(Current, Last)')]

        # update queue_length_record for each queue counter
        for index, queue_counter_id in enumerate(queue_counter_ids):
            self.executor.submit(self[queue_counter_id].update, queue_lengths[index])
        return
    
    def sync(self, type):
        for queue_counter in self.getAll():
            self.executor.submit(queue_counter.sync, type)
        self.executor.wait()
        return
    
    @property
    def max_queue_length(self):
        queue_length_list = [queue_counter.get('current_queue_length') for queue_counter in self.getAll()]
        return max(queue_length_list)

class QueueCounter(Object):
    def __init__(self, com, queue_counters):
        super().__init__()

        # set objects
        self.config = queue_counters.config
        self.executor = queue_counters.executor
        self.queue_counters = queue_counters
        self.network = queue_counters.network

        # set com object
        self.com = com

        # set properties
        self._initProps()
        self._connectObjects()
        return

    def _initProps(self):
        # set id
        self.id = self.com.AttValue('No')

        # initialize queue_length_list
        self.record_list = []
        self.record_df = None
        return 
    
    def _connectObjects(self):
        # set link
        self.link = self.network.links[self.com.Link.AttValue('No')]
        self.link.queue_counter = self

        # set road
        self.road = self.link.road
        self.road.queue_counters.add(self)
        return

    def update(self, value):
        self.record_list.append({
            'time': int(self.network.get('current_time')),
            'value': 0.0 if value is None else value,
        })
        return
    
    def sync(self, type):
        if type == 'dataframe':
            self.record_df = pd.DataFrame(self.record_list)
        else:
            raise NotImplementedError(f"Not supported type: {type}")
        
    @property
    def current_time(self):
        if len(self.record_list) == 0:
            return None
        else:
            return self.record_list[-1]['time']

    @property
    def current_queue_length(self):
        if len(self.record_list) == 0:
            return 0.0
        else: 
            return self.record_list[-1]['value']

    @property
    def delta_queue_length(self):
        if len(self.record_list) > 1:
            return self.record_list[-1]['value'] - self.record_list[-2]['value']

        if len(self.record_list) == 1:
            return self.record_list[-1]['value']

        if len(self.record_list) == 0:
            return 0.0

        
        
