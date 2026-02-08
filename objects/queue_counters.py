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
            self.makeElements()
        elif upper_object.__class__.__name__ == 'Road':
            self.road = upper_object
        else:
            raise NotImplementedError(f"Not supported upper_object class for QueueCounters: {upper_object.__class__.__name__}")
        
        return

    def makeElements(self):
        for queue_counter_com in self.com.GetAll():
            self.add(QueueCounter(queue_counter_com, self))
        return
    
    def update(self):
        # get data from com object
        queue_counter_ids = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('No')]
        queue_lengths = [tmp_data[1] for tmp_data in self.com.GetMultiAttValues('QLen(Current, Last)')]

        # update queue_length_record for each queue counter
        for index, queue_counter_id in enumerate(queue_counter_ids):
            queue_counter = self[queue_counter_id]
            self.executor.submit(queue_counter.update, queue_lengths[index])
        return
    
    @property
    def max_queue_length(self):
        if self.count() == 0:
            raise ValueError("No QueueCounter object in the QueueCounters object.")
        queue_length_list = [queue_counter.get('current_queue_length') for queue_counter in self.getAll()]
        return max(queue_length_list)

class QueueCounter(Object):
    def __init__(self, com, queue_counters):
        super().__init__()

        self.config = queue_counters.config
        self.executor = queue_counters.executor
        self.queue_counters = queue_counters
        self.network = queue_counters.network

        # set com object
        self.com = com

        # set properties
        self._initProps()
        return

    def _initProps(self):
        self.id = self.com.AttValue('No')
        self.queue_length_record = pd.DataFrame(columns=['time', 'queue_length'])
        self.link = self.network.links[self.com.Link.AttValue('No')]
        self.link.set('queue_counter', self)
        self.road = self.link.road
        self.road.queue_counters.add(self)
        return

    def update(self, queue_length): 
        queue_length = 0.0 if queue_length is None else round(queue_length, 1)
        self.queue_length_record.loc[len(self.queue_length_record)] = [self.current_time, queue_length]
        return

    @property
    def current_time(self):
        return self.network.simulation.get('current_time')

    @property
    def current_queue_length(self):
        if len(self.queue_length_record) == 0:
            return 0.0
        return self.queue_length_record.iloc[-1]['queue_length']

    @property
    def delta_queue_length(self):
        if len(self.queue_length_record) < 2:
            return self.current_queue_length
        return self.queue_length_record.iloc[-1]['queue_length'] - self.queue_length_record.iloc[-2]['queue_length']

        
        
