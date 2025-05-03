from libs.container import Container
from libs.object import Object
from objects.links import Links

class Roads(Container): 
    def __init__(self, upper_object, options = None):
        super().__init__()
        self.config = upper_object.config
        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.makeElements()
        elif upper_object.__class__.__name__ == 'Intersection':
            self.intersection = upper_object
            self.type = options['type']
            self.makeElements()
    
    def makeElements(self):
        if hasattr(self, 'network'):
            roads = self.config.get('roads')
            for _, road in roads.iterrows():
                self.add(Road(road, self))
        elif hasattr(self, 'intersection'):
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
            

class Road(Object):
    def __init__(self, road, roads):
        super().__init__()
        self.config = roads.config
        self.roads = roads
        self.id = int(road['id'])
        self.max_speed = int(road['max_speed'])

        self.links = Links(self)
        self.link_types = {}

    def addLink(self, link, link_type):
        self.links.add(link)
        self.link_types[link.get('id')] = link_type
