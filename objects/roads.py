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
            roads_df = self.config.get('roads_df')
            for _, road_df in roads_df.iterrows():
                self.add(Road(road_df, self))
        elif hasattr(self, 'intersection'):
            tags_df = self.config.get('intersection_road_tags_df')
            target_tags_df = tags_df[tags_df['intersection_id'] == self.intersection.get('id') & (tags_df['type'] == self.type)]

            network = self.intersection.getNetwork()
            roads = network.roads

            for _, tag_df in target_tags_df.iterrows():
                road = roads[tag_df['road_id']]
                self.add(road, tag_df['order_id'])

                if self.type == 'input':
                    road.set('output_intersection', self.intersection)
                elif self.type == 'output':
                    road.set('input_intersection', self.intersection)
                
            if self.count() != self.intersection.get('num_roads'):
                raise Exception(f"Intersection {self.intersection.get('id')} has {self.intersection.get('num_roads')} roads, but roads object has {self.count()} {self.type} roads.")
            

class Road(Object):
    def __init__(self, road_df, roads):
        super().__init__()
        self.config = roads.config
        self.roads = roads
        self.id = int(road_df['id'])
        self.max_speed = int(road_df['max_speed'])

        self.links = Links(self)
        self.link_types = {}

    def addLink(self, link, link_type):
        self.links.add(link)
        self.link_types[link.get('id')] = link_type
