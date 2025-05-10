from libs.container import Container
from libs.object import Object

class Links(Container):
    def __init__(self, upper_object, options = None):
        super().__init__()
        self.config = upper_object.config
        if upper_object.__class__.__name__ == 'Network':
            self.network = upper_object
            self.com = self.network.com.Links
            self.makeElements()
            self.setInputs()
            self.makeLinkConnections()
            self.makeRoadConnections()
        elif upper_object.__class__.__name__ == 'Link':
            self.link = upper_object
            self.type = options['type']
        elif upper_object.__class__.__name__ == 'Road':
            self.road = upper_object

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
    def __init__(self, link_com, links):
        super().__init__()
        self.config = links.config
        self.links = links
        self.com = link_com

        self.id = self.com.AttValue('No')

        if self.com.AttValue('ToLink') is None:
            self.type = 'link'
        else:
            self.type = 'connector'
        
        self.from_links = Links(self, {'type': 'from'})
        self.to_links = Links(self, {'type': 'to'})
        