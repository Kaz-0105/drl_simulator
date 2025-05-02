from libs.object import Object

class Container(Object):
    def __init__(self):
        super().__init__()
        self.elements = {}

    def getMultiAttValues(self, property_name: str):
        return [element.get(property_name) for element in self.elements.values()]

    def getKeys(self):
        return self.getMultiAttValues('id')
    
    def count(self):
        return len(self.elements)
    
    def __getitem__(self, key):
        try:
            return self.elements[key]
        except KeyError:
            print('Key ' + str(key) + ' is not found in the container.')
    
    def getAll(self):
        return list(self.elements.values())
    
    def add(self, element, element_id = None):
        if element_id is None:
            self.elements[element.get('id')] = element
        else:
            self.elements[element_id] = element
        
    def delete(self, element_id):
        try: 
            self.elements.pop(element_id)
        except KeyError:
            print('Key ' + str(element_id) + ' is not found in the container.')
    
    def findAll(self, conditions):
        found_elements = []
        for element in self.elements.values():
            if all(element.get(key) == value for key, value in conditions.items()):
                found_elements.append(element)
        return found_elements