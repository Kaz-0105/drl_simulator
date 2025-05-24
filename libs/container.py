from libs.common import Common

class Container(Common):
    def __init__(self):
        super().__init__()
        self.elements = {}

    def getMultiAttValues(self, property_name: str):
        return [element.get(property_name) for element in self.elements.values()]

    def getKeys(self, container_flg = False, sorted_flg = False):
        if container_flg:
            keys = list(self.elements.keys())
        else:
            keys = self.getMultiAttValues('id')
            
        if sorted_flg:
            keys.sort()

        return keys
    
    def count(self):
        return len(self.elements)
    
    def __getitem__(self, key):
        try:
            return self.elements[key]
        except KeyError:
            print('Key ' + str(key) + ' is not found in the container.')
    
    def getAll(self, sorted_flg = False):
        if sorted_flg == False:
            return list(self.elements.values())
    
        elements = []
        for key in self.getKeys(container_flg=True, sorted_flg=True):
            elements.append(self[key])
        
        return elements
    
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