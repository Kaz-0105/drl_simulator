import copy

class Common:
    def get(self, property_name, type='reference'):
        try:
            val = getattr(self, property_name)
            if type == 'reference':
                return val
            elif type == 'copy':
                return copy.deepcopy(val)
            else:
                raise ValueError(f"Not supported type: {type}")
            
        except AttributeError:
            raise AttributeError(f"Property '{property_name}' not found in {self.__class__.__name__}.")
    
    def set(self, property_name, value):
        setattr(self, property_name, value)
        return
    
    def has(self, property_name):
        return hasattr(self, property_name)
    
    def getPropertyNames(self):
        return list(self.__dict__.keys())