class Common:
    def get(self, property_name):
        try:
            return getattr(self, property_name)
        except AttributeError:
            raise AttributeError(f"Property '{property_name}' not found in {self.__class__.__name__}.")
    
    def set(self, property_name, value):
        setattr(self, property_name, value)
    
    def has(self, property_name):
        return hasattr(self, property_name)
    
    def getPropertyNames(self):
        return list(self.__dict__.keys())