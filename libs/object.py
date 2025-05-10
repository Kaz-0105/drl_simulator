class Object:
    def get(self, property_name):
        try:
            return getattr(self, property_name)
        except AttributeError:
            print('指定されたプロパティは存在しません。')
            return None
    
    def set(self, property_name, value):
        setattr(self, property_name, value)
    
    def has(self, property_name):
        return hasattr(self, property_name)
    
    def getPropertyNames(self):
        return list(self.__dict__.keys())

    def __eq__(self, other):
        self_class_name = self.__class__.__name__
        other_class_name = other.__class__.__name__
        
        if self_class_name == other_class_name:
            return self.get('id') == other.get('id')
        else:
            return False
    