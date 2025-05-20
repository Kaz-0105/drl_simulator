from libs.common import Common

class Object(Common):
    def __init__(self):
        # 継承
        super().__init__()

    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        
        if self.get('id') != other.get('id'):
            return False
        
        return True