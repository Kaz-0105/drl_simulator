from libs.common import Common

class Object(Common):
    def __init__(self):
        # 継承
        super().__init__()

    def __eq__(self, other):
        self_class_name = self.__class__.__name__
        other_class_name = other.__class__.__name__
        
        if self_class_name == other_class_name:
            return self.get('id') == other.get('id')
        else:
            return False
    