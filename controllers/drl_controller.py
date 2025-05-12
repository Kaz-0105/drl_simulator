from controllers.base_controller import BaseController

class DRLController(BaseController):
    def __init__(self, controllers, intersection):
        # 継承
        super().__init__(controllers)

        # intersectionオブジェクトと紐づける
        self.intersection = intersection
        intersection.set('controller', self)