from libs.container import Container
from controllers.drl_controller import DRLController

class Controllers(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトと上位の紐づくオブジェクトを取得
        self.config = network.config
        self.executor = network.executor
        self.network = network

        self.makeElements()
    
    def makeElements(self):
        simulator_info = self.config.get('simulator_info')
        if simulator_info['control_method'] == 'drl':
            intersections = self.network.intersections

            for intersection in intersections.getAll():
                self.add(DRLController(self, intersection))
    
    def run(self):
        for controller in self.getAll():
            self.executor.submit(controller.run)
        
        self.executor.wait()


        