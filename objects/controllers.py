from libs.container import Container
from controllers.drl_controller import DRLController

class Controllers(Container):
    def __init__(self, network):
        super().__init__()

        self.config = network.config
        self.executor = network.executor
        self.network = network

        self.makeElements()
    
    def makeElements(self):
        if self.config.get('control_method') == 'drl':
            intersections = self.network.intersections

            for intersection in intersections.getAll():
                self.add(DRLController(self, intersection))
    
    def run(self):
        for controller in self.getAll():
            controller.run()
            # self.executor.submit(controller.run)
        
        self.executor.wait()


        