from libs.container import Container
from libs.object import Object

class SignalControllers(Container):
    def __init__(self, network):
        # 継承
        super().__init__()
        
        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = network.config
        self.network = network
        
        # comオブジェクトを取得
        self.com = self.network.com.SignalControllers

        # 下位の紐づくオブジェクトを初期化
        self.makeElements()

    def makeElements(self):
        for signal_controller_com in self.com.GetAll():
            self.add(SignalController(signal_controller_com, self))

class SignalController(Object):
    def __init__(self, com, signal_controllers):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = signal_controllers.config
        self.signal_controllers = signal_controllers

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = int(self.com.AttValue('No'))

        # 下位の紐づくオブジェクトを初期化
        self.signal_groups = SignalGroups(self)


class SignalGroups(Container):
    def __init__(self, signal_controller):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = signal_controller.config
        self.signal_controller = signal_controller

        # comオブジェクトを取得
        self.com = self.signal_controller.com.SGs

        # 下位の紐づくオブジェクトを初期化
        self.makeElements()
    
    def makeElements(self):
        for signal_group_com in self.com.GetAll():
            self.add(SignalGroup(signal_group_com, self))


class SignalGroup(Object):
    def __init__(self, com, signal_groups):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = signal_groups.config
        self.signal_groups = signal_groups

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = int(self.com.AttValue('No'))
