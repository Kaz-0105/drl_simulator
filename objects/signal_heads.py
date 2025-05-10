from libs.container import Container
from libs.object import Object

class SignalHeads(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = network.config
        self.network = network

        # comオブジェクトを取得
        self.com = self.network.com.SignalHeads

        # 下位の紐づくオブジェクトを初期化
        self.makeElements()
    
    def makeElements(self):
        for signal_head_com in self.com.GetAll():
            self.add(SignalHead(signal_head_com, self))

class SignalHead(Object):
    def __init__(self, com, signal_heads):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = signal_heads.config
        self.signal_heads = signal_heads

        # comオブジェクトを取得
        self.com = com

        # IDを取得
        self.id = int(self.com.AttValue('No'))