from libs.object import Object

class BaseController(Object):
    def __init__(self, controllers):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = controllers.config
        self.executor = controllers.executor
        self.controllers = controllers

        # IDを取得
        self.id = self.controllers.count() + 1

        # 続きは各コントローラで実装