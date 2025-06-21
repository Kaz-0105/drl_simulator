from concurrent.futures import ThreadPoolExecutor
from libs.common import Common

class Executor(Common):
    def __init__(self, vissim):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = vissim.config
        self.vissim = vissim

        # スレッドプールを初期化
        simulator_info = self.config.get('simulator_info')
        self.object = ThreadPoolExecutor(max_workers = simulator_info['max_workers'])

        # 稼働中のfutureオブジェクトを格納する配列を初期化
        self.futures = []

    def submit(self, func, *args):
        # スレッドプールにタスクを追加
        future = self.object.submit(func, *args)
        self.futures.append(future)
    
    def wait(self):
        # 全てのタスクが完了するまで待機
        for future in self.futures:
            future.result()
        
        # futureを空にする
        self.futures = []

    def shutdown(self):
        # スレッドプールをシャットダウン
        self.object.shutdown(wait = True)
        
        # futureを空にする
        self.futures = []

        return
