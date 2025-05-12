from concurrent.futures import ThreadPoolExecutor
from libs.common import Common


class Executor(Common):
    def __init__(self, max_workers):
        # 継承
        super().__init__()

        # スレッドプールを初期化
        self.object = ThreadPoolExecutor(max_workers = max_workers)

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
