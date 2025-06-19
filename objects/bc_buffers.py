from libs.container import Container
from libs.object import Object
from objects.mpc_controllers import MpcControllers

from pathlib import Path
import pickle

class BcBuffers(Container):
    def __init__(self, network):
        super().__init__()

        self.network = network
        self.config = network.config
        self.executor = network.executor

        self._makeElements()
    
    def _makeElements(self):
        mpc_controllers = self.network.mpc_controllers
        num_lanes_list_map =[]
        for mpc_controller in mpc_controllers.getAll():
            roads = mpc_controller.roads
            num_lanes_list = []
            for road_order_id in roads.getKeys(container_flg=True, sorted_flg=True):
                road = roads[road_order_id]
                num_lanes = 0
                for link in road.links.getAll():
                    if link.get('type') == 'connector':
                        continue

                    num_lanes += link.lanes.count()
                
                num_lanes_list.append(num_lanes)

            if num_lanes_list not in num_lanes_list_map:
                num_lanes_list_map.append(num_lanes_list)
                self.add(BcBuffer(self, num_lanes_list))

        return
    
    def saveBcData(self):
        for bc_buffer in self.getAll():
            self.executor.submit(bc_buffer.saveBcData)

        self.executor.wait()
    
    def writeToFile(self):
        for bc_buffer in self.getAll():
            bc_buffer.writeToFile()
        
        self.executor.wait()
        return

class BcBuffer(Object):
    def __init__(self, bc_buffers, num_lanes_list):
        super().__init__()

        # 上位の紐づくオブジェクトを取得
        self.bc_buffers = bc_buffers

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = bc_buffers.config
        self.executor = bc_buffers.executor

        # IDを設定
        self.id = self.bc_buffers.count() + 1

        # 車線数情報と道路数を取得
        self.num_lanes_list = num_lanes_list
        self.num_roads = len(num_lanes_list)

        # 1つのファイルのサイズとネットワークIDを初期化
        self._initParams()

        # バッファをロード
        self._loadBuffer()

        # MPCのコントローラと紐づける
        self._makeControllerConnections()
    
    def _initParams(self):
        # 1つのファイルのサイズを取得
        mpc_info = self.config.get('mpc_info')
        bc_buffer_info = mpc_info['bc_buffer']
        self.size = bc_buffer_info['size']

        # ネットワークIDを取得
        self.network_id = bc_buffer_info['network_id']
        return

    def _loadBuffer(self):
        roads_str = ''.join(map(str, self.num_lanes_list))

        self.count = 0
        while True:
            self.count += 1
            self.buffer_path = Path(f'buffers/bc_buffer_{self.network_id}_{roads_str}_{self.count}.pkl')

            if self.buffer_path.exists():
                continue
            
            # BCバッファが一つも存在しない場合は新規でバッファを作成
            if self.count == 1:
                self.buffer = []
                return
            
            # 一番新しいBCバッファを読み込み
            self.count -= 1
            self.buffer_path = Path(f'buffers/bc_buffer_{roads_str}_{self.count}.pkl')
            with open(self.buffer_path, 'rb') as f:
                self.buffer = pickle.load(f)
            
            # BCバッファのサイズが上限に達していない場合はこれを使用
            if len(self.buffer) != self.size:
                return
            
            # 上限に達しているときは新しいバッファを作成
            self.count += 1
            self.buffer_path = Path(f'buffers/bc_buffer_{roads_str}_{self.count}.pkl')
            self.buffer = []
            return
        
    def _makeControllerConnections(self):
        self.mpc_controllers = MpcControllers(self)

        mpc_controllers = self.bc_buffers.network.mpc_controllers

        for mpc_controller in mpc_controllers.getAll():
            if mpc_controller.get('bc_num_lanes_list') == self.num_lanes_list:
                self.mpc_controllers.add(mpc_controller)
                mpc_controller.set('bc_buffer', self)
        
        return
        
    def saveBcData(self):
        for mpc_controller in self.mpc_controllers.getAll():
            if not mpc_controller.get('signal_change_flg'):
                continue

            bc_data = {
                'state': mpc_controller.get('bc_state'),
                'action': mpc_controller.get('bc_action'),
            }

            self.buffer.append(bc_data)

            if len(self.buffer) < self.size:
                continue
            
            # バッファが上限に達した場合はファイルに書き込み，次のファイルに進む
            self.writeToFile()
            self._initNextBuffer()

        return

    def writeToFile(self):
        with open(self.buffer_path, 'wb') as f:
            pickle.dump(self.buffer, f)
        
        return

    def _initNextBuffer(self):
        # バッファを再初期化
        self.buffer = []

        # 次の保存先を設定
        self.count += 1
        roads_str = ''.join(map(str, self.num_lanes_list))
        self.buffer_path = Path(f'buffers/bc_buffer_{self.network_id}_{roads_str}_{self.count}.pkl')
        return
    




        