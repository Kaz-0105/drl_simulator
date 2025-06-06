from libs.container import Container
from libs.object import Object

import numpy as np
import scipy.linalg as la
import pandas as pd

class MpcControllers(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = network.config
        self.executor = network.executor

        # 上位の紐づくオブジェクトを取得
        self.network = network

        # 要素オブジェクトを初期化
        self._makeElements()
    
    def _makeElements(self):
        for intersection_order_id in self.network.intersections.getKeys(container_flg=True, sorted_flg=True):
            intersection = self.network.intersections[intersection_order_id]
            self.add(MpcController(self, intersection))
        
    def optimize(self):
        for mpc_controller in self.getAll():
            self.executor.submit(mpc_controller.optimize)
        
        self.executor.wait()
    

class MpcController(Object):
    def __init__(self, mpc_controllers, intersection):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = mpc_controllers.config
        self.executor = mpc_controllers.executor

        # 上位の紐づくオブジェクトを取得
        self.mpc_controllers = mpc_controllers

        # intersectionオブジェクトと紐づける
        self.intersection = intersection
        self.intersection.set('mpc_controller', self)

        # roadsオブジェクトを取得
        self.roads = self.intersection.input_roads
        self.num_roads = self.roads.count()

        # signal_controllerオブジェクトを取得
        self.signal_controller = self.intersection.signal_controller

        # IDを設定
        self.id = intersection.get('id')

        # 各種パラメータについて
        self._initParameters()

        # 長さ情報を計算
        self._makeRoadLengthInfoMap()

        # 車線の組み合わせを取得
        self._makeLaneCombinationsMap()
    
    def _initParameters(self):
        simulator_info = self.config.get('simulator_info')
        self.time_step = simulator_info['time_step']

        mpc_info = self.config.get('mpc_info')
        self.horizon = mpc_info['horizon']
        self.used_steps = mpc_info['used_steps']
        self.calc_start_steps = mpc_info['calc_start_steps']
        self.min_successive_steps = mpc_info['min_successive_steps']
        self.num_max_changes = mpc_info['num_max_changes']

    def _makeRoadLengthInfoMap(self):
        road_length_info_map = {}
        for road_order_id in range(1, self.num_roads + 1):
            road = self.roads[road_order_id]
            length_info = {}
            for link in road.links.getAll():
                link_id = link.get('id')
                link_type = link.get('type')
                if link_type == 'main':
                    length_info[link_id] = {
                        'length': link.get('length'),
                        'start_pos': 0,
                    }
                elif link_type == 'connector':
                    length_info[link_id] = {
                        'length': link.get('length'),
                        'start_pos': link.get('from_pos'),
                    }
                elif link_type in ['right', 'left']:
                    from_connector = link.from_links.getAll()[0]
                    length_info[link_id] = {
                        'length': link.get('length'),
                        'start_pos': from_connector.get('from_pos') + from_connector.get('length') - from_connector.get('to_pos'),
                    }
            
            road_length_info_map[road_order_id] = length_info
        self.road_length_info_map = road_length_info_map



    def _makeLaneCombinationsMap(self):
        road_lane_combinations_map = {}

        for road_order_id in range(1, self.num_roads + 1):
            road = self.roads[road_order_id]
            lane_combinations_map = {}

            left_right_flgs = [False] * 2
            for link in road.links.getAll():
                # リンクタイプを取得
                link_type = link.get('type')

                # リンクタイプが右左折リンクでない場合はスキップ
                if link_type not in ['right', 'left']:
                    continue

                # 車線の組み合わせを初期化
                lane_combinations = []

                # 右左折リンクの車線情報を追加
                link_id = link.get('id')
                for lane in link.lanes.getAll():
                    lane_id = lane.get('id')
                    lane_combinations.append(str(link_id) + '-' + str(lane_id))
                
                # メインリンクの車線情報を追加
                main_link = road.getMainLink()
                link_id = main_link.get('id')

                if link_type == 'right':
                    lane = main_link.lanes[1]
                    left_right_flgs[0] = True
                elif link_type == 'left':
                    lane = main_link.lanes[main_link.lanes.count()]
                    left_right_flgs[1] = True
                lane_id = lane.get('id')

                lane_combinations.append(str(link_id) + '-' + str(lane_id))
                lane_combinations_map[len(lane_combinations_map) + 1] = lane_combinations
            
            # メインリンクについて
            main_link = road.getMainLink()
            main_link_id = main_link.get('id')

            from_lane_id = 2 if left_right_flgs[0] else 1
            to_lane_id = (main_link.lanes.count() - 1) if left_right_flgs[1] else main_link.lanes.count()

            for lane_id in range(from_lane_id, to_lane_id + 1):
                lane_combinations = [str(main_link_id) + '-' + str(lane_id)]
                lane_combinations_map[len(lane_combinations_map) + 1] = lane_combinations
            
            road_lane_combinations_map[road_order_id] = lane_combinations_map
        
        self.road_lane_combinations_map = road_lane_combinations_map

    def optimize(self):
        # 残りのステップ数がfixed_stepsと等しくなるまではスキップ
        if not self._shouldCalculate():
            return

        self._updateVehicleData()

        self._updateTrafficFlowModel()

        self._updateSignalConstraints()

        self._updateOptimizationProblem()
    
    def _shouldCalculate(self):
        signal_controller = self.intersection.signal_controller
        future_phase_ids = signal_controller.get('future_phase_ids')

        if len(future_phase_ids) <= self.calc_start_steps:
            self.should_calculate = True
            return True
    
        self.should_calculate = False
        return False
    
    def _transformPositionData(self, road_order_id, vehicle_data):
        # 自動車情報のレコードを走査
        for idx, vehicle in vehicle_data.iterrows():
            # 長さ情報から流入時点での位置を取得し，自動車情報に反映
            link_id = vehicle['link_id']
            length_info = self.road_length_info_map[road_order_id][link_id]
            vehicle_data.at[idx, 'position'] = vehicle['position'] + length_info['start_pos']

        # 位置の降順でソート
        vehicle_data.sort_values(by='position', ascending=False, inplace=True)
        vehicle_data.reset_index(drop=True, inplace=True)

        return vehicle_data

    def _updateVehicleData(self):
        # 道路ごとの自動車情報を格納する変数の初期化
        road_vehicle_data_map = {}
        
        # 道路を走査
        for road_order_id in range(1, self.num_roads + 1):
            # 道路オブジェクトを取得
            road = self.roads[road_order_id]

            # 道路に紐づくリンクのIDのリストを取得
            link_ids = road.links.getKeys()

            # 道路の車両データを取得
            vehicle_data = road.get('vehicle_data')

            if vehicle_data.shape[0] != 0:
                # 距離を道路の入口空の距離に変換
                vehicle_data = self._transformPositionData(road_order_id, vehicle_data)
                # 必要ない情報を削除
                vehicle_data = vehicle_data.drop(columns=['in_queue', 'speed', 'road_id']).copy()
            
                # 新たに信号待ちする車線に関する情報を追加するために配列を初期化
                wait_link_ids = []
                wait_lane_ids = []

                # 車両データを走査
                for _, vehicle in vehicle_data.iterrows():
                    # next_link_idを取得
                    next_link_id = vehicle['next_link_id']

                    # 次のリンクが道路外の場合（交差点のコネクタの場合）
                    if next_link_id not in link_ids:
                        # 今いる車線が信号待ちを行う車線
                        wait_link_ids.append(int(vehicle['link_id']))
                        wait_lane_ids.append(int(vehicle['lane_id']))
                        continue

                    # 次のリンクが道路内の場合（右折レーン，左折レーンに入る場合）
                    next_link = road.links[next_link_id]
                    next_link_type = next_link.get('type')

                    # 次のリンクがコネクタか右左折リンクかで分岐
                    if next_link_type == 'connector':
                        # コネクタリンクの場合はさらに次のリンクと車線が信号待ちするところ
                        next_next_link = next_link.from_links.getAll()[0]
                        next_next_lane = next_link.to_lane
                        wait_link_ids.append(int(next_next_link.get('id')))
                        wait_lane_ids.append(int(next_next_lane.get('id')))
                    else:
                        # 右左折リンクの場合はそこが信号待ちするところ
                        wait_link_ids.append(int(next_link_id))
                        current_link = road.links[vehicle['link_id']]
                        wait_lane_ids.append(int(current_link.to_lane.get('id')))
                    
                # wait_link_idsとwait_lane_idsをデータフレームに追加
                vehicle_data['wait_link_id'] = wait_link_ids
                vehicle_data['wait_lane_id'] = wait_lane_ids
            
            # lane_combinationsごとに分割していく
            vehicle_data_map = {}
            for lane_combination_order_id, lane_combinations in self.road_lane_combinations_map[road_order_id].items():
                # 車両データを取得
                related_vehicle_data = None
                for lane in lane_combinations:
                    wait_link_id, wait_lane_id = lane.split('-')
                    tmp_vehicle_data = vehicle_data[(vehicle_data['wait_link_id'] == int(wait_link_id)) & (vehicle_data['wait_lane_id'] == int(wait_lane_id))].copy()
                    if related_vehicle_data is None: 
                        related_vehicle_data = tmp_vehicle_data
                    else:
                        related_vehicle_data = pd.concat([related_vehicle_data, tmp_vehicle_data], ignore_index=True)
                
                # いらない列を削除
                if related_vehicle_data.shape[0] != 0:
                    related_vehicle_data = related_vehicle_data.drop(columns=['next_link_id']).copy()
                vehicle_data_map[lane_combination_order_id] = related_vehicle_data
            
            road_vehicle_data_map[road_order_id] = vehicle_data_map
        
        self.road_vehicle_data_map = road_vehicle_data_map
        

    def _updateTrafficFlowModel(self):
        # 初期化
        self.traffic_flow_model = {}

        # 各行列を更新する
        self._updateA()
        self._updateB1()
        self._updateB2()
        self._updateB3()
        self._updateC()
        self._updateD1()
        self._updateD2()
        self._updateD3()
        self._updateE()
    
    def _updateA(self):
        A_matrix = None
        for road_order_id in range(1, self.num_roads + 1):
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            for combination_order_id, vehicle_data in vehicle_data_map.items():
                num_vehicles = vehicle_data.shape[0]
                if num_vehicles == 0:
                    continue
                
                if A_matrix is None:
                    A_matrix = np.eye((num_vehicles))
                else:
                    A_matrix = la.block_diag(A_matrix, np.eye(num_vehicles))
        
        self.traffic_flow_model['A'] = A_matrix
    
    def _updateB1(self):
        B1_matrix = None
        for road_order_id in range(1, self.num_roads + 1):
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            for combination_order_id, vehicle_data in vehicle_data_map.items():
                print('test')
                

    def _updateB2(self):
        pass

    def _updateB3(self):
        pass

    def _updateC(self):
        pass

    def _updateD1(self):
        pass

    def _updateD2(self):
        pass

    def _updateD3(self):
        pass

    def _updateE(self):
        pass

    def _updateSignalConstraints(self):
        pass
    def _updateOptimizationProblem(self):
        pass
        




