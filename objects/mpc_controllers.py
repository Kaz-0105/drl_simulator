from libs.container import Container
from libs.object import Object
from objects.links import Lanes

import numpy as np
import scipy.linalg as la
import pandas as pd
from collections import deque
from scipy.optimize import milp, LinearConstraint, Bounds
import torch

class MpcControllers(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

        if upper_object.__class__.__name__ == 'Network':
            # 上位の紐づくオブジェクトを取得
            self.network = upper_object

            # 要素オブジェクトを初期化
            self._makeElements()
        
        elif upper_object.__class__.__name__ == 'BcBuffer':
            # 上位の紐づくオブジェクトを取得
            self.bc_buffer = upper_object
    
    def _makeElements(self):
        for intersection_order_id in self.network.intersections.getKeys(container_flg=True, sorted_flg=True):
            intersection = self.network.intersections[intersection_order_id]
            self.add(MpcController(self, intersection))
        
    def optimize(self):
        for mpc_controller in self.getAll():
            self.executor.submit(mpc_controller.optimize)
        
        self.executor.wait()

        for mpc_controller in self.getAll():
            mpc_controller.showOptimizationResult()

    
    def updateBcData(self):
        for mpc_controller in self.getAll():
            self.executor.submit(mpc_controller.updateBcData)
        
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
        self.num_signals = self.signal_controller.signal_groups.count()

        # IDを設定
        self.id = intersection.get('id')

        # フェーズの一覧を取得
        self._makePhases()

        # MPCパラメータを取得
        self._initMPCParameters()

        # 長さ情報を計算
        self._makeRoadLengthInfoMap()

        # 車線の組み合わせを取得
        self._makeRoadCombinationsMap()

        # 道路パラメータを取得
        self._initRoadParameters()

        # 過去の信号機変化の有無を保存するリストを初期化
        self.phi_record = deque([np.float64(0)] * self.min_successive_steps, maxlen=self.min_successive_steps)

        # BCバッファのデータ集めに必要な情報を作成
        self._makeBcRoadLanesMap()
        self._makeBcNumLanesList()

    def _makePhases(self):
        # フェーズ情報を取得
        mpc_info = self.config.get('mpc_info')
        for phase_info in mpc_info['phases']:
            if phase_info['num_roads'] == self.num_roads:
                break

        self.num_phases = int(phase_info['type'].split('-')[0])
    
        phases = self.signal_controller.get('phases')
        self.phases = {}
        for phase_order_id, phase_list in phases.items():
            if phase_order_id > self.num_phases:
                break

            self.phases[phase_order_id] = phase_list

    def _initMPCParameters(self):
        simulator_info = self.config.get('simulator_info')
        self.time_step = simulator_info['time_step']

        # MPCのパラメータを取得
        mpc_info = self.config.get('mpc_info')
        self.horizon = mpc_info['horizon']
        self.utilize_steps = mpc_info['utilize_steps']
        self.remained_steps = mpc_info['remained_steps']
        self.min_successive_steps = mpc_info['min_successive_steps']
        self.num_max_changes = mpc_info['num_max_changes']

        # 行動クローンを行うかどうかのフラグを取得
        bc_buffer_info = mpc_info['bc_buffer']
        self.bc_flg = bc_buffer_info['flg']

        # 行動クローンのデータ集めをするとき自動車の特徴量群を取得
        if self.bc_flg:
            drl_info = self.config.get('drl_info')
            self.bc_features_info = drl_info['features']
            self.bc_num_vehicles = drl_info['num_vehicles']

    def _initRoadParameters(self):
        road_combination_params_map = {}
        for road_order_id in range(1, self.num_roads + 1):
            # roadオブジェクトを取得
            road = self.roads[road_order_id]

            # combinations_mapを取得
            combinations_map = self.road_combinations_map[road_order_id]
            
            # 対応する長さ情報を取得
            length_info = self.road_length_info_map[road_order_id]

            # 法定速度を取得
            max_speed = road.get('max_speed')

            combination_params_map = {}

            # 組み合わせごとに操作
            for combination_order_id, combinations in combinations_map.items():
                # 道路パラメータを初期化
                if len(combinations) == 1:
                    params = {'p_s': {}}                
                else:
                    params = {'p_s': {}, 'D_b': {}}

                # 信号機の位置p_sを取得
                for lane_str in combinations:
                    lane_info = lane_str.split('-')
                    link_id = int(lane_info[0])

                    link = road.links[link_id]
                    link_type = link.get('type')

                    if link_type == 'main':
                        params['p_s'][lane_str] = length_info[link_id]['length']
                    
                    else:
                        params['p_s'][lane_str] = length_info[link_id]['start_pos'] + length_info[link_id]['length']

                # 法定速度v_maxを取得
                params['v_max'] = max_speed * 1000 / 3600

                # 信号の影響圏内に入る距離を定義
                params['D_s'] = max_speed

                # 停止線から信号までの距離を定義
                params['d_s'] = 0

                # 先行車の影響圏に入る距離を定義
                params['D_f'] = max_speed if max_speed > 60 else max_speed - 15

                # 先行車と最接近したときの距離
                params['d_f'] = 5

                # モデルで使う定数を定義k_s, k_f
                params['k_s'] = 1 / (params['D_s'] - params['d_s'])
                params['k_f'] = 1 / (params['D_f'] - params['d_f'])

                # 目的関数で見る信号機付近の範囲
                params['D_t'] = 50

                # 車線分岐点から信号までの距離を取得
                if len(combinations) != 1:
                    link_lane_str_map = {}
                    for lane_str in combinations:
                        lane_info = lane_str.split('-')
                        link_id = int(lane_info[0])

                        link = road.links[link_id]
                        link_type = link.get('type')

                        if link_type == 'main':
                            link_lane_str_map[link_id] = lane_str
                        
                    for lane_str in combinations:
                        lane_info = lane_str.split('-')
                        link_id = int(lane_info[0])

                        link = road.links[link_id]
                        link_type = link.get('type')

                        if link_type == 'main':
                            params['D_b'][lane_str] = params['D_b'][lane_str] + length_info[link_id]['length'] if lane_str in params['D_b'] else length_info[link_id]['length']
                        
                        else:
                            from_connector = link.from_links.getAll()[0]
                            from_connector_id = from_connector.get('id')
                            params['D_b'][lane_str] = length_info[link_id]['start_pos'] + length_info[link_id]['length'] - length_info[from_connector_id]['start_pos']

                            from_link = from_connector.from_links.getAll()[0]
                            from_link_id = from_link.get('id')
                            params['D_b'][link_lane_str_map[from_link_id]] = params['D_b'][link_lane_str_map[from_link_id]] - length_info[from_connector_id]['start_pos'] if from_link_id in params['D_b'] else - length_info[from_connector_id]['start_pos']

                combination_params_map[combination_order_id] = params

            road_combination_params_map[road_order_id] = combination_params_map
        
        self.road_combination_params_map = road_combination_params_map

    def _makeBcRoadLanesMap(self):
        # 行動クローンのデータ集めをしない場合はスキップ
        if not self.bc_flg:
            return
        
        # 道路のIDからlanesオブジェクトへのマップを初期化
        self.bc_road_lanes_map = {}
        
        for road_order_id in range(1, self.num_roads + 1):
            road = self.roads[road_order_id]
            lanes = Lanes(self)

            # 右折分岐車線，メインの車線，左折分岐車線の順番でlaneオブジェクトをlanesオブジェクトに追加
            for link in road.links.getAll():
                if link.get('type') != 'right':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            for link in road.links.getAll():
                if link.get('type') != 'main':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            for link in road.links.getAll():
                if link.get('type') != 'left':
                    continue

                for lane_id in link.lanes.getKeys(sorted_flg=True):
                    lane = link.lanes[lane_id]
                    lanes.add(lane, lanes.count() + 1)
            
            # lanesオブジェクトをroad_lanes_mapに追加
            self.bc_road_lanes_map[road_order_id] = lanes
        
        return

    def _makeBcNumLanesList(self):
        # 行動クローンのデータ集めをしない場合はスキップ
        if not self.bc_flg:
            return

        # 道路のIDから車線数へのマップを初期化
        self.bc_num_lanes_list = []
        for road_order_id in range(1, self.num_roads + 1):  
            road = self.roads[road_order_id]
            num_lanes = 0
            for link in road.links.getAll():  
                if link.get('type') == 'connector':
                    continue

                num_lanes += link.lanes.count()
            
            self.bc_num_lanes_list.append(num_lanes)
        
        return
    
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

    def _makeRoadCombinationsMap(self):
        road_combinations_map = {}

        for road_order_id in range(1, self.num_roads + 1):
            road = self.roads[road_order_id]
            combinations_map = {}

            left_right_flgs = [False] * 2
            for link in road.links.getAll():
                # リンクタイプを取得
                link_type = link.get('type')

                # リンクタイプが右左折リンクでない場合はスキップ
                if link_type not in ['right', 'left']:
                    continue

                # 車線の組み合わせを初期化
                combinations = []

                # 右左折リンクの車線情報を追加
                link_id = link.get('id')
                for lane in link.lanes.getAll():
                    lane_id = lane.get('id')
                    combinations.append(str(link_id) + '-' + str(lane_id))
                
                # メインリンクの車線情報を追加
                main_link = road.getMainLink()
                link_id = main_link.get('id')

                if link_type == 'right':
                    lane = main_link.lanes[1]
                    left_right_flgs[0] = True
                elif link_type == 'left':
                    lane = main_link.lanes[main_link.lanes.count()]
                    left_right_flgs1 = True
                lane_id = lane.get('id')

                combinations.append(str(link_id) + '-' + str(lane_id))
                combinations_map[len(combinations_map) + 1] = combinations
            
            # メインリンクについて
            main_link = road.getMainLink()
            main_link_id = main_link.get('id')

            from_lane_id = 2 if left_right_flgs[0] else 1
            to_lane_id = (main_link.lanes.count() - 1) if left_right_flgs[1] else main_link.lanes.count()

            for lane_id in range(from_lane_id, to_lane_id + 1):
                combinations = [str(main_link_id) + '-' + str(lane_id)]
                combinations_map[len(combinations_map) + 1] = combinations
            
            road_combinations_map[road_order_id] = combinations_map
        
        self.road_combinations_map = road_combinations_map

    def optimize(self):
        # 残りのステップ数がfixed_stepsと等しくなるまではスキップ
        if not self._shouldCalculate():
            return

        # 自動車のデータを更新
        self._updateVehicleData()

        # 交通流モデルを更新
        self._updateTrafficFlowModel()

        # 最適化問題を更新
        self._updateOptimizationProblem()

        # 最適化問題を解く
        self._solveOptimizationProblem()

        # 将来の信号機のフェーズを更新
        self._updateFuturePhaseIds()

        # 信号が変化したかどうかのフラグを更新（次の最適化計算で使う）
        self._updatePhiRecord()

        return
    
    def _shouldCalculate(self):
        signal_controller = self.intersection.signal_controller
        future_phase_ids = signal_controller.get('future_phase_ids')

        if len(future_phase_ids) <= self.remained_steps:
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

            # 車両データが存在しないとき
            if vehicle_data.shape[0] == 0:
                vehicle_data_map = {}
                for combination_order_id in self.road_combinations_map[road_order_id].keys():
                    vehicle_data_map[combination_order_id] = pd.DataFrame(columns=['id', 'position', 'lane_id', 'link_id', 'direction_id', 'wait_link_id', 'wait_lane_id', 'signal_id'])
                road_vehicle_data_map[road_order_id] = vehicle_data_map
                continue

            # 距離を道路の入口空の距離に変換
            vehicle_data = self._transformPositionData(road_order_id, vehicle_data)
            # 必要ない情報を削除
            vehicle_data = vehicle_data.drop(columns=['in_queue', 'speed', 'road_id']).copy()
        
            # 新たに信号待ちする車線に関する情報および従う信号機を追加するために配列を初期化
            wait_link_ids = []
            wait_lane_ids = []
            signal_ids = []

            # 車両データを走査
            for _, vehicle in vehicle_data.iterrows():
                # next_link_idを取得
                next_link_id = vehicle['next_link_id']

                # 信号機のIDを取得（まだコースが決まってないものの方向は1にしておく）
                direction_id = vehicle['direction_id'] if vehicle['direction_id'] != 0 else 1
                signal_id = (road_order_id - 1) * (self.num_roads - 1) + direction_id
                signal_ids.append(int(signal_id))

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
                    next_next_link = next_link.to_links.getAll()[0]
                    next_next_lane = next_link.to_lane
                    wait_link_ids.append(int(next_next_link.get('id')))
                    wait_lane_ids.append(int(next_next_lane.get('id')))
                else:
                    # 右左折リンクの場合はそこが信号待ちするところ
                    wait_link_ids.append(int(next_link_id))
                    current_link = road.links[vehicle['link_id']]
                    wait_lane_ids.append(int(current_link.to_lane.get('id')))
                
            # wait_link_idsとwait_lane_idsとsignal_idsをデータフレームに追加
            vehicle_data['wait_link_id'] = wait_link_ids
            vehicle_data['wait_lane_id'] = wait_lane_ids
            vehicle_data['signal_id'] = signal_ids
            
            # combinationsごとに分割していく
            vehicle_data_map = {}
            for combination_order_id, combinations in self.road_combinations_map[road_order_id].items():
                # 車両データを取得
                related_vehicle_data = None
                for lane in combinations:
                    wait_link_id, wait_lane_id = lane.split('-')
                    tmp_vehicle_data = vehicle_data[(vehicle_data['wait_link_id'] == int(wait_link_id)) & (vehicle_data['wait_lane_id'] == int(wait_lane_id))].copy()
                    
                    if related_vehicle_data is None: 
                        related_vehicle_data = tmp_vehicle_data.reset_index(drop=True)
                    else:
                        related_vehicle_data = pd.concat([related_vehicle_data, tmp_vehicle_data], ignore_index=True)
                
                # いらない列を削除
                related_vehicle_data = related_vehicle_data.drop(columns=['next_link_id']).copy()

                # 位置の降順でソート
                related_vehicle_data.sort_values(by='position', ascending=False, inplace=True)
                related_vehicle_data.reset_index(drop=True, inplace=True)
                vehicle_data_map[combination_order_id] = related_vehicle_data
            
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

        # 自動車の位置の初期値を更新
        self._updatePosVehs()

        # 変数のリストを更新
        self._updateVariableListMap()

        return
    
    def _updateA(self):
        for road_order_id in range(1, self.num_roads + 1):
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            for _, vehicle_data in vehicle_data_map.items():
                num_vehicles = vehicle_data.shape[0]
                if num_vehicles == 0:
                    continue
                
                tmp_A = np.eye((num_vehicles))
                A_matrix = la.block_diag(A_matrix, tmp_A) if 'A_matrix' in locals() else tmp_A
        
        # 結果とともに自動車が存在するかのフラグも作成
        if 'A_matrix' not in locals():
            self.vehicle_exist_flg = False
        else:
            self.vehicle_exist_flg = True
            self.traffic_flow_model['A'] = A_matrix
 
        return
    
    def _updateB1(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return
        
        for road_order_id in range(1, self.num_roads + 1):
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            for combination_order_id, vehicle_data in vehicle_data_map.items():
                num_vehicles = vehicle_data.shape[0]
                if num_vehicles == 0:
                    continue

                tmp_B1 = np.zeros((num_vehicles, self.num_signals))
                B1_matrix = np.vstack([B1_matrix, tmp_B1]) if 'B1_matrix' in locals() else tmp_B1
        
        self.traffic_flow_model['B1'] = B1_matrix    
        return

    def _updateB2(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return
        
        for road_order_id in range(1, self.num_roads + 1):
            # 自動車データを取得
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            dt = self.time_step

            for combination_order_id, vehicle_data in vehicle_data_map.items():
                # 車両データが空の場合はスキップ
                if vehicle_data.shape[0] == 0:
                    continue

                # 車線の組み合わせとパラメータを取得
                combinations = self.road_combinations_map[road_order_id][combination_order_id]
                params = self.road_combination_params_map[road_order_id][combination_order_id]
                
                # 必要なパラメータを取得
                v = params['v_max']
                k_s = params['k_s']
                k_f = params['k_f']
                
                if len(combinations) == 1:
                    for idx, vehicle in vehicle_data.iterrows():
                        if idx == 0:
                            b2 = np.array([-k_s]) * v * dt
                        else:
                            b2 = np.array([-k_s, k_f, -k_f]) * v * dt
                        
                        B2_matrix = la.block_diag(B2_matrix, b2) if 'B2_matrix' in locals() else b2
                else:
                    first_end_flg = {}
                    for lane_str in combinations:
                        first_end_flg[lane_str] = False

                    for idx, vehicle in vehicle_data.iterrows():
                        lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                        if idx == 0:
                            b2 = np.array([-k_s]) * v * dt
                            first_end_flg[lane_str] = True
                        elif not first_end_flg[lane_str]:
                            b2 = np.array([-k_s, k_f, -k_f]) * v * dt
                            first_end_flg[lane_str] = True
                        else:
                            b2 = np.array([-k_s, k_f, -k_f, k_f, -k_f]) * v * dt

                        B2_matrix = la.block_diag(B2_matrix, b2) if 'B2_matrix' in locals() else b2

        self.traffic_flow_model['B2'] = B2_matrix
        return

    def _updateB3(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return
        
        for road_order_id in range(1, self.num_roads + 1):
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            dt = self.time_step

            for combination_order_id, vehicle_data in vehicle_data_map.items():
                if vehicle_data.shape[0] == 0:
                    continue

                # 車線の組み合わせとパラメータを取得
                combinations = self.road_combinations_map[road_order_id][combination_order_id]
                params = self.road_combination_params_map[road_order_id][combination_order_id]

                # 必要なパラメータを取得
                v = params['v_max']
                k_s = params['k_s']
                k_f = params['k_f']
                d_s = params['d_s']
                d_f = params['d_f']
                
                # 分岐車線があるかどうかで場合分け
                if len(combinations) == 1:
                    p_s = params['p_s'][combinations[0]]
                    for idx, vehicle in vehicle_data.iterrows():
                        if idx == 0:    
                            b3 = np.array([0, 0, k_s * (p_s - d_s) - 1, 0, 0, 0, 1]) * v * dt
                        else:
                            b3 = np.array([0, 0, 0, k_s * (p_s - d_s) - 1, -k_f * d_f - 1, 0, 0, 0, 1]) * v * dt
                        
                        B3_matrix = la.block_diag(B3_matrix, b3) if 'B3_matrix' in locals() else b3
                else:
                    first_end_flg = {}
                    for lane_str in combinations:
                        first_end_flg[lane_str] = False

                    for idx, vehicle in vehicle_data.iterrows():
                        lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                        p_s = params['p_s'][lane_str]
                        if idx == 0:
                            b3 = np.array([0, 0, k_s * (p_s - d_s) - 1, 0, 0, 0, 1]) * v * dt
                            lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                            first_end_flg[lane_str] = True
                        elif not first_end_flg[lane_str]:
                            b3 = np.array([0, 0, 0, 0, k_s * (p_s - d_s) - 1, -k_f * d_f - 1, 0, 0, 0, 1]) * v * dt
                            first_end_flg[lane_str] = True
                        else:
                            b3 = np.array([0, 0, 0, 0, 0, k_s * (p_s - d_s) - 1, -k_f * d_f - 1, -k_f * d_f - 1, 0, 0, 0, 1]) * v * dt

                        B3_matrix = la.block_diag(B3_matrix, b3) if 'B3_matrix' in locals() else b3

        self.traffic_flow_model['B3'] = B3_matrix
        return
    
    def _updateC(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return

        for road_order_id in range(1, self.num_roads + 1):
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            for combination_order_id, vehicle_data in vehicle_data_map.items():
                num_vehicles = vehicle_data.shape[0]
                if num_vehicles == 0:
                    continue

                # tmp_C_matrixを初期化
                tmp_C_matrix = None

                # 車線の組み合わせとパラメータを取得
                combinations = self.road_combinations_map[road_order_id][combination_order_id]
                
                if len(combinations) == 1:
                    for idx, vehicle in vehicle_data.iterrows():
                        if idx == 0:
                            c = np.zeros((16, num_vehicles))

                            # delta_dの定義
                            c[[0, 1], idx] = [1, -1]

                            # delta_pの定義
                            c[[2, 3], idx] = [-1, 1]  

                            # delta_t1の定義
                            c[[6, 7], idx] = [1, -1]

                            # z_1の定義
                            c[[14, 15], idx] = [1, -1]

                        else:
                            c = np.zeros((28, num_vehicles))

                            # delta_dの定義
                            c[[0, 1], idx] = [1, -1]

                            # delta_pの定義
                            c[[2, 3], idx] = [-1, 1]

                            # delta_fの定義
                            c[[4, 5], idx-1] = [1, -1]
                            c[[4, 5], idx] = [-1, 1]

                            # delta_t1の定義
                            c[[10, 11], idx] = [1, -1]

                            # z_1の定義
                            c[[18, 19], idx] = [1, -1]

                            # z_2の定義
                            c[[22, 23], idx-1] = [1, -1]

                            # z_3の定義
                            c[[26, 27], idx] = [1, -1]
                        
                        # tmp_C_matrixに追加
                        tmp_C_matrix = c if tmp_C_matrix is None else np.block([[tmp_C_matrix], [c]])
                    
                else:
                    # 車線ごとにモデル化を終えた車両の最後のインデックスを保持する辞書を初期化
                    last_veh_indices = {}
                    for lane_str in combinations:
                        last_veh_indices[lane_str] = -1
                    
                    for idx, vehicle in vehicle_data.iterrows():
                        lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                        if idx == 0:
                            # 先頭車のc行列を初期化
                            c = np.zeros((16, num_vehicles))

                            # delta_dの定義
                            c[[0, 1], idx] = [1, -1]

                            # delta_pの定義
                            c[[2, 3], idx] = [-1, 1]
                            
                            # delta_t1の定義
                            c[[6, 7], idx] = [1, -1]

                            # z_1の定義
                            c[[14, 15], idx] = [1, -1]

                        elif last_veh_indices[lane_str] == -1:
                            # 準先頭車（分岐前は非先頭車，分岐後は先頭車）のc行列を初期化
                            c = np.zeros((30, num_vehicles))

                            # delta_dの定義
                            c[[0, 1], idx] = [1, -1]

                            # delta_pの定義
                            c[[2, 3], idx] = [-1, 1]

                            # delta_f1の定義
                            c[[4, 5], idx-1] = [1, -1]
                            c[[4, 5], idx] = [-1, 1]

                            # delta_bの定義
                            c[[6, 7], idx] = [1, -1]

                            # delta_t1の定義
                            c[[12, 13], idx] = [1, -1]

                            # z_1の定義
                            c[[20, 21], idx] = [1, -1]

                            # z_2の定義
                            c[[24, 25], idx-1] = [1, -1]

                            # z_3の定義
                            c[[28, 29], idx] = [1, -1]
      
                        else:
                            # 先頭車以外のc行列を初期化
                            c = np.zeros((42, num_vehicles))

                            # 分岐後に追従するべき先行車のインデックスを取得
                            follow_idx = last_veh_indices[lane_str]

                            # delta_dの定義
                            c[[0, 1], idx] = [1, -1]

                            # delta_pの定義
                            c[[2, 3], idx] = [-1, 1]

                            # delta_f1の定義
                            c[[4, 5], idx-1] = [1, -1]
                            c[[4, 5], idx] = [-1, 1]

                            # delta_f2の定義
                            c[[6, 7], follow_idx] = [1, -1]
                            c[[6, 7], idx] = [-1, 1]

                            # delta_bの定義
                            c[[8, 9], idx] = [1, -1]

                            # delta_t1の定義
                            c[[16, 17], idx] = [1, -1]

                            # z_1の定義
                            c[[24, 25], idx] = [1, -1]

                            # z_2の定義
                            c[[28, 29], idx-1] = [1, -1]

                            # z_3の定義
                            c[[32, 33], idx] = [1, -1]

                            # z_4の定義
                            c[[36, 37], follow_idx] = [1, -1]

                            # z_5の定義
                            c[[40, 41], idx] = [1, -1]

                        # last_veh_indicesを更新
                        last_veh_indices[lane_str] = idx
                        
                        # tmp_C_matrixに追加
                        tmp_C_matrix = c if tmp_C_matrix is None else np.block([[tmp_C_matrix], [c]])
                    
                # C_matrixに追加
                C_matrix = la.block_diag(C_matrix, tmp_C_matrix) if 'C_matrix' in locals() else tmp_C_matrix

        # C_matrixを交通流モデルに追加
        self.traffic_flow_model['C'] = C_matrix
        return
    
    def _updateD1(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return

        # 道路ごとに走査
        for road_order_id in range(1, self.num_roads + 1):
            # 道路に紐づくvehicle_data_mapを取得
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            # 各車線の組み合わせごとに走査
            for combination_order_id, vehicle_data in vehicle_data_map.items():
                # 車両データが空の場合はスキップ
                if vehicle_data.shape[0] == 0:
                    continue

                # 車線の組み合わせを取得
                combinations = self.road_combinations_map[road_order_id][combination_order_id]

                # 車線数を取得
                num_lanes = len(combinations)

                # 車線数が複数あるかどうか（分岐があるかどうか）で場合分け
                if num_lanes == 1:
                    for idx, vehicle in vehicle_data.iterrows():
                        if idx == 0:
                            # 先頭車に対するD1行列を初期化
                            d1 = np.zeros((16, self.num_signals))
                            
                            # delta_1の定義
                            d1[[4, 5], int(vehicle['signal_id']) - 1] = [1, -1]

                            # delta_t2の定義
                            d1[[8, 9], int(vehicle['signal_id']) - 1] = [1, -1]

                        else:
                            # 先頭車以外のD1行列を初期化
                            d1 = np.zeros((28, self.num_signals))

                            # delta_1の定義
                            d1[[6, 7], int(vehicle['signal_id']) - 1] = [1, -1]

                            # delta_t2の定義
                            d1[[12, 13], int(vehicle['signal_id']) - 1] = [1, -1]
                    
                        # D1_matrixに追加
                        D1_matrix = np.vstack([D1_matrix, d1]) if 'D1_matrix' in locals() else d1
                else:
                    # 先頭車の処理が終わったかどうかを示すフラグを初期化
                    first_end_flg = {}
                    for lane_str in combinations:
                        first_end_flg[lane_str] = False

                    for idx, vehicle in vehicle_data.iterrows():
                        lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                        if idx == 0:
                            # 先頭車に対するD1行列を初期化
                            d1 = np.zeros((16, self.num_signals))

                            # delta_1の定義
                            d1[[4, 5], int(vehicle['signal_id']) - 1] = [1, -1]

                            # delta_t2の定義
                            d1[[8, 9], int(vehicle['signal_id']) - 1] = [1, -1]

                            # 先頭車のフラグを更新
                            first_end_flg[lane_str] = True

                        elif not first_end_flg[lane_str]:
                            # 準先頭車に対するD1行列を初期化
                            d1 = np.zeros((30, self.num_signals))

                            # delta_1の定義
                            d1[[8, 9], int(vehicle['signal_id']) - 1] = [1, -1]

                            # delta_t2の定義
                            d1[[14, 15], int(vehicle['signal_id']) - 1] = [1, -1]
                        
                            # 準先頭車のフラグを更新
                            first_end_flg[lane_str] = True
                        
                        else:
                            # 先頭車以外のD1行列を初期化
                            d1 = np.zeros((42, self.num_signals))

                            # delta_1の定義
                            d1[[10, 11], int(vehicle['signal_id']) - 1] = [1, -1]

                            # delta_t2の定義
                            d1[[18, 19], int(vehicle['signal_id']) - 1] = [1, -1]
                            
                        # D1_matrixに追加
                        D1_matrix = np.vstack([D1_matrix, d1]) if 'D1_matrix' in locals() else d1
                       
        # D1_matrixを交通流モデルに追加
        self.traffic_flow_model['D1'] = D1_matrix
        return

    def _updateD2(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return

        # 道路ごとに走査
        for road_order_id in range(1, self.num_roads + 1):
            # 道路に紐づくvehicle_data_mapを取得
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            # 各車線の組み合わせごとに走査
            for combination_order_id, vehicle_data in vehicle_data_map.items():
                # 車両データが空の場合はスキップ
                if vehicle_data.shape[0] == 0:
                    continue 

                # 車線の組み合わせを取得
                combinations = self.road_combinations_map[road_order_id][combination_order_id]

                # 車線数を取得
                num_lanes = len(combinations)

                # 車線数が複数あるかどうか（分岐があるかどうか）で場合分け
                if num_lanes == 1:
                    for idx, vehicle in vehicle_data.iterrows():
                        if idx == 0:
                            # 先頭車に対するD2行列を初期化
                            d2 = np.zeros((16, 1))

                            # z_1の定義
                            d2[12:16, 0] = [-1, 1, -1, 1]
                        else:
                            # 先頭車以外のD2行列を初期化
                            d2 = np.zeros((28, 3))

                            # z_1の定義
                            d2[16:20, 0] = [-1, 1, -1, 1]

                            # z_2の定義
                            d2[20:24, 1] = [-1, 1, -1, 1]

                            # z_3の定義
                            d2[24:28, 2] = [-1, 1, -1, 1]
                        
                        D2_matrix = la.block_diag(D2_matrix, d2) if 'D2_matrix' in locals() else d2

                else:
                    # 先頭車の処理が終わったかどうかを示すフラグを初期化
                    first_end_flg = {}
                    for lane_str in combinations:
                        first_end_flg[lane_str] = False

                    for idx, vehicle in vehicle_data.iterrows():
                        lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                        if idx == 0:
                            # 先頭車に対するD2行列を初期化
                            d2 = np.zeros((16, 1))

                            # z_1の定義
                            d2[12:16, 0] = [-1, 1, -1, 1]

                            # 先頭車のフラグを更新
                            first_end_flg[lane_str] = True
                        elif not first_end_flg[lane_str]:
                            # 準先頭車に対するD2行列を初期化
                            d2 = np.zeros((30, 3))

                            # z_1の定義
                            d2[18:22, 0] = [-1, 1, -1, 1]

                            # z_2の定義
                            d2[22:26, 1] = [-1, 1, -1, 1]

                            # z_3の定義
                            d2[26:30, 2] = [-1, 1, -1, 1]

                            # 準先頭車のフラグを更新
                            first_end_flg[lane_str] = True
                        else:
                            # 先頭車以外のD2行列を初期化
                            d2 = np.zeros((42, 5))

                            # z_1の定義
                            d2[22:26, 0] = [-1, 1, -1, 1]

                            # z_2の定義
                            d2[26:30, 1] = [-1, 1, -1, 1]

                            # z_3の定義
                            d2[30:34, 2] = [-1, 1, -1, 1]

                            # z_4の定義
                            d2[34:38, 3] = [-1, 1, -1, 1]

                            # z_5の定義
                            d2[38:42, 4] = [-1, 1, -1, 1]

                        # D2_matrixに追加
                        D2_matrix = la.block_diag(D2_matrix, d2) if 'D2_matrix' in locals() else d2
          
        # D2_matrixを交通流モデルに追加
        self.traffic_flow_model['D2'] = D2_matrix
        return

    def _updateD3(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return

        # 道路ごとに走査
        for road_order_id in range(1, self.num_roads + 1):
            # 道路に紐づくvehicle_data_mapを取得
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            # 道路に紐づく組み合わせのマップとパラメータのマップを取得
            combinations_map = self.road_combinations_map[road_order_id]
            combination_params_map = self.road_combination_params_map[road_order_id]

            # 各車線の組み合わせごとに走査
            for combination_order_id, vehicle_data in vehicle_data_map.items():
                # 車両データが空の場合はスキップ
                if vehicle_data.shape[0] == 0:
                    continue

                # 車線の組み合わせとパラメータを取得
                combinations = combinations_map[combination_order_id]
                params = combination_params_map[combination_order_id]

                # 車線数を取得
                num_lanes = len(combinations)

                # 必要なパラメータを取得
                v = params['v_max']
                p_max = vehicle_data.iloc[0]['position'] + v * self.time_step * (self.horizon + 1)
                p_min = - v * self.time_step
                D_s = params['D_s']
                d_s = params['d_s']
                D_f = params['D_f']
                D_t = params['D_t']
                h3_min = - p_max + p_min + D_f
                h3_max = - p_min + p_max + D_f
                h4_min = - p_max + p_min + D_s
                h4_max = - p_min + p_max + D_s

                # 車線数が複数あるかどうか（分岐があるかどうか）で場合分け
                if num_lanes == 1:
                    # 進路ごとに最後にモデル化を終えた車両のインデックスを保持する辞書を初期化
                    last_vehs_map = {}
                    for direction_id in range(1, self.num_roads):
                        last_vehs_map[direction_id] = {
                            'idx': -1,
                            'row': [-2, -1],
                            'col': -1,
                        }
                    
                    # 必要なパラメータを取得
                    p_s = params['p_s'][combinations[0]]
                    h1_min = - p_max + p_s - D_s
                    h1_max = - p_min + p_s - D_s
                    h2_min = p_min - p_s + d_s
                    h2_max = p_max - p_s + d_s
                    h6_min = - p_max + p_s - D_t
                    h6_max = - p_min + p_s - D_t

                    for idx, vehicle in vehicle_data.iterrows():
                        if idx == 0:
                            # 先頭車に対するD3行列を初期化
                            d3 = np.zeros((16, 7))

                            # delta_d(0)の定義
                            d3[0, 0] = - h1_min
                            d3[1, 0] = - h1_max

                            # delta_p(1)の定義
                            d3[2, 1] = - h2_min
                            d3[3, 1] = - h2_max

                            # delta_1(2)の定義
                            d3[4, [0, 1, 2]] = [1, 1, 3]
                            d3[5, [0, 1, 2]] = [-1, -1, -1]

                            # delta_t1(3)の定義
                            d3[6, 3] = - h6_min
                            d3[7, 3] = - h6_max

                            # delta_t2(4)の定義
                            d3[8, [1, 3, 4]] = [1, 1, 3]
                            d3[9, [1, 3, 4]] = [-1, -1, -1]

                            # delta_t3(5)の定義
                            d3[10, 5] = -1
                            d3[11, 5] = 1

                            # d3のサイズを取得
                            rows_delta_t3 = [10, 11]
                            col_delta_t3 = 5

                            # z_1の定義
                            d3[12:16, 2] = [p_min, -p_max, p_max, -p_min] 

                        else:
                            # 先頭車以外のD3行列を初期化
                            d3 = np.zeros((28, 9))

                            # delta_d(0)の定義
                            d3[0, 0] = - h1_min
                            d3[1, 0] = - h1_max

                            # delta_p(1)の定義
                            d3[2, 1] = - h2_min
                            d3[3, 1] = - h2_max

                            # delta_f(2)の定義
                            d3[4, 2] = - h3_min
                            d3[5, 2] = - h3_max

                            # delta_1(3)の定義
                            d3[6, [0, 1, 3]] = [1, 1, 3]
                            d3[7, [0, 1, 3]] = [-1, -1, -1]

                            # delta_2(4)の定義
                            d3[8, [2, 3, 4]] = [-1, 1, 2]
                            d3[9, [2, 3, 4]] = [1, -1, -1]

                            # delta_t1(5)の定義
                            d3[10, 5] = - h6_min
                            d3[11, 5] = - h6_max

                            # delta_t2(6)の定義
                            d3[12, [1, 5, 6]] = [1, 1, 3]
                            d3[13, [1, 5, 6]] = [-1, -1, -1]

                            # delta_t3(7)の定義
                            target_idx = -1
                            target_direction_id = None
                            for direction_id in range(1, self.num_roads):
                                if direction_id == int(vehicle['direction_id']):
                                    continue
                                
                                if last_vehs_map[direction_id]['idx'] > target_idx:
                                    target_idx = last_vehs_map[direction_id]['idx']
                                    target_direction_id = direction_id
                            
                            if target_idx == -1:
                                d3[14, 7] = -1
                                d3[15, 7] = 1
                            else:
                                d3[14, [1, 5, 6, 7]] = [1, 1, 1, 4]
                                d3[15, [1, 5, 6, 7]] = [-1, -1, -1, -1]

                            # d3のサイズを取得
                            rows_delta_t3 = [14, 15]
                            col_delta_t3 = 7
                            
                            # z_1の定義
                            d3[16:20, 3] = [p_min, -p_max, p_max, -p_min]

                            # z_2の定義
                            d3[20:24, 4] = [p_min, -p_max, p_max, -p_min]

                            # z_3の定義
                            d3[24:28, 4] = [p_min, -p_max, p_max, -p_min]

                        # D3_matrixのサイズを取得
                        row_D3 = D3_matrix.shape[0] if 'D3_matrix' in locals() else 0
                        col_D3 = D3_matrix.shape[1] if 'D3_matrix' in locals() else 0

                        # last_veh_indicesを更新
                        last_vehs_map[int(vehicle['direction_id'])] = {
                            'idx': idx,
                            'rows': [row + row_D3 for row in rows_delta_t3],
                            'col': col_delta_t3 + col_D3,
                        }
                        # D3_matrixにd3を追加
                        D3_matrix = la.block_diag(D3_matrix, d3) if 'D3_matrix' in locals() else d3

                        # target_idxが存在するとき（自分と同じ車線かつ進路の異なる車両が存在するとき）
                        if idx != 0 and target_direction_id is not None:
                            target_rows = last_vehs_map[int(vehicle['direction_id'])]['rows']
                            target_col = last_vehs_map[target_direction_id]['col']

                            D3_matrix[target_rows, target_col] = [-1, 1]                    
                        
                else:
                    # 先頭車の処理が終わったかどうかを示すフラグを初期化
                    first_end_flg = {}
                    for lane_str in combinations:
                        first_end_flg[lane_str] = False
                    
                    # 進路ごとに最後にモデル化を終えた車両のインデックスを保持する辞書を初期化
                    last_vehs_map = {}
                    for lane_str in combinations:
                        tmp_last_vehs_map = {}
                        for direction_id in range(1, self.num_roads):
                            tmp_last_vehs_map[direction_id] = {
                                'idx': -1,
                                'rows': [-2, -1],
                                'col': -1,
                            }
                        last_vehs_map[lane_str] = tmp_last_vehs_map

                    for idx, vehicle in vehicle_data.iterrows():
                        lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                        
                        # 必要なパラメータを取得
                        p_s = params['p_s'][lane_str]
                        D_b = params['D_b'][lane_str]
                        h1_min = - p_max + p_s - D_s
                        h1_max = - p_min + p_s - D_s
                        h2_min = p_min - p_s + d_s
                        h2_max = p_max - p_s + d_s
                        h5_min = - p_max + p_s - D_b
                        h5_max = - p_min + p_s - D_b
                        h6_min = - p_max + p_s - D_t
                        h6_max = - p_min + p_s - D_t

                        if idx == 0:
                            # 先頭車に対するD3行列を初期化
                            d3 = np.zeros((16, 7))

                            # delta_d(0)の定義
                            d3[0, 0] = - h1_min
                            d3[1, 0] = - h1_max

                            # delta_p(1)の定義
                            d3[2, 1] = - h2_min
                            d3[3, 1] = - h2_max

                            # delta_1(2)の定義
                            d3[4, [0, 1, 2]] = [1, 1, 3]
                            d3[5, [0, 1, 2]] = [-1, -1, -1]

                            # delta_t1(3)の定義
                            d3[6, 3] = - h6_min
                            d3[7, 3] = - h6_max

                            # delta_t2(4)の定義
                            d3[8, [1, 3, 4]] = [1, 1, 3]
                            d3[9, [1, 3, 4]] = [-1, -1, -1]

                            # delta_t3(5)の定義
                            d3[10, 5] = -1
                            d3[11, 5] = 1

                            # d3のサイズを取得
                            rows_delta_t3 = [10, 11]
                            col_delta_t3 = 5

                            # z_1の定義
                            d3[12:16, 2] = [p_min, -p_max, p_max, -p_min]

                            # 先頭車のフラグを更新
                            first_end_flg[lane_str] = True
                        
                        elif not first_end_flg[lane_str]:
                            # 準先頭車に対するD3行列を初期化
                            d3 = np.zeros((30, 10))

                            # delta_d(0)の定義
                            d3[0, 0] = - h1_min
                            d3[1, 0] = - h1_max

                            # delta_p(1)の定義
                            d3[2, 1] = - h2_min
                            d3[3, 1] = - h2_max

                            # delta_f1(2)の定義
                            d3[4, 2] = - h3_min
                            d3[5, 2] = - h3_max

                            # delta_b(3)の定義
                            d3[6, 3] = - h5_min
                            d3[7, 3] = - h5_max

                            # delta_1(4)の定義
                            d3[8, [0, 1, 4]] = [1, 1, 3]
                            d3[9, [0, 1, 4]] = [-1, -1, -1]

                            # delta_2(5)の定義
                            d3[10, [2, 3, 4, 5]] = [-1, -1, 1, 3]
                            d3[11, [2, 3, 4, 5]] = [1, 1, -1, -1]

                            # delta_t1(6)の定義
                            d3[12, 6] = - h6_min
                            d3[13, 6] = - h6_max

                            # delta_t2(7)の定義
                            d3[14, [1, 6, 7]] = [1, 1, 3]
                            d3[15, [1, 6, 7]] = [-1, -1, -1]

                            # delta_t3(8)の定義
                            d3[16, 8] = -1
                            d3[17, 8] = 1
                            
                            # d3のサイズを取得
                            rows_delta_t3 = [16, 17]
                            col_delta_t3 = 8

                            # z_1の定義
                            d3[18:22, 4] = [p_min, -p_max, p_max, -p_min]

                            # z_2の定義
                            d3[22:26, 5] = [p_min, -p_max, p_max, -p_min]

                            # z_3の定義
                            d3[26:30, 5] = [p_min, -p_max, p_max, -p_min]

                            # 準先頭車のフラグを更新
                            first_end_flg[lane_str] = True

                            target_direction_id = None
                        else:
                            # 先頭車以外のD3行列を初期化
                            d3 = np.zeros((42, 12))

                            # delta_d(0)の定義
                            d3[0, 0] = - h1_min
                            d3[1, 0] = - h1_max

                            # delta_p(1)の定義
                            d3[2, 1] = - h2_min
                            d3[3, 1] = - h2_max

                            # delta_f1(2)の定義
                            d3[4, 2] = - h3_min
                            d3[5, 2] = - h3_max

                            # delta_f2(3)の定義
                            d3[6, 3] = - h4_min
                            d3[7, 3] = - h4_max

                            # delta_b(4)の定義
                            d3[8, 4] = - h5_min
                            d3[9, 4] = - h5_max

                            # delta_1(5)の定義
                            d3[10, [0, 1, 5]] = [1, 1, 3]
                            d3[11, [0, 1, 5]] = [-1, -1, -1]

                            # delta_2(6)の定義
                            d3[12, [2, 4, 5, 6]] = [-1, -1, 1, 3]
                            d3[13, [2, 4, 5, 6]] = [1, 1, -1, -1]

                            # delta_3(7)の定義
                            d3[14, [3, 4, 6, 7]] = [-1, 1, 1, 3]
                            d3[15, [3, 4, 6, 7]] = [1, -1, -1, -1]

                            # delta_t1(8)の定義
                            d3[16, 8] = - h6_min
                            d3[17, 8] = - h6_max

                            # delta_t2(9)の定義
                            d3[18, [1, 8, 9]] = [1, 1, 3]
                            d3[19, [1, 8, 9]] = [-1, -1, -1]

                            # delta_t3(10)の定義
                            target_idx = -1
                            target_direction_id = None
                            for direction_id in range(1, self.num_roads):
                                if direction_id == int(vehicle['direction_id']):
                                    continue
                                
                                if last_vehs_map[lane_str][direction_id]['idx'] > target_idx:
                                    target_idx = last_vehs_map[lane_str][direction_id]['idx']
                                    target_direction_id = direction_id
                            
                            if target_idx == -1:
                                d3[20, 10] = -1
                                d3[21, 10] = 1
                            else:
                                d3[20, [1, 8, 9, 10]] = [1, 1, 1, 4]
                                d3[21, [1, 8, 9, 10]] = [-1, -1, -1, -1]

                            # d3のサイズを取得
                            rows_delta_t3 = [20, 21]
                            col_delta_t3 = 10

                            # z_1の定義
                            d3[22:26, 5] = [p_min, -p_max, p_max, -p_min]

                            # z_2の定義
                            d3[26:30, 6] = [p_min, -p_max, p_max, -p_min]

                            # z_3の定義
                            d3[30:34, 6] = [p_min, -p_max, p_max, -p_min]

                            # z_4の定義
                            d3[34:38, 7] = [p_min, -p_max, p_max, -p_min]

                            # z_5の定義
                            d3[38:42, 7] = [p_min, -p_max, p_max, -p_min]

                        # D3_matrixのサイズを取得
                        row_D3 = D3_matrix.shape[0] if 'D3_matrix' in locals() else 0
                        col_D3 = D3_matrix.shape[1] if 'D3_matrix' in locals() else 0

                        # last_vehs_mapを更新
                        last_vehs_map[lane_str][int(vehicle['direction_id'])] = {
                            'idx': idx,
                            'rows': [row + row_D3 for row in rows_delta_t3],
                            'col': col_delta_t3 + col_D3,
                        }

                        # D3_matrixにd3を追加
                        D3_matrix = la.block_diag(D3_matrix, d3) if 'D3_matrix' in locals() else d3

                        # target_idxが存在するとき（自分と同じ車線かつ進路の異なる車両が存在するとき）
                        if idx != 0 and target_direction_id is not None:
                            target_rows = last_vehs_map[lane_str][int(vehicle['direction_id'])]['rows']
                            target_col = last_vehs_map[lane_str][target_direction_id]['col']

                            D3_matrix[target_rows, target_col] = [-1, 1]
                        
        # D3_matrixを交通流モデルに追加
        self.traffic_flow_model['D3'] = D3_matrix
        return

    def _updateE(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return

        # 道路ごとに走査
        for road_order_id in range(1, self.num_roads + 1):
            # 道路に紐づくvehicle_data_mapを取得
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]


            # 道路に紐づく組み合わせのマップと道路パラメータのマップを取得
            combinations_map = self.road_combinations_map[road_order_id]
            combination_params_map = self.road_combination_params_map[road_order_id]

            # 各車線の組み合わせごとに走査
            for combination_order_id, vehicle_data in vehicle_data_map.items():
                # 車両データが空の場合はスキップ
                if vehicle_data.shape[0] == 0:
                    continue
                
                # 道路パラメータと車線の組み合わせを取得
                params = combination_params_map[combination_order_id]
                combinations = combinations_map[combination_order_id]

                # 車線数を取得
                num_lanes = len(combinations)

                # 必要なパラメータを取得
                v = params['v_max']
                p_max = vehicle_data.iloc[0]['position'] + v * self.time_step * (self.horizon + 1)
                p_min = - v * self.time_step
                D_s = params['D_s'] 
                d_s = params['d_s']
                D_f = params['D_f']  
                D_t = params['D_t']
                h3_min = - p_max + p_min + D_f
                h4_min = - p_max + p_min + D_f    

                # 車線数が複数あるかどうか（分岐があるかどうか）で場合分け
                if num_lanes == 1:
                    # 進路ごとに最後にモデル化を終えた車両のインデックスを保持する辞書を初期化
                    last_veh_indices = {}
                    for direction_id in range(1, self.num_roads):
                        last_veh_indices[direction_id] = -1

                    # 必要なパラメータの取得
                    p_s = params['p_s'][combinations[0]]
                    h1_min = - p_max + p_s - D_s
                    h2_min = p_min - p_s + d_s
                    h6_min = -p_max + p_s - D_t

                    for idx, vehicle in vehicle_data.iterrows():
                        if idx == 0:
                            # 先頭車に対するE行列を初期化
                            e = np.zeros((16, 1))

                            # delta_dの定義
                            e[[0, 1], 0] = [p_s - D_s - h1_min, -p_s + D_s]

                            # delta_pの定義
                            e[[2, 3], 0] = [-p_s + d_s - h2_min, p_s - d_s]

                            # delta_1の定義
                            e[[4, 5], 0] = [3, -1]

                            # delta_t1の定義
                            e[[6, 7], 0] = [p_s - D_t - h6_min, -p_s + D_t]

                            # delta_t2の定義
                            e[[8, 9], 0] = [3, -1]

                            # delta_t3の定義
                            e[[10, 11], 0] = [0, 0]

                            # z_1の定義
                            e[12:16, 0] = [0, 0, p_max, -p_min]

                            # last_veh_indicesを更新
                            last_veh_indices[int(vehicle['direction_id'])] = idx
                        else:
                            # 先頭車以外のE行列を初期化
                            e = np.zeros((28, 1))

                            # delta_dの定義
                            e[[0, 1], 0] = [p_s - D_s - h1_min, -p_s + D_s]

                            # delta_pの定義
                            e[[2, 3], 0] = [-p_s + d_s - h2_min, p_s - d_s]

                            # delta_fの定義
                            e[[4, 5], 0] = [D_f - h3_min, -D_f]

                            # delta_1の定義
                            e[[6, 7], 0] = [3, -1]

                            # delta_2の定義
                            e[[8, 9], 0] = [1, 0]

                            # delta_t1の定義
                            e[[10, 11], 0] = [p_s - D_t - h6_min, -p_s + D_t]

                            # delta_t2の定義
                            e[[12, 13], 0] = [3, -1]

                            # delta_t3の定義
                            target_idx = -1
                            for direction_id in range(1, self.num_roads):
                                if int(vehicle['direction_id']) == direction_id:
                                    continue

                                target_idx = max(target_idx, last_veh_indices[direction_id])

                            if target_idx == -1:
                                e[[14, 15], 0] = [0, 0]
                            else:
                                e[[14, 15], 0] = [3, 0]
                            
                            # z_1の定義
                            e[16:20, 0] = [0, 0, p_max, -p_min]

                            # z_2の定義
                            e[20:24, 0] = [0, 0, p_max, -p_min]

                            # z_3の定義
                            e[24:28, 0] = [0, 0, p_max, -p_min]

                            # last_veh_indicesを更新
                            last_veh_indices[int(vehicle['direction_id'])] = idx
                        
                        # E_matrixに追加
                        E_matrix = np.vstack([E_matrix, e]) if 'E_matrix' in locals() else e

                else:
                    # 先頭車の処理が終わったかどうかを示すフラグを初期化
                    first_end_flg = {}
                    for lane_str in combinations:
                        first_end_flg[lane_str] = False

                    # 進路ごとに最後にモデル化を終えた車両のインデックスを保持する辞書を初期化
                    last_veh_indices = {}
                    for lane_str in combinations:
                        last_veh_indices[lane_str] = {}
                        for direction_id in range(1, self.num_roads):
                            last_veh_indices[lane_str][direction_id] = -1

                    for idx, vehicle in vehicle_data.iterrows():
                        lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                        
                        # 必要なパラメータの取得
                        p_s = params['p_s'][lane_str]
                        D_b = params['D_b'][lane_str]
                        h1_min = - p_max + p_s - D_s
                        h2_min = p_min - p_s + d_s
                        h5_min = -p_max + p_s - D_b
                        h6_min = -p_max + p_s - D_t

                        if idx == 0:
                            # 先頭車に対するE行列を初期化
                            e = np.zeros((16, 1))

                            # delta_dの定義
                            e[[0, 1], 0] = [p_s - D_s - h1_min, -p_s + D_s]

                            # delta_pの定義
                            e[[2, 3], 0] = [-p_s + d_s - h2_min, p_s - d_s]

                            # delta_1の定義
                            e[[4, 5], 0] = [3, -1]

                            # delta_t1の定義
                            e[[6, 7], 0] = [p_s - D_t - h6_min, -p_s + D_t]

                            # delta_t2の定義
                            e[[8, 9], 0] = [3, -1]

                            # delta_t3の定義
                            e[[10, 11], 0] = [0, 0]

                            # z_1の定義
                            e[12:16, 0] = [0, 0, p_max, -p_min]

                            # 先頭車のフラグを更新
                            first_end_flg[lane_str] = True

                        elif not first_end_flg[lane_str]:
                            # 準先頭車に対するE行列を初期化
                            e = np.zeros((30, 1))

                            # delta_dの定義
                            e[[0, 1], 0] = [p_s - D_s - h1_min, -p_s + D_s]

                            # delta_pの定義
                            e[[2, 3], 0] = [-p_s + d_s - h2_min, p_s - d_s]

                            # delta_f1の定義
                            e[[4, 5], 0] = [D_f - h3_min, -D_f]

                            # delta_bの定義
                            e[[6, 7], 0] = [p_s - D_b - h5_min, -p_s + D_b]

                            # delta_1の定義
                            e[[8, 9], 0] = [3, -1]

                            # delta_2の定義
                            e[[10, 11], 0] = [1, 1]

                            # delta_t1の定義
                            e[[12, 13], 0] = [p_s - D_t - h6_min, -p_s + D_t]

                            # delta_t2の定義
                            e[[14, 15], 0] = [3, -1]

                            # delta_t3の定義
                            e[[16, 17], 0] = [0, 0]

                            # z_1の定義
                            e[18:22, 0] = [0, 0, p_max, -p_min]

                            # z_2の定義
                            e[22:26, 0] = [0, 0, p_max, -p_min]

                            # z_3の定義
                            e[26:30, 0] = [0, 0, p_max, -p_min]

                            # 準先頭車のフラグを更新
                            first_end_flg[lane_str] = True
                        else:
                            # 先頭車以外のE行列を初期化
                            e = np.zeros((42, 1))

                            # delta_dの定義
                            e[[0, 1], 0] = [p_s - D_s - h1_min, -p_s + D_s]

                            # delta_pの定義
                            e[[2, 3], 0] = [-p_s + d_s - h2_min, p_s - d_s]

                            # delta_f1の定義
                            e[[4, 5], 0] = [D_f - h3_min, -D_f]

                            # delta_f2の定義
                            e[[6, 7], 0] = [D_f - h4_min, -D_f]

                            # delta_bの定義
                            e[[8, 9], 0] = [p_s - D_b - h5_min, -p_s + D_b]

                            # delta_1の定義
                            e[[10, 11], 0] = [3, -1]

                            # delta_2の定義
                            e[[12, 13], 0] = [1, 1]

                            # delta_3の定義
                            e[[14, 15], 0] = [2, 0]

                            # delta_t1の定義
                            e[[16, 17], 0] = [p_s - D_t - h6_min, -p_s + D_t]

                            # delta_t2の定義
                            e[[18, 19], 0] = [3, -1]

                            # delta_t3の定義
                            target_idx = -1
                            for direction_id in range(1, self.num_roads):
                                if int(vehicle['direction_id']) == direction_id:
                                    continue

                                target_idx = max(target_idx, last_veh_indices[lane_str][direction_id])
                            
                            if target_idx == -1:
                                e[[20, 21], 0] = [0, 0]
                            else:
                                e[[20, 21], 0] = [3, 0]
                            
                            # z_1の定義
                            e[22:26, 0] = [0, 0, p_max, -p_min]

                            # z_2の定義
                            e[26:30, 0] = [0, 0, p_max, -p_min]

                            # z_3の定義
                            e[30:34, 0] = [0, 0, p_max, -p_min]

                            # z_4の定義
                            e[34:38, 0] = [0, 0, p_max, -p_min]

                            # z_5の定義
                            e[38:42, 0] = [0, 0, p_max, -p_min]

                        # last_veh_indicesを更新
                        last_veh_indices[lane_str][int(vehicle['direction_id'])] = idx

                        # E_matrixに追加
                        E_matrix = np.vstack([E_matrix, e]) if 'E_matrix' in locals() else e
        
        # E_matrixを交通流モデルに追加
        self.traffic_flow_model['E'] = E_matrix
        return

    def _updatePosVehs(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return

        # 道路ごとに走査
        for road_order_id in range(1, self.num_roads + 1):
            # 道路に紐づくvehicle_data_mapを取得
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            # 各車線の組み合わせごとに走査
            for combination_order_id, vehicle_data in vehicle_data_map.items():
                # 車両データが空の場合はスキップ
                if vehicle_data.shape[0] == 0:
                    continue
                
                # 車両の位置を取得
                tmp_pos_vehs = vehicle_data['position'].values.reshape(-1, 1)

                # pos_vehsに追加
                pos_vehs = np.vstack([pos_vehs, tmp_pos_vehs]) if 'pos_vehs' in locals() else tmp_pos_vehs

        # 位置ベクトルを交通流モデルに追加
        self.traffic_flow_model['pos_vehs'] = pos_vehs
        return

    def _updateVariableListMap(self):
        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return
        
        # 変数リストマップを初期化（現状必要ないのはコメントアウト）
        variable_list_map = {
            'z_1': [],
            'z_2': [],
            'z_3': [],
            'z_4': [],
            'z_5': [],
            # 'delta_d': [],
            # 'delta_p': [],
            # 'delta_f': [],
            # 'delta_f1': [],
            # 'delta_f2': [],
            # 'delta_b': [],
            # 'delta_1': [],
            # 'delta_2': [],
            # 'delta_3': [],
            # 'delta_t1': [],
            'delta_t2': [],
            'delta_t3': [],
            'delta_c': [],
            'phi': [],
        }

        # フェーズの変数リストを追加
        for phase_order_id in range(1, self.num_phases + 1):
            variable_list_map['p_' + str(phase_order_id)] = []

        variable_length_map = {
            'u': self.num_signals,
            'z': None,
            'delta': None,
            'v': None,
        }

        # 現在の変数の数を初期化
        count = variable_length_map['u'] - 1

        # zに関して変数リストを更新
        for road_order_id in range(1, self.num_roads + 1):
            # 道路に紐づくvehicle_data_mapを取得
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            # 道路に紐づく組み合わせのマップを取得
            combinations_map = self.road_combinations_map[road_order_id]

            # 各車線の組み合わせごとに走査
            for combination_order_id, vehicle_data in vehicle_data_map.items():
                # 車両データが空の場合はスキップ
                if vehicle_data.shape[0] == 0:
                    continue

                # 組み合わせを取得
                combinations = combinations_map[combination_order_id]

                if len(combinations) == 1:
                    for idx, vehicle in vehicle_data.iterrows():
                        if idx == 0:
                            # 先頭車の変数を追加
                            variable_list_map['z_1'].append(count + 1)

                            count += 1
                        
                        else:
                            # 先頭車以外の変数を追加
                            variable_list_map['z_1'].append(count + 1)
                            variable_list_map['z_2'].append(count + 2)
                            variable_list_map['z_3'].append(count + 3)

                            count += 3
                else:
                    # 先頭車の処理が終わったかどうかを示すフラグを初期化
                    first_end_flg = {}
                    for lane_str in combinations:
                        first_end_flg[lane_str] = False

                    for idx, vehicle in vehicle_data.iterrows():
                        lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                        if idx == 0:
                            # 先頭車の変数を追加
                            variable_list_map['z_1'].append(count + 1)

                            count += 1

                            # 先頭車のフラグを更新
                            first_end_flg[lane_str] = True
                        
                        elif not first_end_flg[lane_str]:
                            # 準先頭車の変数を追加
                            variable_list_map['z_1'].append(count + 1)
                            variable_list_map['z_2'].append(count + 2)
                            variable_list_map['z_3'].append(count + 3)

                            count += 3

                            # 準先頭車のフラグを更新
                            first_end_flg[lane_str] = True
                        
                        else:
                            # 先頭車以外の変数を追加
                            variable_list_map['z_1'].append(count + 1)
                            variable_list_map['z_2'].append(count + 2)
                            variable_list_map['z_3'].append(count + 3)
                            variable_list_map['z_4'].append(count + 4)
                            variable_list_map['z_5'].append(count + 5)

                            count += 5

        # zの変数の長さを更新                   
        variable_length_map['z'] = (count + 1) - variable_length_map['u']

        # deltaに関して変数リストを更新
        for road_order_id in range(1, self.num_roads + 1):
            # 道路に紐づくvehicle_data_mapを取得
            vehicle_data_map = self.road_vehicle_data_map[road_order_id]

            # 道路に紐づく組み合わせのマップを取得
            combinations_map = self.road_combinations_map[road_order_id]

            # 各車線の組み合わせごとに走査
            for combination_order_id, vehicle_data in vehicle_data_map.items():
                # 車両データが空の場合はスキップ
                if vehicle_data.shape[0] == 0:
                    continue

                # 組み合わせを取得
                combinations = combinations_map[combination_order_id]

                if len(combinations) == 1:
                    for idx, vehicle in vehicle_data.iterrows():
                        if idx == 0:
                            # 先頭車の変数を追加（現状必要ないのはコメントアウト）
                            # variable_list_map['delta_d'].append(count + 1)
                            # variable_list_map['delta_p'].append(count + 2)
                            # variable_list_map['delta_1'].append(count + 3)
                            # variable_list_map['delta_t1'].append(count + 4)
                            variable_list_map['delta_t2'].append(count + 5)
                            variable_list_map['delta_t3'].append(count + 6)
                            variable_list_map['delta_c'].append(count + 7)

                            count += 7
                        else:
                            # 先頭車以外の変数を追加
                            # variable_list_map['delta_d'].append(count + 1)
                            # variable_list_map['delta_p'].append(count + 2)
                            # variable_list_map['delta_f'].append(count + 3)
                            # variable_list_map['delta_1'].append(count + 4)
                            # variable_list_map['delta_2'].append(count + 5)
                            # variable_list_map['delta_t1'].append(count + 6)
                            variable_list_map['delta_t2'].append(count + 7)
                            variable_list_map['delta_t3'].append(count + 8)
                            variable_list_map['delta_c'].append(count + 9)

                            count += 9
                    
                else:
                    # 先頭車の処理が終わったかどうかを示すフラグを初期化
                    first_end_flg = {}
                    for lane_str in combinations:
                        first_end_flg[lane_str] = False

                    for idx, vehicle in vehicle_data.iterrows():
                        lane_str = str(int(vehicle['wait_link_id'])) + '-' + str(int(vehicle['wait_lane_id']))
                        if idx == 0:
                            # 先頭車の変数を追加
                            # variable_list_map['delta_d'].append(count + 1)
                            # variable_list_map['delta_p'].append(count + 2)
                            # variable_list_map['delta_1'].append(count + 3)
                            # variable_list_map['delta_t1'].append(count + 4)
                            variable_list_map['delta_t2'].append(count + 5)
                            variable_list_map['delta_t3'].append(count + 6)
                            variable_list_map['delta_c'].append(count + 7)

                            count += 7

                            # 先頭車のフラグを更新
                            first_end_flg[lane_str] = True
                        
                        elif not first_end_flg[lane_str]:
                            # 準先頭車の変数を追加
                            # variable_list_map['delta_d'].append(count + 1)
                            # variable_list_map['delta_p'].append(count + 2)
                            # variable_list_map['delta_f1'].append(count + 3)
                            # variable_list_map['delta_b'].append(count + 4)
                            # variable_list_map['delta_1'].append(count + 5)
                            # variable_list_map['delta_2'].append(count + 6)
                            # variable_list_map['delta_t1'].append(count + 7)
                            variable_list_map['delta_t2'].append(count + 8)
                            variable_list_map['delta_t3'].append(count + 9)
                            variable_list_map['delta_c'].append(count + 10)

                            count += 10

                            # 準先頭車のフラグを更新
                            first_end_flg[lane_str] = True
                        
                        else:
                            # 先頭車以外の変数を追加
                            # variable_list_map['delta_d'].append(count + 1)
                            # variable_list_map['delta_p'].append(count + 2)
                            # variable_list_map['delta_f1'].append(count + 3)
                            # variable_list_map['delta_f2'].append(count + 4)
                            # variable_list_map['delta_b'].append(count + 5)
                            # variable_list_map['delta_1'].append(count + 6)
                            # variable_list_map['delta_2'].append(count + 7)
                            # variable_list_map['delta_3'].append(count + 8)
                            # variable_list_map['delta_t1'].append(count + 9)
                            variable_list_map['delta_t2'].append(count + 10)
                            variable_list_map['delta_t3'].append(count + 11)
                            variable_list_map['delta_c'].append(count + 12)

                            count += 12

        # deltaの変数の長さを更新
        variable_length_map['delta'] = (count + 1) - variable_length_map['u'] - variable_length_map['z']

        # u, z, deltaの変数を合わせた変数の長さを更新
        variable_length_map['v'] = count + 1

        # phiに関して変数リストを更新
        v_length = variable_length_map['v']
        for step in range(1, self.horizon):
            variable_list_map['phi'].append(v_length * self.horizon + self.num_phases * self.horizon + (self.num_signals + 1) * step - 1)

        # フェーズの変数リストを更新
        for phase_order_id in range(1, self.num_phases + 1):
            phase_str = 'p_' + str(phase_order_id)
            for step in range(1, self.horizon + 1):
                variable_list_map[phase_str].append(v_length * self.horizon + self.num_phases * (step - 1) + phase_order_id - 1)

        # 変数リストマップをインスタンスとして保持
        self.variable_list_map = variable_list_map

        # 変数の長さマップをインスタンスとして保持
        self.variable_length_map = variable_length_map
        return
                    
    def _updateOptimizationProblem(self):
        # 最適化問題の係数をまとめるための辞書を初期化
        self.optimization_problem = {}

        # 自動車が存在しない場合はスキップ
        if not self.vehicle_exist_flg:
            return

        # 不等式制約と等式制約を更新
        self._updateConstraints()

        # 目的関数の更新
        self._updateObjectiveFunction()

        # 変数の上限・下限を更新
        self._updateBounds()

        # バイナリ変数のタイプを更新
        self._updateIntegrality()
        return 
    
    def _updateConstraints(self):
        # 交通流モデルのホライゾン分のステップを1つの不等式にまとめる
        P_matrix, q_matrix, Peq_matrix, qeq_matrix = self._reshapeTrafficModel()

        # 信号機制約を足していく
        P_matrix, q_matrix, Peq_matrix, qeq_matrix = self._updateSignalConstraints(P_matrix, q_matrix, Peq_matrix, qeq_matrix)

        # インスタンスとして保持
        self.optimization_problem['P'] = P_matrix
        self.optimization_problem['q'] = q_matrix
        self.optimization_problem['Peq'] = Peq_matrix
        self.optimization_problem['qeq'] = qeq_matrix
        return
    
    def _reshapeTrafficModel(self):
        # 係数を取得
        A_matrix = self.traffic_flow_model['A']
        B_matrix = np.block([self.traffic_flow_model['B1'], self.traffic_flow_model['B2'], self.traffic_flow_model['B3']])
        C_matrix = self.traffic_flow_model['C']
        D_matrix = np.block([self.traffic_flow_model['D1'], self.traffic_flow_model['D2'], self.traffic_flow_model['D3']])
        E_matrix = self.traffic_flow_model['E']

        # 初期値を取得
        pos_vehs = self.traffic_flow_model['pos_vehs']

        # 行列をホライゾン分に拡張
        A_bar = np.kron(np.ones((self.horizon, 1)), A_matrix)
        B_bar = np.kron(np.tril(np.ones((self.horizon, self.horizon)), k=-1), B_matrix)
        C_bar = np.kron(np.eye(self.horizon), C_matrix)
        D_bar = np.kron(np.eye(self.horizon), D_matrix)
        E_bar = np.kron(np.ones((self.horizon, 1)), E_matrix)

        # 1つの行列不等式にまとめる
        P_matrix = C_bar @ B_bar + D_bar
        q_matrix = E_bar - C_bar @ A_bar @ pos_vehs

        # 信号機の変数分行列を拡張
        signal_matrix = np.zeros((P_matrix.shape[0], self.num_phases * self.horizon + (self.num_signals + 1) * (self.horizon - 1)))
        P_matrix = np.block([P_matrix, signal_matrix])

        # 変数の数をインスタンスとして保持
        self.num_variables = P_matrix.shape[1]

        # delta_c = 1の制約を作成
        delta_c_list = self.variable_list_map['delta_c']
        v_length = self.variable_length_map['v']

        for idx in delta_c_list:
            for step in range(1, self.horizon + 1):
                peq = np.zeros((1, self.num_variables))
                peq[:, v_length * (step - 1) + idx] = 1
                
                qeq = np.array([[1]])

                # Peq_matrix, qeq_matrixに追加
                Peq_matrix = np.vstack([Peq_matrix, peq]) if 'Peq_matrix' in locals() else peq
                qeq_matrix = np.vstack([qeq_matrix, qeq]) if 'qeq_matrix' in locals() else qeq

        return P_matrix, q_matrix, Peq_matrix, qeq_matrix

    def _updateSignalConstraints(self, P_matrix, q_matrix, Peq_matrix, qeq_matrix):
        # 交通流モデルの1ステップ分の変数を取得
        v_length = self.variable_length_map['v']

        # フェーズの変数の定義
        for phase_id in range(1, self.num_phases + 1):
            signal_ids = self.phases[phase_id]

            for step in range(1, self.horizon + 1):
                p = np.zeros((2, self.num_variables))

                for signal_id in signal_ids:
                    p[:, v_length * (step - 1) + signal_id - 1] = [-1, 1]
                
                p[:, v_length * self.horizon + phase_id + self.num_phases * (step - 1) - 1] = [self.num_roads, -1]

                q = np.array([[0], [self.num_roads - 1]])

                # P_matrix, q_matrixに追加
                P_matrix = np.vstack([P_matrix, p])
                q_matrix = np.vstack([q_matrix, q])

        # 信号現示の変化のバイナリ変数を定義
        for step in range(1, self.horizon):
            for signal_id in range(1, self.num_signals + 1):
                p = np.zeros((4, self.num_variables))
                p[:, v_length * (step - 1) + signal_id - 1] = [1, -1, -1, 1]
                p[:, v_length * step + signal_id - 1] = [1, -1, 1, -1]
                p[:, v_length * self.horizon + self.num_phases * self.horizon + (self.num_signals + 1) * (step - 1) + signal_id - 1] = [1, 1, -1, -1]

                q = np.array([[2], [0], [0], [0]])

                # P_matrix, q_matrixに追加
                P_matrix = np.vstack([P_matrix, p])
                q_matrix = np.vstack([q_matrix, q])
        

        # 青になっていい信号の数の制限
        for step in range(1, self.horizon + 1):
            p = np.zeros((1, self.num_variables))
            p[:, (v_length * (step - 1)) : (v_length * (step - 1) + self.num_signals)] = [1] * self.num_signals

            q = np.array([[self.num_roads]])

            # P_matrix, q_matrixに追加
            P_matrix = np.vstack([P_matrix, p])
            q_matrix = np.vstack([q_matrix, q])

        # 青になっていいフェーズの数の制限
        for step in range(1, self.horizon + 1):
            peq = np.zeros((1, self.num_variables))
            for phase_id in range(1, self.num_phases + 1):
                peq[:, v_length * self.horizon + self.num_phases * (step - 1) + phase_id - 1] = 1
            
            qeq = np.array([[1]])

            # Peq_matrix, qeq_matrixに追加
            Peq_matrix = np.vstack([Peq_matrix, peq])
            qeq_matrix = np.vstack([qeq_matrix, qeq])
        
        # 信号の変化の回数の制限（一回の予測につき何回変化を許容するか）
        p = np.zeros((1, self.num_variables))
        for step in range(1, self.horizon):
            p[:, v_length * self.horizon + self.num_phases * self.horizon + (self.num_signals + 1) * step - 1] = 1
        
        q = np.array([[self.num_max_changes]])

        # P_matrix, q_matrixに追加
        P_matrix = np.vstack([P_matrix, p])
        q_matrix = np.vstack([q_matrix, q])

        # 信号機全体で変化しているかのバイナリの定義
        for step in range(1, self.horizon):
            p = np.zeros((2, self.num_variables))
            for signal_id in range(1, self.num_signals + 1):
                p[:, v_length * self.horizon + self.num_phases * self.horizon + (self.num_signals + 1) * (step - 1) + signal_id - 1] = [-1, 1]
            p[:, v_length * self.horizon + self.num_phases * self.horizon + (self.num_signals + 1) * step - 1] = [1, - self.num_signals]

            q = np.array([[0], [0]])

            # P_matrix, q_matrixに追加
            P_matrix = np.vstack([P_matrix, p])
            q_matrix = np.vstack([q_matrix, q])
        
        # 最小連続回数についての制約
        for step in range(1, self.horizon):
            p = np.zeros((1, self.num_variables))
            q = np.array([[1.0]])

            for tmp_step in range(1, step + 1):
                p[:, v_length * self.horizon + self.num_phases * self.horizon + (self.num_signals + 1) * tmp_step - 1] = 1.0

            for tmp_step in range(1, self.min_successive_steps - step + 1):
                q -= self.phi_record[-tmp_step]
            
            # P_matrix, q_matrixに追加
            P_matrix = np.vstack([P_matrix, p])
            q_matrix = np.vstack([q_matrix, q])

        # 採用するステップ以降の入力の固定について
        if self.horizon > self.remained_steps + self.utilize_steps:
            for step in range(self.remained_steps + self.utilize_steps + 1, self.horizon + 1):
                peq = np.zeros((self.num_signals, self.num_variables))
                for signal_id in range(1, self.num_signals + 1):
                    peq[signal_id - 1, v_length * (self.remained_steps + self.utilize_steps - 1) + signal_id - 1] = 1.0
                    peq[signal_id - 1, v_length * (step - 1) + signal_id - 1] = - 1.0
                
                qeq = np.zeros((self.num_signals, 1))

                Peq_matrix = np.vstack([Peq_matrix, peq])
                qeq_matrix = np.vstack([qeq_matrix, qeq])
        
        self.tmp_P_matrix = P_matrix
        self.tmp_q_matrix = q_matrix
        self.tmp_Peq_matrix = Peq_matrix
        self.tmp_qeq_matrix = qeq_matrix

        # 初期値の固定
        future_phase_ids = self.signal_controller.get('future_phase_ids')
        if len(future_phase_ids) != 0:
            for step in range(1, self.remained_steps + 1):
                peq = np.zeros((self.num_phases, self.num_variables))
                for phase_id in range(1, self.num_phases + 1):
                    peq[phase_id - 1, v_length * self.horizon + self.num_phases * (step - 1) + phase_id - 1] = 1.0
                
                qeq = np.zeros((self.num_phases, 1))
                qeq[future_phase_ids[step - 1] - 1, 0] = 1.0
                    
                # Peq_matrix, qeq_matrixに追加
                Peq_matrix = np.vstack([Peq_matrix, peq])
                qeq_matrix = np.vstack([qeq_matrix, qeq])

        return P_matrix, q_matrix, Peq_matrix, qeq_matrix
        
    def _updateObjectiveFunction(self):
        # 目的関数の係数を初期化
        f_matrix = np.zeros(self.num_variables)

        # delta_t2とdelta_t3のリストを取得
        delta_t2_list = self.variable_list_map['delta_t2']
        delta_t3_list = self.variable_list_map['delta_t3']

        # 交通流モデルの変数の長さを取得
        v_length = self.variable_length_map['v']

        # 目的関数を定義（delta_t2とdelta_t3の累積和とする）
        for step in range(1, self.horizon + 1):
            for idx in range(len(delta_t2_list)):
                f_matrix[v_length * (step - 1) + delta_t2_list[idx]] = 1

            for idx in range(len(delta_t3_list)):
                f_matrix[v_length * (step - 1) + delta_t3_list[idx]] = 1

        # 目的関数の係数をインスタンスとして保持
        self.optimization_problem['f'] = f_matrix
        return

    def _updateBounds(self):
        # 変数の下限と上限を初期化
        lb = np.zeros(self.num_variables)
        ub = np.ones(self.num_variables)

        # zに関する変数のリストを取得
        z_list = []
        z_list.extend(self.variable_list_map['z_1'])
        z_list.extend(self.variable_list_map['z_2'])
        z_list.extend(self.variable_list_map['z_3'])
        z_list.extend(self.variable_list_map['z_4'])
        z_list.extend(self.variable_list_map['z_5'])

        # 交通流モデルの変数の長さを取得
        v_length = self.variable_length_map['v']

        # z_1, z_2, z_3, z_4, z_5の変数の下限と上限を設定
        for step in range(1, self.horizon + 1):
            for idx in z_list:
                ub[v_length * (step - 1) + idx] = np.inf

        # インスタンスとして保持
        self.optimization_problem['lb'] = lb
        self.optimization_problem['ub'] = ub
        return

    def _updateIntegrality(self):
        # 変数のタイプを初期化
        integrality = np.full(self.num_variables, 1) # 1は整数変数である

        # zに関する変数のリストを取得
        z_list = []
        z_list.extend(self.variable_list_map['z_1'])
        z_list.extend(self.variable_list_map['z_2'])
        z_list.extend(self.variable_list_map['z_3'])
        z_list.extend(self.variable_list_map['z_4'])
        z_list.extend(self.variable_list_map['z_5'])

        # 交通流モデルの変数の長さを取得
        v_length = self.variable_length_map['v']

        # z_1, z_2, z_3, z_4, z_5の変数のタイプを設定
        for step in range(1, self.horizon + 1):
            for idx in z_list:
                integrality[v_length * (step - 1) + idx] = 0 # 0は連続変数を示す

        # インスタンスとして保持
        self.optimization_problem['integrality'] = integrality
        return

    def _solveOptimizationProblem(self):
        # 自動車が存在するかで場合分け
        if not self.vehicle_exist_flg:
            return
        
        # 目的関数の係数を取得
        f_matrix = self.optimization_problem['f']
        P_matrix = self.optimization_problem['P']
        q_matrix = self.optimization_problem['q'].flatten()
        Peq_matrix = self.optimization_problem['Peq']
        qeq_matrix = self.optimization_problem['qeq'].flatten()
        lb_matrix = self.optimization_problem['lb']
        ub_matrix = self.optimization_problem['ub']
        integrality_matrix = self.optimization_problem['integrality']

        # 問題を定義
        bounds = Bounds(lb=lb_matrix, ub=ub_matrix)
        constraints_ineq = LinearConstraint(P_matrix, np.full(P_matrix.shape[0], -np.inf), q_matrix)
        constraints_eq = LinearConstraint(Peq_matrix, qeq_matrix, qeq_matrix)
        constraints = [constraints_ineq, constraints_eq]

        # オプションを設定
        options = {
            'disp': False,
            'mip_rel_gap': 0.01,
        }

        # 問題を解く（インスタンスとして保持）
        self.response = milp(c=f_matrix, integrality=integrality_matrix, bounds=bounds, constraints=constraints, options=options)

        # 失敗したときのデバッグ用
        if not self.response.success:
            constraints_ineq = LinearConstraint(self.tmp_P_matrix, np.full(self.tmp_P_matrix.shape[0], -np.inf), self.tmp_q_matrix.flatten())
            constraints_eq = LinearConstraint(self.tmp_Peq_matrix, self.tmp_qeq_matrix.flatten(), self.tmp_qeq_matrix.flatten())
            constraints = [constraints_ineq, constraints_eq]
            response = milp(c=f_matrix, integrality=integrality_matrix, bounds=bounds, constraints=constraints)
        
        return
    
    def showOptimizationResult(self):
        # 計算を行っていないときはスキップ
        if not self.should_calculate:
            return
        
        # 自動車が存在するかで場合分け
        if not self.vehicle_exist_flg:
            return
        
        # 最適化結果を表示
        print(f"intersection id: {self.intersection.get('id')}")
        print(f"optimization result:")
        if self.response.success:
            print(f"  condition: optimization problem solved successfully.")
            print(f"  optimal objective values: {round(self.response.fun, 1)}")
        
        else: 
            print(f"  condition: optimization problem failed.") 

        return        

    def _updateFuturePhaseIds(self):
        # signal_controllerから将来のフェーズを取得
        future_phase_ids = self.signal_controller.get('future_phase_ids')

        # 自動車が存在しないときと実行可能解が存在しないときは現在のフェーズを維持
        if not self.vehicle_exist_flg or not self.response.success:
            if len(future_phase_ids) == 0:
                utilize_phase_ids = [1] * (self.remained_steps + self.utilize_steps)
            else:
                utilize_phase_ids = [future_phase_ids[-1]] * (self.utilize_steps)

            self.signal_controller.setNextPhases(utilize_phase_ids)
            return
        
        # 最適化が成功しているとき
        if self.response.success:
            # 最適化計算の結果からフェーズIDを抽出
            optimized_phase_ids = []
            for step in range(1, self.horizon + 1):
                for phase_id in range(1, self.num_phases + 1):
                    value = self.response.x[self.variable_list_map['p_' + str(phase_id)][step - 1]]
                    if round(value) == 1:
                        optimized_phase_ids.append(phase_id)
                        break
            
            # 初めての最適化計算時とそうでないときで場合分け
            if len(future_phase_ids) != 0:
                utilize_phase_ids = optimized_phase_ids[self.remained_steps : (self.remained_steps + self.utilize_steps)]
            else:
                # 最初だけ残ってる将来のフェーズが少ないのでその分多く採用
                utilize_phase_ids = optimized_phase_ids[:(self.remained_steps + self.utilize_steps)]
            
            # 利用する結果をプッシュ
            self.signal_controller.setNextPhases(utilize_phase_ids)
        return
    
    def _updatePhiRecord(self):
        # 自動車が存在しないときと実行可能解がなかったときは現状維持なので0を追加
        if not self.vehicle_exist_flg or not self.response.success:
            phi_values = [0] * self.utilize_steps
            self.phi_record.extend(phi_values)
            return
        
        # phiの変数リストを取得
        phi_list = self.variable_list_map['phi']

        # 最適化結果からphiの値を取得
        phi_values = self.response.x[phi_list]

        # phiの記録を更新
        self.phi_record.extend(phi_values[: self.utilize_steps])
        return
    
    def updateBcData(self):
        # 行動クローンのためのデータ集めをしない場合はスキップ
        if not self.bc_flg:
            return
        
        # 信号変化が起きていない場合は何もしない
        self.signal_change_flg = self.signal_controller.get('signal_change_flg')
        if not self.signal_change_flg:
            return
        
        # 対象とするネットワークによって分岐
        if self.bc_buffer.get('network_id') == 1:
            # BC用の自動車に関する情報を更新
            self._updateBcVehicleData()

            # BC用の状態量を更新
            self._updateBcState()

            # BC用の行動を更新
            self._updateBcAction()

        return

    def _updateBcVehicleData(self):
        # 行動クローンのデータ集めをしない場合はスキップ
        if not self.bc_flg:
            return
        
        # 行動クローン用の車線から車両データへのマップを初期化
        self.bc_lane_str_vehicle_data_map = {}

        # 道路ごとに走査
        for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
            # roadオブジェクトを取得
            road = self.roads[road_order_id]

            # lanesオブジェクトを取得
            lanes = self.bc_road_lanes_map[road_order_id]

            # direction_signal_value_mapを取得
            direction_signal_value_map = road.get('direction_signal_value_map')

            # 車線を走査
            for lane_order_id in lanes.getKeys(container_flg=True, sorted_flg=True):
                # 車線を規定する文字列を作成
                lane_str = f"{road_order_id}-{lane_order_id}"

                # laneオブジェクトを取得
                lane = lanes[lane_order_id]

                # vehicle_dataを位置情報でソート
                vehicle_data = lane.get('vehicle_data').copy()
                vehicle_data.sort_values(by='position', ascending=False, inplace=True)
                vehicle_data.reset_index(drop=True, inplace=True)

                # 先頭からnum_vehicles台の車両を取得
                vehicle_data = vehicle_data.head(self.bc_num_vehicles).copy()

                # 距離情報を信号との距離に変換
                length_info = lane.get('length_info')
                vehicle_data['position'] = length_info['length'] - vehicle_data['position']

                # near_flg（交差点に近いかどうか）を初期化
                near_flgs = []
                for _, row in vehicle_data.iterrows():
                    if row['position'] <= 100:
                        near_flgs.append(True)
                    else:
                        near_flgs.append(False)
                
                # near_flgsをvehicle_dataに追加
                vehicle_data['near_flg'] = near_flgs

                # wait_flg（信号待ちかどうか）を初期化
                wait_flgs = []

                # direction_idを取得
                direction_ids = vehicle_data['direction_id']
                for idx, row in vehicle_data.iterrows():
                    # 交差点に近くない自動車はスコープから外す
                    if not near_flgs[idx]:
                        wait_flgs.append(False)
                        continue

                    # 信号が赤の場合は信号待ち（3は青信号，1は赤信号を表す）
                    signal_value = 3 if row['direction_id'] == 0 else direction_signal_value_map[row['direction_id']]
                    if signal_value == 1:
                        wait_flgs.append(True)
                        continue
                    
                    # 先頭車の場合
                    if len(wait_flgs) == 0:
                        wait_flgs.append(False)
                        continue

                    # 先頭車でない場合は進路が異なる先行車を探す
                    # その自動車が信号待ちをしていたら自分も信号待ちにする
                    found_flg = False
                    for tmp_idx in range(len(wait_flgs) - 1, - 1, -1):
                        if direction_ids[tmp_idx] != row['direction_id']:
                            wait_flgs.append(True if wait_flgs[tmp_idx] else False)
                            found_flg = True
                            break
                    
                    # 先行車が見つからないとき
                    if not found_flg:
                        wait_flgs.append(False)
                            
                # wait_flgsをvehicle_dataに追加
                vehicle_data['wait_flg'] = wait_flgs
                
                self.bc_lane_str_vehicle_data_map[lane_str] = vehicle_data
        
        return
    
    def _updateBcState(self):
        # 行動クローンのデータ集めをしない場合はスキップ
        if not self.bc_flg:
            return
        
        if self.bc_buffer.get('network_id') == 1:
            # 状態量を初期化
            self.bc_state = {}

            # 道路群の状態量を初期化
            roads_state = {}

            # 道路を走査
            for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
                # roadオブジェクトを取得
                road = self.roads[road_order_id]

                # 道路の状態量を初期化
                road_state = {}
                
                # 車線群の状態量を初期化
                lanes_state = {}

                # lanesオブジェクトを取得
                lanes = self.bc_road_lanes_map[road_order_id]

                # 車線を走査
                for lane_order_id in lanes.getKeys(container_flg=True, sorted_flg=True):
                    # laneオブジェクトを取得
                    lane = lanes[lane_order_id]

                    # 車線の状態量を初期化
                    lane_state = {}

                    # 自動車のデータを取得
                    vehicle_data = self.bc_lane_str_vehicle_data_map.get(f"{road_order_id}-{lane_order_id}")
                    
                    # 車両に関する状態を取得
                    vehicles_state = {}
                    for index in range(self.bc_num_vehicles):
                        if index < vehicle_data.shape[0]:
                            # レコードを取得
                            vehicle = vehicle_data.iloc[index]

                            # 車両の状態量を初期化
                            vehicle_state = []

                            # 特徴量を走査
                            for feature_name, feature_flg in self.bc_features_info['vehicle'].items():
                                # 使わない状態量はスキップ
                                if feature_flg == False:
                                    continue

                                # 方向に関する状態量はone-hotベクトルに変換，それ以外はそのまま追加
                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.intersection.get('num_roads'))
                                    direction_vector[int(vehicle['direction_id'])] = 1
                                    vehicle_state.extend(direction_vector)
                                else: 
                                    vehicle_state.append(float(vehicle[feature_name]))
                            
                            # 自動車が存在するかどうかのフラグの状態量を追加
                            vehicle_state.append(1) 

                            # テンソルに変換してからvehicles_stateに追加  
                            vehicles_state[len(vehicles_state) + 1] = torch.tensor(vehicle_state).float()                    
                        else:
                            # 車両の状態量を初期化
                            vehicle_state = []

                            # 特徴量を走査
                            for feature_name, feature_flg in self.bc_features_info['vehicle'].items():
                                # 使わない状態量はスキップ
                                if feature_flg == False:
                                    continue
                                
                                # 方向に関する状態量はone-hotベクトルに変換，それ以外はそのまま追加
                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.intersection.get('num_roads'))
                                    vehicle_state.extend(direction_vector)
                                else: 
                                    vehicle_state.append(0.0)
                            
                            # 自動車が存在するかどうかのフラグの状態量を追加
                            vehicle_state.append(0)

                            # テンソルに変換してからvehicles_stateに追加
                            vehicles_state[len(vehicles_state) + 1] = torch.tensor(vehicle_state, dtype=torch.float32)
                    
                    # 車線の状態量に追加
                    lane_state['vehicles'] = dict(sorted(vehicles_state.items()))

                    # 評価指標に関する状態量を取得
                    lane_state['metric'] = torch.tensor([lane.get('num_vehicles')], dtype=torch.float32)

                    # 道路の情報を取得
                    length_info = lane.get('length_info')
                    
                    # 車線情報に関する状態量を取得（長さ，メインリンクかサブリンクか）
                    if lane.link.get('type') == 'main':
                        lane_state['shape'] = torch.tensor([int(length_info['length']), 1, 0], dtype=torch.float32)
                    elif lane.link.get('type') == 'right' or lane.link.get('type') == 'left':
                        lane_state['shape'] = torch.tensor([int(length_info['length']), 0, 1], dtype=torch.float32)

                    # lanes_stateにlane_stateを追加
                    lanes_state[lane_order_id] = lane_state
                
                # road_stateに車線の状態量を追加
                road_state['lanes'] = dict(sorted(lanes_state.items()))

                # 評価指標の状態量について
                metric_state = []
                metric_state.append(int(road.get('max_queue_length')))
                metric_state.append(int(road.get('average_delay')))

                # road_stateに評価指標の状態量を追加
                road_state['metric'] = torch.tensor(metric_state, dtype=torch.float32)

                # roads_stateにroad_stateを追加
                roads_state[road_order_id] = road_state
            
            # statesに道路の状態量を追加
            self.bc_state['roads'] = dict(sorted(roads_state.items()))

            # フェーズに関する状態量を取得
            current_phase_id = self.intersection.get('current_phase_id')
            phase_state = [0] * (self.intersection.get('num_phases'))
            if current_phase_id is not None:
                phase_state[current_phase_id - 1] = 1
            else:
                phase_state[0] = 1

            # statesに交差点の状態量を追加
            self.bc_state['phase'] = torch.tensor(phase_state, dtype=torch.float32)

        return
    
    def _updateBcAction(self):
        # 行動クローンのデータ集めをしない場合はスキップ
        if not self.bc_flg:
            return
        
        # 行動クローン用のアクションを初期化
        self.bc_action = self.signal_controller.get('next_phase_id')
        return

        
