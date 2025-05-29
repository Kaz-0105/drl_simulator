from libs.container import Container
from libs.object import Object
from objects.links import Lanes
from neural_networks.apex_net import QNet

import torch
import random
from collections import deque

class LocalAgents(Container):
    def __init__(self, upper_object):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = upper_object.config
        self.executor = upper_object.executor

        # 上位オブジェクトによって分岐
        if upper_object.__class__.__name__ == 'Network':
            # 上位の紐づくオブジェクトを取得
            self.network = upper_object

            self._makeElements()

        elif upper_object.__class__.__name__ == 'MasterAgent':
            # 上位の紐づくオブジェクトを取得
            self.master_agent = upper_object
    
    def _makeElements(self):
        intersections = self.network.intersections
        for intersection in intersections.getAll(sorted_flg=True):
            self.add(LocalAgent(self, intersection))
    
    def getState(self):
        # 非同期で状態量を取得
        for agent in self.getAll():
            self.executor.submit(agent.getState)
        
        # 全ての状態量取得が終わるまで待機
        self.executor.wait()
    
    def getAction(self):
        # 非同期で行動を取得
        for agent in self.getAll():
            self.executor.submit(agent.getAction)

        # 全ての行動取得が終わるまで待機
        self.executor.wait()
    
    def getReward(self):
        # 非同期で報酬を取得
        for agent in self.getAll():
            self.executor.submit(agent.getReward)

        # 全ての報酬取得が終わるまで待機
        self.executor.wait()

    def makeLearningData(self):
        # データを送信
        for agent in self.getAll():
            self.executor.submit(agent.makeLearningData)
        
        # 全てのデータ保存が終わるまで待機
        self.executor.wait()
    
    @property
    def done_flg(self):
        for agent in self.getAll():
            if agent.done_flg:
                return True
    
        return False

    
class LocalAgent(Object):
    def __init__(self, local_agents, intersection):
        # 継承
        super().__init__()

        # 設定オブジェクトと非同期処理オブジェクトを取得
        self.config = local_agents.config
        self.executor = local_agents.executor

        # 上位オブジェクトを取得
        self.local_agents = local_agents

        # IDを設定
        self.id = self.local_agents.count() + 1

        # intersectionオブジェクトと紐づける
        self.intersection = intersection
        self.intersection.set('local_agent', self)

        # master_agentと紐づける
        self._makeMasterAgentConnections()

        # roadオブジェクトおよびlaneオブジェクトと紐づける（一方通行）
        self.roads = self.intersection.input_roads
        self._makeRoadLanesMap()

        # 1回の推論で決定する時間幅を取得
        apex_info = self.config.get('apex_info')
        self.duration_steps = apex_info['duration_steps']

        # 強化学習関連のハイパーパラメータを取得
        self.td_steps = apex_info['td_steps']
        self.epsilon = apex_info['epsilon']
        self.gamma = apex_info['gamma']

        # ネットワーク関連のハイパーパラメータを取得
        drl_info = self.config.get('drl_info')
        self.num_vehicles = drl_info['num_vehicles']
        self.num_lanes_map = self.master_agent.num_lanes_map

        # 特徴量に関する設定を取得
        self.features_info = drl_info['features']

        # ネットワークを作成
        self._makeNetwork()

        # 状態量，行動，報酬，終了フラグを初期化
        self.current_state = None
        self.current_action = None
        self.current_reward = None
        self.done_flg = False

        # バッファーに送る学習データを格納するためのリストを初期化
        self.learning_data = []

        # 状態，行動，報酬を一時的にストックするための変数を初期化
        self.state_record = deque(maxlen=self.td_steps + 1)
        self.action_record = deque(maxlen=self.td_steps)
        self.reward_record = deque(maxlen=self.td_steps)
    
    def _makeMasterAgentConnections(self):
        # master_agentを取得
        master_agent = self.intersection.master_agent
        self.master_agent = master_agent
        self.master_agent.local_agents.add(self)
    
    def _makeRoadLanesMap(self):
        # road_lanes_mapを初期化
        road_lanes_map = {}

        # 道路を走査
        for road_order_id in self.roads.getKeys(container_flg=True):
            road = self.intersection.input_roads[road_order_id]
            lanes = Lanes(self)

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
            
            road_lanes_map[road_order_id] = lanes

        self.road_lanes_map = road_lanes_map

    def _makeNetwork(self):
        if self.config.get('drl_info')['method'] =='apex':
            # モデルを初期化
            self.model = QNet(self.config, self.master_agent.num_vehicles, self.master_agent.num_lanes_map)

            # 推論用にする
            self.model.eval()

            # master_agentのモデルと同期させる
            self._syncModel()
        
    def _syncModel(self):
        # master_agentのパラメータを取得
        model_state_dict = self.master_agent.model.state_dict()

        # 自分のモデルにパラメータをセット
        self.model.load_state_dict(model_state_dict)

    def _shouldInfer(self):
        # 現在残っている将来のフェーズを取得
        signal_controller = self.intersection.signal_controller
        future_phase_ids = signal_controller.get('future_phase_ids')

        # 2ステップ以上残っていた場合は推論しなくてよい
        if len(future_phase_ids) > 1:
            self.infer_flg = False
            return False
        
        self.infer_flg = True
        return True
    
    def getState(self):
        # 状態量を取得するタイミングかどうかを確認
        self._shouldInfer()
        if self.infer_flg == False:
            return
        
        # Ape-Xの場合
        if self.config.get('drl_info')['method'] == 'apex':
            # 状態量を初期化
            state = {}

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
                lanes = self.road_lanes_map[road_order_id]

                # 車線を走査
                for lane_order_id in lanes.getKeys(container_flg=True, sorted_flg=True):
                    # laneオブジェクトを取得
                    lane = lanes[lane_order_id]

                    # 車線の状態量を初期化
                    lane_state = {}

                    # vehicle_dataを位置情報でソート
                    vehicle_data = lane.get('vehicle_data').copy()
                    vehicle_data.sort_values(by='position', ascending=False, inplace=True)
                    vehicle_data.reset_index(drop=True, inplace=True)

                    # 先頭からnum_vehicles台の車両を取得
                    vehicle_data = vehicle_data.head(self.num_vehicles).copy()

                    # 距離情報を信号との距離に変換
                    length_info = lane.get('length_info')
                    vehicle_data['position'] = length_info['length'] - vehicle_data['position']

                    # 車両に関する状態を取得
                    vehicles_state = {}
                    feature_names = ['position', 'speed', 'in_queue', 'direction']
                    for index in range(self.num_vehicles):
                        if index < vehicle_data.shape[0]:
                            # レコードを取得
                            vehicle = vehicle_data.iloc[index]

                            # 車両の状態量を初期化
                            vehicle_state = []

                            # 特徴量を走査
                            for feature_name, feature_flg in self.features_info['vehicle'].items():
                                # 使わない状態量はスキップ
                                if feature_flg == False:
                                    continue

                                # 方向に関する状態量はone-hotベクトルに変換，それ以外はそのまま追加
                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.intersection.get('num_roads'))
                                    direction_vector[int(vehicle['direction_id'])] = 1
                                    vehicle_state.extend(direction_vector)
                                else: 
                                    vehicle_state.append(int(vehicle[feature_name]))
                            
                            # 自動車が存在するかどうかのフラグの状態量を追加
                            vehicle_state.append(1) 

                            # テンソルに変換してからvehicles_stateに追加  
                            vehicles_state[len(vehicles_state) + 1] = torch.tensor(vehicle_state).float()                    
                        else:
                            # 車両の状態量を初期化
                            vehicle_state = []

                            # 特徴量を走査
                            for feature_name, feature_flg in self.features_info['vehicle'].items():
                                # 使わない状態量はスキップ
                                if feature_flg == False:
                                    continue
                                
                                # 方向に関する状態量はone-hotベクトルに変換，それ以外はそのまま追加
                                if feature_name == 'direction':
                                    direction_vector = [0] * (self.intersection.get('num_roads'))
                                    vehicle_state.extend(direction_vector)
                                else: 
                                    vehicle_state.append(0)
                            
                            # 自動車が存在するかどうかのフラグの状態量を追加
                            vehicle_state.append(0)

                            # テンソルに変換してからvehicles_stateに追加
                            vehicles_state[len(vehicles_state) + 1] = torch.tensor(vehicle_state).float()
                    
                    # 車線の状態量に追加
                    lane_state['vehicles'] = dict(sorted(vehicles_state.items()))

                    # 評価指標に関する状態量を取得
                    lane_state['metric'] = torch.tensor([lane.get('num_vehicles')], dtype=torch.float32)
                    
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
            state['roads'] = dict(sorted(roads_state.items()))

            # フェーズに関する状態量を取得
            current_phase_id = self.intersection.get('current_phase_id')
            phase_state = [0] * (self.intersection.get('num_phases'))
            if current_phase_id is not None:
                phase_state[current_phase_id - 1] = 1
            else:
                phase_state[0] = 1

            # statesに交差点の状態量を追加
            state['phase'] = torch.tensor(phase_state, dtype=torch.float32)

            # 状態量をインスタンス変数に保存
            self.current_state = state
            self.state_record.append(state)

    def getAction(self):
        # 推論の必要がないときはスキップ
        if self.infer_flg == False:
            return
        
        # ε-greedy法に従って行動を選択
        if random.random() < self.epsilon:
            action = random.randint(1, 8)
        else:
            with torch.no_grad():
                action_values = self.model([self.current_state])
                action = torch.argmax(action_values).item() + 1
        
        # 行動をインスタンス変数に保存
        self.current_action = action
        self.action_record.append(action)

        # 信号機の将来のフェーズに追加
        self.intersection.signal_controller.setNextPhase([self.current_action] * self.duration_steps)
    
    def getReward(self):
        # 報酬を計算するタイミングかどうかを確認
        if self.infer_flg == False:
            return
        
        
        # # キューにいる自動車台数を取得
        # road_score_list = []
        # for lanes in self.road_lanes_map.values():
        #     lane_score_list = []
        #     for lane in lanes.getAll():
        #         num_vehs_in_queue = lane.get('num_vehs_in_queue')
        #         lane_score = max(-1, - 2 * num_vehs_in_queue / self.num_vehicles + 1)
        #         lane_score_list.append(lane_score)

        #     # 車線のスコアの平均を取得
        #     road_score = sum(lane_score_list) / len(lane_score_list)
        #     road_score_list.append(road_score)
        
        # # 道路のスコアの平均を取得
        # average_road_score = sum(road_score_list) / len(road_score_list)
        # worst_road_score = min(road_score_list)

        # # 報酬を計算
        # if average_road_score < - 0.8:
        #     self.done_flg = True
        #     self.current_reward = -10
        # else:
        #     self.current_reward = (average_road_score + worst_road_score) / 2
        
        # self.reward_record.append(self.current_reward)


        # # 残っている空きスペースを計算
        # empty_length_list = []
        # road_length_list = []
        # for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
        #     road = self.roads[road_order_id]
        #     road_length = road.get('length')
        #     empty_length_list.append(road_length - road.get('max_queue_length'))
        #     road_length_list.append(road_length)

        # # 空きスペースの長さの最小値を正規化
        # reward = min(empty_length_list) / max(road_length_list)

        # # 空きスペースがない場合はそこで終了
        # if reward <= 0.3:
        #     # 空きスペースがない場合はそこで終了
        #     self.done_flg = True

        #     # 報酬をインスタンス変数に保存
        #     self.current_reward = -50
        #     self.reward_record.append(self.current_reward)
        # else:
        #     # 報酬をインスタンス変数に保存
        #     self.current_reward = reward
        #     self.reward_record.append(reward)
        
    def makeLearningData(self):
        # データが溜まっていない場合はスキップ
        if len(self.state_record) != self.td_steps + 1:
            return
        
        # 状態について
        state = self.state_record[0]
        next_state = self.state_record[-1]

        # 行動について
        action = self.action_record[0]

        # 累積報酬について
        cumulative_reward = 0
        for reward in list(reversed(self.reward_record)):
            cumulative_reward = reward + self.gamma * cumulative_reward

        # 終了フラグについて
        done = int(self.done_flg)

        # マスターに送るデータを作成
        self.learning_data.append((state, action, cumulative_reward, next_state, done))