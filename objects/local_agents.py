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

            self.makeElements()

        elif upper_object.__class__.__name__ == 'MasterAgent':
            # 上位の紐づくオブジェクトを取得
            self.master_agent = upper_object
    
    def makeElements(self):
        intersections = self.network.intersections
        for intersection in intersections.getAll(sorted_flg=True):
            self.add(LocalAgent(self, intersection))
    
    def infer(self):
        for agent in self.getAll():
            self.executor.submit(agent.run)
        
        self.executor.wait()
        
    def calculateReward(self):
        # 非同期で報酬を計算
        for agent in self.getAll():
            self.executor.submit(agent.calculateReward())

        # 全ての報酬計算が終わるまで待機
        self.executor.wait()

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
        self.makeMasterAgentConnections()

        # roadオブジェクトおよびlaneオブジェクトと紐づける（一方通行）
        self.roads = self.intersection.input_roads
        self.makeRoadLanesMap()

        # 1回の推論で決定する時間幅を取得
        drl_info = self.config.get('drl_info')
        self.duration_steps = drl_info['duration_steps']

        # 強化学習関連のハイパーパラメータを取得
        self.epsilon = drl_info['parameters']['epsilon']

        # ネットワーク関連のハイパーパラメータを取得
        self.num_vehicles = drl_info['num_vehicles']
        self.num_lanes_map = self.master_agent.num_lanes_map

        # 特徴量に関する設定を取得
        self.features_info = drl_info['features']

        # ネットワークを作成
        self.makeNetwork()

        # 状態，行動，報酬を一時的にストックするための変数を初期化
        self.state_record = deque(maxlen=2)
        self.action_record = deque(maxlen=2)
        self.reward_record = deque(maxlen=2)

    
    def makeMasterAgentConnections(self):
        # master_agentを取得
        master_agent = self.intersection.master_agent
        self.master_agent = master_agent
        self.master_agent.local_agents.add(self)
    
    def makeRoadLanesMap(self):
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

    def makeNetwork(self):
        if self.config.get('drl_info')['method'] =='apex':
            # モデルを初期化
            self.model = QNet(self.config, self.master_agent.num_vehicles, self.master_agent.num_lanes_map)

            # 推論用にする
            self.model.eval()

            # master_agentのモデルと同期させる
            self.syncModel()
        
    def syncModel(self):
        # master_agentのパラメータを取得
        model_state_dict = self.master_agent.model.state_dict()

        # 自分のモデルにパラメータをセット
        self.model.load_state_dict(model_state_dict)

    def infer(self):
        # 新しい入力を推論するタイミングかどうかを確認
        if self.shouldInfer() == False:
            return
        
        # 状態量の取得
        self.getState()

        # データが溜まっていたら

        # 行動の選択
        self.getAction()

        # 信号機の将来のフェーズに追加
        self.intersection.signal_controller.setNextPhase([self.action] * self.duration_steps)

        return

    def shouldInfer(self):
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
                    vehicle_data = lane.get('vehicle_data')
                    vehicle_data.sort_values(by='position', ascending=False, inplace=True)
                    vehicle_data.reset_index(drop=True, inplace=True)

                    # 先頭からnum_vehicles台の車両を取得
                    vehicle_data = vehicle_data.head(self.num_vehicles)

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
                                    direction_vector = [0] * (self.intersection.get('num_roads') - 1)
                                    direction_vector[int(vehicle['direction_id']) - 1] = 1
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
                                    direction_vector = [0] * (self.intersection.get('num_roads') - 1)
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
            phase_state[current_phase_id - 1] = 1

            # statesに交差点の状態量を追加
            state['phase'] = torch.tensor(phase_state, dtype=torch.float32)

            # 状態量をインスタンス変数に保存
            self.state = state
            self.state_record.append(self.state)

    def getAction(self):
        # ε-greedy法に従って行動を選択
        if random.random() < self.epsilon:
            action = random.randint(1, self.model.get('output_size'))
        else:
            with torch.no_grad():
                action_values = self.model([self.state])
                action = torch.argmax(action_values).item() + 1
        
        # 行動をインスタンス変数に保存
        self.action = action
        self.action_record.append(action)
    
    def calculateReward(self):
        empty_length_list = []
        road_length_list = []
        for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
            road = self.roads[road_order_id]
            road_length = road.get('length')
            empty_length_list.append(road_length - road.get('max_queue_length'))
            road_length_list.append(road_length)

        reward = min(empty_length_list) / max(road_length_list)

        # 報酬をインスタンス変数に保存
        self.reward = reward
        self.reward_record.append(reward)
        
    def sendDataToMaster(self):
        self.master_agent.set()