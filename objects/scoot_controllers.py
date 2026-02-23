from libs.container import Container
from libs.object import Object

from collections import deque

class ScootControllers(Container):
    def __init__(self, network):
        # 継承
        super().__init__()

        # 設定オブジェクト，非同期オブジェクト，共有情報オブジェクトを初期化
        self.config = network.config
        self.executor = network.executor
        self.shared_resources = network.shared_resources

        # 上位オブジェクトと紐づける
        self.network = network

        # 要素オブジェクトを初期化
        self._initElements()
        return
    
    def _initElements(self):
        intersections = self.network.intersections

        for intersection_order_id in intersections.getKeys(container_flg=True, sorted_flg=True):
            intersection = intersections[intersection_order_id]
            scoot_controller = ScootController(self, intersection)
            self.add(scoot_controller)
        
        return
    
    def updateParameters(self):
        for scoot_controller in self.getAll():
            # self.executor.submit(scoot_controller.updateParameters)
            scoot_controller.updateParameters()
        
        # self.executor.wait()
        return
    
class ScootController(Object):
    def __init__(self, scoot_controllers, intersection):
        # 継承
        super().__init__()

        # 設定オブジェクト，非同期オブジェクト，共有情報オブジェクトを初期化
        self.config = scoot_controllers.config
        self.executor = scoot_controllers.executor
        self.shared_resources = scoot_controllers.shared_resources

        # 上位オブジェクトと紐づける
        self.scoot_controllers = scoot_controllers
        self.network = scoot_controllers.network

        # intersectionと紐づける
        self.intersection = intersection
        self.intersection.set('scoot_controller', self)

        # signal_controllerと紐づける
        self.signal_controller = self.intersection.signal_controller

        # 流入道路と紐づける
        self.roads = self.intersection.input_roads

        # idを設定
        self.id = intersection.get('id')

        # パラメータを初期化
        self._initParams()

        # 最初のフェーズを設定
        self._initFutureValues()

        # 交差点内の自動車台数および飽和度を保存するリストを初期化
        self.num_vehicles_record = []
        self.saturations_record = []

        # フェーズごとの飽和度を保存する辞書を初期化
        self.phase_saturation_map = {}
        for phase_id in range(1, self.num_phases + 1):
            self.phase_saturation_map[phase_id] = 0

        self.phase_num_vehicles_map = {}
        for phase_id in range(1, self.num_phases + 1):
            self.phase_num_vehicles_map[phase_id] = 0

        # update_flgsを初期化
        self.update_flgs = {'cycle': False, 'split': False}

        return
    
    def _initParams(self):
        # 設定オブジェクトから情報を取得
        scoot_info = self.config.get('scoot_info')

        # フェーズの順序を設定
        phase_order = [1, 3, 2, 4]

        # 初期パラメータを設定
        self.params = {}
        self.params['cycle'] = scoot_info['initial_parameters']['cycle']
        self.params['split'] = {}
        for phase_id in range(1, len(phase_order) + 1):
            self.params['split'][phase_id] =scoot_info['initial_parameters']['split'][phase_id - 1]

        # 変更タイミングまでの残りステップ数を初期化
        self.remain_steps_info = {}
        self.remain_steps_info['cycle'] = scoot_info['initial_parameters']['cycle']

        sum_steps = 0
        self.remain_steps_info['split'] = deque(maxlen=len(phase_order))
        for idx in range(len(scoot_info['initial_parameters']['split'])):
            sum_steps += scoot_info['initial_parameters']['split'][idx]
            self.remain_steps_info['split'].append({
                'phase': {
                    'from': phase_order[idx],
                    'to': phase_order[(idx + 1) % 4],
                },
                'steps': sum_steps,
                'fixed': False
            })

        # フェーズ数を設定
        self.num_phases = len(phase_order)

        # インクリメント値を設定
        self.change_steps = scoot_info['change_steps']

        # 1サイクルの最大値と1スプリットの最小値を設定
        self.max_cycle = scoot_info['max_cycle']
        self.min_split = scoot_info['min_split']

        # スペーシング閾値を設定
        self.spacing_threshold = scoot_info['spacing_threshold']
        self.saturation_threshold = 1 / self.spacing_threshold

        # フェーズごとのeffective_storage_lengthを設定
        self.phase_effective_storage_length_map = {}
        for phase_id in range(1, self.num_phases + 1):
            self.phase_effective_storage_length_map[phase_id] = 0.0
        
        for road_order_id in range(1, self.roads.count() + 1):
            road = self.roads[road_order_id]
            effective_storage_lengths = road.get('effective_storage_lengths')
            if road_order_id == 1 or road_order_id == 3:
                self.phase_effective_storage_length_map[1] += effective_storage_lengths['left'] + effective_storage_lengths['straight']
                self.phase_effective_storage_length_map[3] += effective_storage_lengths['right']
            elif road_order_id == 2 or road_order_id == 4:
                self.phase_effective_storage_length_map[2] += effective_storage_lengths['left'] + effective_storage_lengths['straight']
                self.phase_effective_storage_length_map[4] += effective_storage_lengths['right']
            else:
                raise ValueError('Invalid road order ID in SCOOT controller.')
        return
    
    def _initFutureValues(self):
        next_split_info = self.remain_steps_info['split'][0]
        phase_ids = [next_split_info['phase']['from']] * next_split_info['steps']
        self.signal_controller.setNextPhases(phase_ids)
        return
    
    def updateParameters(self):
        # 交通情報の更新
        self._updateTrafficInfo()

        # パラメータの調整
        self._checkUpdateNeeds('cycle')
        if self.update_flgs['cycle']:
            self._updateCycleParameters()

        self._checkUpdateNeeds('split')
        if self.update_flgs['split']:
            self._updateSplitParameters()

        # 1ステップ進める（パラメータを減らす，実際の動きはsimulationクラスで行う）
        self._proceedOneStep()

        return
    
    def _checkUpdateNeeds(self, type):
        if type == 'cycle':
            self.update_flgs['cycle'] = (self.remain_steps_info['cycle'] == 0)
        elif type == 'split':
            self.update_flgs['split'] = (self.remain_steps_info['split'][0]['steps'] == self.change_steps['split'] and not self.remain_steps_info['split'][0]['fixed'])
        return
    
    def _updateTrafficInfo(self):
        # num_vehicles_recordの更新
        tmp_num_vehicles = {}
        for phase_id in range(1, self.num_phases + 1):
            tmp_num_vehicles[phase_id] = 0
        
        for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
            road = self.roads[road_order_id]
            route_num_vehs_map = road.get('route_num_vehs_map')

            if road_order_id == 1 or road_order_id == 3:
                tmp_num_vehicles[1] += route_num_vehs_map[1] + route_num_vehs_map[2] # 左折と直進の自動車台数をフェーズ1の台数に加算
                tmp_num_vehicles[3] += route_num_vehs_map[3] # 右折の自動車台数をフェーズ3の台数に加算
            elif road_order_id == 2 or road_order_id == 4:
                tmp_num_vehicles[2] += route_num_vehs_map[1] + route_num_vehs_map[2] # 左折と直進の自動車台数をフェーズ2の台数に加算
                tmp_num_vehicles[4] += route_num_vehs_map[3] # 右折の自動車台数をフェーズ4の台数に加算
            else:
                raise ValueError('Invalid road order ID in SCOOT controller.')
        
        self.num_vehicles_record.append(tmp_num_vehicles)

        # saturations_recordの更新
        tmp_saturations = {}
        for phase_id in range(1, self.num_phases + 1):
            tmp_saturations[phase_id] = tmp_num_vehicles[phase_id] / self.phase_effective_storage_length_map[phase_id]
        
        self.saturations_record.append(tmp_saturations)

        # phase_saturation_mapの更新
        num_vehs_record_length = len(self.num_vehicles_record)
        if num_vehs_record_length == 1:
            # 1番最初の更新時
            for phase_id in range(1, self.num_phases + 1):
                self.phase_saturation_map[phase_id] = self.saturations_record[-1][phase_id]

                self.phase_num_vehicles_map[phase_id] = self.num_vehicles_record[-1][phase_id]

        elif num_vehs_record_length <= self.params['cycle']:
            # レコードの数がサイクル時間に満たないとき
            for phase_id in range(1, self.num_phases + 1):
                self.phase_saturation_map[phase_id] *= (num_vehs_record_length - 1) / num_vehs_record_length
                self.phase_saturation_map[phase_id] += self.saturations_record[-1][phase_id] / num_vehs_record_length

                self.phase_num_vehicles_map[phase_id] *= (num_vehs_record_length - 1) / num_vehs_record_length
                self.phase_num_vehicles_map[phase_id] += self.num_vehicles_record[-1][phase_id] / num_vehs_record_length

        else:
            # サイクル時間分のレコードがそろっているとき
            for phase_id in range(1, self.num_phases + 1):
                self.phase_saturation_map[phase_id] += self.saturations_record[-1][phase_id] / self.params['cycle']
                self.phase_saturation_map[phase_id] -= self.saturations_record[-1 - self.params['cycle']][phase_id] / self.params['cycle']

                self.phase_num_vehicles_map[phase_id] += self.num_vehicles_record[-1][phase_id] / self.params['cycle']
                self.phase_num_vehicles_map[phase_id] -= self.num_vehicles_record[-1 - self.params['cycle']][phase_id] / self.params['cycle']
        return
    
    def _updateSplitParameters(self):
        tmp_partition = self.remain_steps_info['split'][0]

        from_phase = tmp_partition['phase']['from']
        to_phase = tmp_partition['phase']['to']

        from_phase_saturation = self.phase_saturation_map[from_phase]
        to_phase_saturation = self.phase_saturation_map[to_phase]

        if from_phase_saturation > to_phase_saturation:
            # 変化量を決定
            if self.params['split'][to_phase] >= self.change_steps['split'] + self.min_split:
                change_value = self.change_steps['split']
            else:
                change_value = self.params['split'][to_phase] - self.min_split

            # remained_steps_infoの更新
            tmp_partition['steps'] += change_value
            tmp_partition['fixed'] = True

            # paramsの更新
            self.params['split'][from_phase] += change_value
            self.params['split'][to_phase] -= change_value

            # signal_controllerも更新
            self.signal_controller.setNextPhases([from_phase] * change_value)

        elif from_phase_saturation < to_phase_saturation:
            # 変化量を決定
            if self.params['split'][from_phase] >= self.change_steps['split'] + self.min_split:
                change_value = self.change_steps['split']
            else:
                change_value = self.params['split'][from_phase] - self.min_split

            # remained_steps_infoの更新
            tmp_partition['steps'] -= change_value
            tmp_partition['fixed'] = True

            # paramsの更新
            self.params['split'][from_phase] -= change_value
            self.params['split'][to_phase] += change_value

            # signal_controllerも更新
            self.signal_controller.deletePhases(type='end', steps=change_value)
        
        if tmp_partition['steps'] != self.signal_controller.get('remaining_steps'):
            raise ValueError('Inconsistent remaining steps between ScootController and SignalController.')

        return

    def _updateCycleParameters(self):
        # フェーズごとの飽和度の平均を計算（自動車台数で重みづけ）
        avg_saturation = 0.0
        for phase_id in range(1, self.num_phases + 1):
            avg_saturation += self.phase_saturation_map[phase_id] * self.phase_num_vehicles_map[phase_id]
        avg_saturation /= self.total_num_vehicles
        
        if avg_saturation < self.saturation_threshold:
            # 今のフェーズが終わるタイミングとかぶったらスキップ
            if self.remain_steps_info['split'][0]['steps'] == 0:
                return
            
            cumulative_change_value = 0
            for idx in range(len(self.remain_steps_info['split'])):
                # remained_steps_info['split']の更新
                tmp_partition = self.remain_steps_info['split'][idx]
                from_phase = tmp_partition['phase']['from']

                if self.params['split'][from_phase] >= self.change_steps['cycle'] + self.min_split:
                    change_value = self.change_steps['cycle']
                else:
                    change_value = self.params['split'][from_phase] - self.min_split
                
                cumulative_change_value += change_value

                tmp_partition['steps'] -= cumulative_change_value

                # params['split']の更新
                self.params['split'][from_phase] -= change_value

                # signal_controllerの更新
                if idx == 0:
                    self.signal_controller.deletePhases(type='end', steps=change_value)

            # params['cycle']の更新
            self.params['cycle'] -= cumulative_change_value

        elif avg_saturation > self.saturation_threshold:
            if self.params['cycle'] + self.change_steps['cycle']*self.num_phases <= self.max_cycle:
                cumulative_change_value = 0
                for idx in range(len(self.remain_steps_info['split'])):
                    # remained_steps_info['split']の更新
                    tmp_partition = self.remain_steps_info['split'][idx]
                    from_phase = tmp_partition['phase']['from']

                    cumulative_change_value += self.change_steps['cycle']

                    tmp_partition['steps'] += cumulative_change_value

                    # params['split']の更新
                    self.params['split'][from_phase] += self.change_steps['cycle']

                    # signal_controllerの更新
                    if idx == 0:
                        self.signal_controller.setNextPhases([from_phase] * self.change_steps['cycle'])

                # params['cycle']の更新
                self.params['cycle'] += cumulative_change_value
            else:
                # フェーズごとに増やす量を決める（saturationが大きい順に増やす）
                phase_orders = sorted(range(1, self.num_phases + 1), key=lambda x: self.phase_saturation_map[x], reverse=True)
                phase_change_value_map = {}
                for phase_id in range(1, self.num_phases + 1):
                    phase_change_value_map[phase_id] = 0

                adjustable_change_value = self.max_cycle - self.params['cycle']
                for idx in range(adjustable_change_value):
                    phase_id = phase_orders[idx % self.num_phases]
                    phase_change_value_map[phase_id] += 1
                
                cumulative_change_value = 0
                for idx in range(len(self.remain_steps_info['split'])):
                    # remained_steps_info['split']の更新
                    tmp_partition = self.remain_steps_info['split'][idx]
                    from_phase = tmp_partition['phase']['from']

                    cumulative_change_value += phase_change_value_map[from_phase]

                    tmp_partition['steps'] += cumulative_change_value

                    # params['split']の更新
                    self.params['split'][from_phase] += phase_change_value_map[from_phase]

                    # signal_controllerの更新
                    if idx == 0:
                        self.signal_controller.setNextPhases([from_phase] * phase_change_value_map[from_phase])

                # params['cycle']の更新
                self.params['cycle'] += cumulative_change_value

        if self.remain_steps_info['split'][0]['steps'] != self.signal_controller.get('remaining_steps'):
            raise ValueError('Inconsistent remaining steps between ScootController and SignalController.')
        
        return

    def _proceedOneStep(self):
        # remain_steps_info['split']の更新
        phase_change_flg = False
        if self.remain_steps_info['split'][0]['steps'] == 0:
            first_partition = self.remain_steps_info['split'].popleft()
            first_partition['steps'] = self.params['cycle']
            first_partition['fixed'] = False
            self.remain_steps_info['split'].append(first_partition)
            phase_change_flg = True

        # signal_controllerに新しいフェーズを設定
        if phase_change_flg:
            first_partition = self.remain_steps_info['split'][0]
            from_phase = first_partition['phase']['from']
            steps = first_partition['steps']
            self.signal_controller.setNextPhases([from_phase] * steps)
        
        for partition in self.remain_steps_info['split']:
            partition['steps'] -= 1

        # remain_steps_info['cycle']の更新
        if self.remain_steps_info['cycle'] == 0:
            self.remain_steps_info['cycle'] = self.params['cycle']
        
        self.remain_steps_info['cycle'] -= 1
        return

    @property
    def total_num_vehicles(self):
        return sum(self.phase_num_vehicles_map.values())