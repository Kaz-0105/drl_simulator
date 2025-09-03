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
        self.makeElements()
        return
    
    def makeElements(self):
        intersections = self.network.intersections

        for intersection_order_id in intersections.getKeys(container_flg=True, sorted_flg=True):
            intersection = intersections[intersection_order_id]
            scoot_controller = ScootController(self, intersection)
            self.add(scoot_controller)
        
        return
    
    def updateParameters(self):
        for scoot_controller in self.getAll():
            self.executor.submit(scoot_controller.updateParameters)
        
        self.executor.wait()
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

        # 変更タイミングを初期化
        self._initParams()

        # 最初のフェーズを設定
        self._initFutureValues()
        return
    
    def _initParams(self):
        scoot_info = self.config.get('scoot_info')
        self.params = {}
        self.params['cycle'] = scoot_info['initial_parameters']['cycle']
        self.params['split'] = deque(maxlen=4)
        sum_steps = 0
        phase_order = [1, 3, 2, 4]
        for idx in range(len(scoot_info['initial_parameters']['split'])):
            sum_steps += scoot_info['initial_parameters']['split'][idx]
            self.params['split'].append({
                'phase': phase_order[(idx + 1) % 4],
                'steps': sum_steps
            })
        return
    
    def _initFutureValues(self):
        next_split_info = self.params['split'][0]
        phase_ids = [next_split_info['phase']] * next_split_info['steps']
        self.signal_controller.setNextPhases(phase_ids)
        return
    
    def updateParameters(self):
        self._checkUpdateNeeds()
        if not self.update_flgs['split'] and not self.update_flgs['cycle']:
            return
        
        self._updateTrafficInfo()

        if self.update_flgs['split']:
            self._updateSplitParameters()

        if self.update_flgs['cycle']:
            self._updateCycleParameters()

        return
    
    def _checkUpdateNeeds(self):
        self.update_flgs = {}

        self.update_flgs['split'] = (self.params['split'][0]['steps'] == 1)
        self.update_flgs['cycle'] = (self.params['cycle'] == 0)
        return
    
    def _updateTrafficInfo(self):
        self.road_saturation_map = {}
        for road_order_id in self.roads.getKeys(container_flg=True, sorted_flg=True):
            road = self.roads[road_order_id]
            self.road_saturation_map[road_order_id] = road.get('saturation_info')
        return
    
    def _updateSplitParameters(self):
        return

    def _updateCycleParameters(self):
        return
