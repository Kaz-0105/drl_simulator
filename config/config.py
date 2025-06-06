import yaml
import pandas as pd
from libs.common import Common 

class Config(Common):
    def __init__(self):
        # 継承
        super().__init__()

        # config.yamlを読み込む
        self.readConfigFile()

        # ネットワークのパラメータをcsvから読み込む
        self.roads = pd.read_csv('layout/' + self.simulator_info['network_name'] + '/roads.csv')
        self.road_link_tags = pd.read_csv('layout/' + self.simulator_info['network_name'] + '/road_link_tags.csv')
        self.intersections = pd.read_csv('layout/' + self.simulator_info['network_name'] + '/intersections.csv')
        self.intersection_road_tags = pd.read_csv('layout/' + self.simulator_info['network_name'] + '/intersection_road_tags.csv')
        self.link_input_tags = pd.read_csv('layout/' + self.simulator_info['network_name'] + '/link_input_tags.csv')
        self.intersection_turn_ratio_tags = pd.read_csv('layout/' + self.simulator_info['network_name'] + '/intersection_turn_ratio_tags.csv')

        # 旋回率のテンプレートを取得する
        self.getNumRoadTurnRatioMap()

        # フェーズの一覧を取得する
        self.getNumRoadPhasesMap()
    
    def readConfigFile(self):
        with open('layout/config.yaml', 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

            # simulatorの基本情報について
            self.simulator_info = data['simulator']

            # DRLに関する情報について
            self.drl_info = data['drl']

            # Ape-Xに関する情報について
            self.apex_info = data['apex']

            # MPCに関する情報について
            self.mpc_info = data['mpc']

            # 記録する情報について
            self.records_info = data['records']

    def getNumRoadTurnRatioMap(self):
        self.num_roads_turn_ratio_map = {}
        for num_roads in [3, 4, 5]:
            self.num_roads_turn_ratio_map[num_roads] = pd.read_csv('layout/turn_ratio_templates' + str(num_roads) + '.csv')
    
    def getNumRoadPhasesMap(self):
        self.num_roads_phases_map = {}
        for num_roads in [3, 4, 5]:

            # 3車線と5車線は後で実装する
            if num_roads == 3 or num_roads == 5:
                continue

            self.num_roads_phases_map[num_roads] = pd.read_csv('layout/phases' + str(num_roads) + '.csv')
        
            



