import yaml
import pandas as pd
from libs.common import Common 

class Config(Common):
    def __init__(self, vissim):
        # 継承
        super().__init__()

        self.vissim = vissim
        self.root_dir_path = self.vissim.get('root_dir_path')

        # config.yamlを読み込む
        self.readConfigFile()

        # ネットワークのパラメータをcsvから読み込む
        self._initNetworkProps()
        
        # 旋回率のテンプレートを取得する
        self._getNumRoadTurnRatioMap()

        # フェーズの一覧を取得する
        self._getNumRoadPhasesMap()

        # 行動クローンのときは1交差点系限定とする
        self._validateBcEnvironment()

        # epsilonのスケジューリングを取得
        self._getEpsilonSchedule()

        # 状態のtemplateを取得
        self._getNetworkStateMap()

        # drl_infoの整形
        self.reshapeDrlInfo()

        # symmetry_phase_tagsについて
        self._getSymmetryPhaseTags()
        return
    
    def readConfigFile(self):
        with open('layout/config.yaml', 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

            # simulatorの基本情報について
            self.simulator_info = data['simulator']

            # DRLに関する情報について
            self.drl_info = data['drl']

            # Ape-Xに関する情報について
            self.apex_info = data['apex']

            if type(data['apex']['learning_rate']) == str:
                self.apex_info['learning_rate'] = float(data['apex']['learning_rate'])  # 文字列になってしまうのでfloatに変換
            
            if type(data['apex']['weight_decay']) == str:
                self.apex_info['weight_decay'] = float(data['apex']['weight_decay'])  # 文字列になってしまうのでfloatに変換

            self.mpc_info = data['mpc']

            self.bc_info = data['bc']
            if type(self.bc_info['weight_decay']) == str:
                self.bc_info['weight_decay'] = float(self.bc_info['weight_decay'])  # 文字列になってしまうのでfloatに変換
            
            if type(self.bc_info['learning_rate']) == str:
                self.bc_info['learning_rate'] = float(self.bc_info['learning_rate'])  # 文字列になってしまうのでfloatに変換

            self.scoot_info = data['scoot']
            self.save_info = data['save']
            return

    def _initNetworkProps(self):
        common_dir_path = self.root_dir_path / 'layout' / self.simulator_info['layout_name']
        with open(common_dir_path / 'roads.csv', 'r', encoding='utf-8') as f:
            self.roads = pd.read_csv(f, index_col=False)
        
        with open(common_dir_path / 'road_link_tags.csv', 'r', encoding='utf-8') as f:
            self.road_link_tags = pd.read_csv(f, index_col=False)
        
        with open(common_dir_path / 'intersections.csv', 'r', encoding='utf-8') as f:
            self.intersections = pd.read_csv(f, index_col=False)

        with open(common_dir_path / 'intersection_road_tags.csv', 'r', encoding='utf-8') as f:
            self.intersection_road_tags = pd.read_csv(f, index_col=False)

        with open(common_dir_path / 'intersection_turn_ratio_tags.csv', 'r', encoding='utf-8') as f:
            self.intersection_turn_ratio_tags = pd.read_csv(f, index_col=False) 

        link_input_tags_dir_path = common_dir_path / 'link_input_tags'
        if not link_input_tags_dir_path.exists():
            raise FileNotFoundError(f"Not found: {link_input_tags_dir_path}")
        
        self.link_input_tags_map = {}
        for link_input_tags_file_path in link_input_tags_dir_path.glob('*.csv'):
            with open(link_input_tags_file_path, 'r', encoding='utf-8') as f:
                self.link_input_tags_map[link_input_tags_file_path.stem] = pd.read_csv(f, index_col=False)
        return
    
    def _getNumRoadTurnRatioMap(self):
        self.num_roads_turn_ratio_map = {}
        for num_roads in [3, 4, 5]:
            turn_ration_file_path = self.root_dir_path / 'layout' / f"turn_ratio_templates{num_roads}.csv"
            if not turn_ration_file_path.exists():
                continue
            self.num_roads_turn_ratio_map[num_roads] = pd.read_csv(turn_ration_file_path, index_col=False)
        
        return

    def _getNumRoadPhasesMap(self):
        self.num_roads_phases_map = {}
        for num_roads in [3, 4, 5]:
            if num_roads == 3 or num_roads == 5:
                continue

            phase_file_path = self.root_dir_path / 'layout' / f"phases{num_roads}.csv"
            if not phase_file_path.exists():
                continue
            self.num_roads_phases_map[num_roads] = pd.read_csv(phase_file_path, index_col=False)
        
        return

    def _validateBcEnvironment(self):
        if self.simulator_info['control_method'] != 'bc':
            return
        
        if self.intersections.shape[0] != 1:
            raise ValueError('The simulation of behavior cloning is only available for a single intersection environment. \n Please check the network configuration in config.yaml.')

    def _getEpsilonSchedule(self):
        if not self.apex_info['epsilon']['schedule_flg']:
            return
        
        eps_schedule_file_path = self.root_dir_path / 'layout' / 'epsilon_schedule.csv'
        if not eps_schedule_file_path.exists():
            raise FileNotFoundError(f"Not found: {eps_schedule_file_path}")
        self.epsilon_schedule = pd.read_csv(eps_schedule_file_path, index_col=False)
        return

    def _getNetworkStateMap(self):
        self.network_state_map = {}

        for network_id in [1]:
            self.network_state_map[network_id] = pd.read_csv(self.root_dir_path / 'layout' / f"state_template{network_id}.csv", index_col=False)
        return
    
    def reshapeDrlInfo(self):
        state_template_df = self.network_state_map[self.drl_info['network_id']]
        target_record = state_template_df[state_template_df['id'] == self.drl_info['state_id']]
        if target_record.empty:
            raise ValueError(f"State ID {self.drl_info['state_id']} not found in state_template{self.drl_info['network_id']}.csv")
        
        self.drl_info['features'] = {
            'vehicle' : {
                'position': bool(target_record['position'].values[0]),
                'speed': bool(target_record['speed'].values[0]),
                'direction': bool(target_record['direction'].values[0])
            },
            'lane' : {
                'metric': {
                    'num_vehicles': bool(target_record['num_vehicles'].values[0]),
                },
                'shape' : {
                    'length': bool(target_record['lane_length'].values[0]),
                    'type': bool(target_record['lane_type'].values[0])
                }
            }, 
            'road' : {
                'metric': {
                    'queue_length': bool(target_record['queue_length'].values[0]),
                    'delay': bool(target_record['delay'].values[0])
                }
            }
        }
        return
    
    def _getSymmetryPhaseTags(self):
        self.symmetry_phase_tags = {}
        for num_roads in [4]:
            self.symmetry_phase_tags[num_roads] = pd.read_csv(f'layout/symmetry_phase_tags{num_roads}.csv', index_col=False)
        
        return
