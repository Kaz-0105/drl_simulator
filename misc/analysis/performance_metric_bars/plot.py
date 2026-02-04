import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import copy
import re

from libs.figure_config import initFigureConfig


# configuration
initFigureConfig()
config_file_path = root_dir_path / 'misc' / 'analysis' / 'performance_metric_bars' / 'config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set wild_card_keys_list
def searchWildCardParams(config_yaml, wild_card_keys_list, key_list=[]):
    for key, value in config_yaml.items():
        tmp_keys = copy.deepcopy(key_list)
        tmp_keys.append(key)
        if isinstance(value, dict):
            searchWildCardParams(value, wild_card_keys_list, tmp_keys)
            continue

        if value is None:
            wild_card_keys_list.append(tmp_keys)
    return

wild_card_keys_list = []
searchWildCardParams(config_yaml, wild_card_keys_list)
if len(wild_card_keys_list) > 1:
    raise ValueError('Only one wild card parameter is allowed.')
elif len(wild_card_keys_list) == 0:
    wild_card_keys = None
else:
    wild_card_keys = wild_card_keys_list[0]

# set data_dir_path
data_dir_path = root_dir_path / 'data'

# make analysis directory if not exist
analysis_dir_path = data_dir_path / 'analysis'
analysis_dir_path.mkdir(exist_ok=True)

# set performance_metrics_dir_path
performance_metrics_dir_path = data_dir_path / 'performance_metrics'

# set target_dir_path_map
def getSubSetFlg(sub_config, main_config, wild_card_key_list=None, key_list=[]):
    for sub_key, sub_value in sub_config.items():
        tmp_keys = copy.deepcopy(key_list)
        tmp_keys.append(sub_key)

        main_value = main_config[sub_key]

        if isinstance(sub_value, dict):
            if getSubSetFlg(sub_value, main_value, wild_card_key_list, tmp_keys):
                continue
            else:
                return False

        if wild_card_key_list is not None and tmp_keys == wild_card_key_list:
            continue

        if sub_value != main_config[sub_key]:
            return False
        
    return True

# set target_dir_paths_map
target_dir_paths_map = {}
for simulator_dir_path in performance_metrics_dir_path.rglob('simulator_*'):
    keys = (
        simulator_dir_path.parts[-3], # layout_name
        simulator_dir_path.parts[-2], # inflow_name
        int(re.match(rf"simulator_(\d+)", simulator_dir_path.parts[-1]).group(1)), # simulator_id
    )
    path_info = {}
    match_config_count = 0
    for control_method in ['mpc', 'scoot']:
        if not config_yaml['comparison'][control_method]['flg']:
            continue
        
        path_info[control_method] = {}
        control_method_dir_path = simulator_dir_path / control_method
        for config_dir_path in control_method_dir_path.glob('config_*'):
            # set main_config_yaml and sub_config_yaml
            main_config_yaml = config_yaml[control_method]
            with open(config_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                sub_config_yaml = yaml.safe_load(f)
            
            # check if it is sub set or not
            if wild_card_keys is None or wild_card_keys[0] != control_method:
                sub_set_flg = getSubSetFlg(sub_config_yaml, main_config_yaml)
            else:
                sub_set_flg = getSubSetFlg(sub_config_yaml, main_config_yaml, wild_card_keys[1:])

            # skip if it is not sub set
            if not sub_set_flg:
                continue

            # increment match_config_count
            match_config_count += 1
            
            # push to path_info
            if wild_card_keys is None or wild_card_keys[0] != control_method:
                path_info[control_method] = config_dir_path
            else:
                # set wild_card_value
                wild_card_value = sub_config_yaml
                for wild_card_key in wild_card_keys[1:]:
                    wild_card_value = wild_card_value[wild_card_key]

                path_info[control_method][wild_card_value] = config_dir_path
                

    # skip if no matched configuration
    if match_config_count == 0:
        continue

    target_dir_paths_map[keys] = path_info      

# make figures

print('test')







