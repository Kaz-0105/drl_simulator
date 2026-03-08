import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import yaml
import numpy as np
import pandas as pd

from libs.figure_config import init_figure_config

# load config.yaml
with open(root_dir_path / 'misc' / 'analysis' / 'multi_intersection_input_calculation' / 'config.yaml', 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)


system_info = config_yaml['system']
input_info = config_yaml['input']
route_selection_info = config_yaml['route_selection']


if system_info['row'] == 3 and system_info['column'] == 3:
    roads_dir_path = root_dir_path / 'misc' / 'analysis' / 'multi_intersection_input_calculation' / '3-3' / 'roads.csv'
    if not roads_dir_path.exists():
        raise FileNotFoundError(f'{roads_dir_path} does not exist.')
    with open(roads_dir_path, 'r', encoding='utf-8') as f:
        roads_df = pd.read_csv(f)
    
    num_variables = (system_info['row'] + 1) * system_info['column'] * 2 + (system_info['column'] + 1) * system_info['row'] * 2
    A_matrix = np.zeros((num_variables, num_variables))

    for intersection_id in range(1, system_info['row'] * system_info['column'] + 1):
        input_roads_df = roads_df[roads_df['intersection'] == intersection_id & roads_df['type'] == 'input'].copy()
        output_roads_df = roads_df[roads_df['intersection'] == intersection_id & roads_df['type'] == 'output'].copy()

        for direction in ['north', 'south', 'east', 'west']:
            tmp_output_road_df = output_roads_df[output_roads_df['direction'] == direction]
            tmp_input_roads_df = input_roads_df[input_roads_df['direction'] != direction]

            

            

else:
    raise NotImplementedError('Only 3x3 system is implemented.')