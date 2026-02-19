import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import pandas as pd
import yaml
import re

from libs.figure_config import initFigureConfig

# reflect figure configuration
initFigureConfig()

# get config_yaml
config_file_path = root_dir_path / 'misc' / 'analysis' / 'performance_metric_time_series_plots' / 'road_level' / 'config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set data and performance_metrics directory paths
data_dir_path = root_dir_path / 'data'
performance_metrics_dir_path = data_dir_path / 'performance_metrics'
save_base_dir_path = data_dir_path / 'analysis' / 'performance_metric_time_series_plots' / 'road_level'

# get 

