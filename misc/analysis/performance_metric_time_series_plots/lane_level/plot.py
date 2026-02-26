import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import re

from libs.figure_config import init_figure_config

# set config_yaml
config_file_path = root_dir_path / 'misc' / 'analysis' / 'performance_metric_time_series_plots' / 'lane_level' / 'config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f) 

# reflect figure configuration
init_figure_config()



