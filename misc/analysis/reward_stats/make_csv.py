import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml

from libs.figure_config import init_figure_config


def main():
    # get config_info
    config_file_path = Path(__file__).parent / 'config.yaml'
    if not config_file_path.exists():
        raise NotImplementedError('No config file found.')
    with open(config_file_path, 'r') as f:
        config_info = yaml.safe_load(f)
    
    # get session_df
    session_df = getSessionDf(config_info)

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'reward_stats'
    save_dir_path.mkdir(parents=True, exist_ok=True)  

    # save reward_stats.csv
    makeRewardStatsCsv(config_info, session_df, save_dir_path)
    return

def getSessionDf(config_info):
    drl_dir_path = root_dir_path / 'data' / 'drl'
    if not drl_dir_path.exists():
        raise NotImplementedError('No drl data found.')
    
    found_flg = False
    for simulator_dir_path in drl_dir_path.glob('simulator_*'):
        with open(simulator_dir_path / 'config.yaml', 'r') as f:
            target_simulator_info = yaml.safe_load(f)

        if target_simulator_info == config_info['simulator']:
            found_flg = True
            break
    
    if not found_flg:
        raise NotImplementedError('No matching simulator config found.')
    
    found_flg = False
    for config_dir_path in simulator_dir_path.glob('config_*'):
        with open(config_dir_path / 'config.yaml', 'r') as f:
            target_config_info = yaml.safe_load(f)

        if target_config_info == config_info['drl']:
            found_flg = True
            break
    
    if not found_flg:
        raise NotImplementedError('No matching DRL config found.')
    
    session_file_path = config_dir_path / 'session' / 'session.csv'
    if not session_file_path.exists():
        raise NotImplementedError('No session file found.')
    session_df = pd.read_csv(session_file_path)
    return session_df

def makeRewardStatsCsv(config_info, session_df, save_dir_path):
    # get save_stat_list
    save_stat_list = [stat for stat, flag in config_info['target']['stat'].items() if flag]

    # get stats_df
    stats_df = session_df.groupby(['layout', 'inflow'])['total_reward'].agg(save_stat_list)
    stats_df = stats_df.reset_index()
    stats_df = stats_df.pivot(index='inflow', columns='layout', values=save_stat_list)
    stats_df.columns = [f"{stat}_{layout}" for stat, layout in stats_df.columns]
    stats_df = stats_df.reset_index()

    # save stats_df to csv
    save_file_path = save_dir_path / 'reward_stats.csv'
    stats_df.to_csv(save_file_path, index=False)
    return

if __name__ == "__main__":
    init_figure_config()
    main()
    print('Finished!')