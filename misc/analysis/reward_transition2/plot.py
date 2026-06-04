import sys
from pathlib import Path
from turtle import position
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import copy

from libs.figure_config import init_figure_config

def main():
    # reflect figure configuration
    init_figure_config()
    
    # get config_info
    config_file_path = Path(__file__).parent / 'config.yaml'
    with open(config_file_path, 'r') as f:
        config_info = yaml.safe_load(f)

    # get session_df
    session_df_map = getSessionDfMap(config_info)

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'reward_transition2'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # plot reward transition
    plotRewardTransition(
        figure_info=config_info['figure'], 
        session_df_map=session_df_map, 
        save_dir_path=save_dir_path
    )
    return

def getSessionDfMap(config_info):
    session_df_map = {}
    for method, method_flg in config_info['target']['method'].items():
        if not method_flg:
            continue

        session_df_map[method] = getSessionDf(copy.deepcopy(config_info), method)
    return session_df_map

def getSessionDf(config_info, method):
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
        
        if method == 'macro':
            if target_config_info['state']['vehicle']['number'] != config_info['drl']['state']['vehicle']['number']:
                continue
            number = target_config_info['state']['vehicle'].pop('number')
                    
            if any(flg == True for flg in target_config_info['state']['vehicle'].values()):
                continue

            target_config_info['state']['vehicle'] = {}
            target_config_info['state']['vehicle']['number'] = number
        
        elif method == 'micro':
            if target_config_info['state']['vehicle']['number'] != config_info['drl']['state']['vehicle']['number']:
                continue
            number = target_config_info['state']['vehicle'].pop('number')
            
            if any(flg == False for flg in target_config_info['state']['vehicle'].values()):
                continue

            target_config_info['state']['vehicle'] = {}
            target_config_info['state']['vehicle']['number'] = number

        else:
            raise NotImplementedError(f'Unsupported method: {method}')

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

def plotRewardTransition(figure_info, session_df_map, save_dir_path):
    session_df = None
    for method, tmp_session_df in session_df_map.items():
        tmp_session_df['method'] = method
        tmp_session_df['method_label'] = figure_info['legend']['labels'][method]
        session_df = pd.concat([session_df, tmp_session_df], ignore_index=True) if session_df is not None else tmp_session_df
        
    y_min = session_df['total_reward'].min()
    y_max = session_df['total_reward'].max()
    x_min = session_df['episode'].min()
    x_max = session_df['episode'].max()

    fig, ax = plt.subplots()
    sns.lineplot(
        data=session_df,
        x='episode',
        y='total_reward',
        hue='method_label',
        ax=ax,
        marker='o',
    )
    ax.set_title(figure_info['title'])
    ax.set_xlabel(figure_info['x_axis']['label'])
    ax.set_ylabel(figure_info['y_axis']['label'])
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 5, y_max + 5)

    ax.legend(
        title=figure_info['legend']['title'],
        ncol=figure_info['legend']['ncol'],
        loc=figure_info['legend']['loc'],
    )
    fig.tight_layout()
    plt.savefig(save_dir_path / f'reward_transition.png')
    return

if __name__ == "__main__":
    main()
