import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import re

from libs.figure_config import init_figure_config

init_figure_config()


def main():
    # get config_info
    config_file_path = Path(__file__).parent / 'config.yaml'
    with open(config_file_path, 'r') as f:
        config_info = yaml.safe_load(f)

    # get session_df
    session_df = getSessionDf(config_info)

    # get save_dir_path
    save_dir_path = root_dir_path / 'data' / 'analysis' / 'reward_transition'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # plot reward transition
    plotRewardTransition(
        figure_info=config_info['figure'], 
        session_df=session_df, 
        save_dir_path=save_dir_path
    )
    return

def getSessionDf(config_info):
    drl_dir_path = root_dir_path / 'data' / 'drl'
    if not drl_dir_path.exists():
        raise NotImplementedError('No drl data found.')
    
    found_flg = False
    for config_dir_path in drl_dir_path.glob('config_*'):
        with open(config_dir_path / 'config.yaml', 'r') as f:
            target_config_info = yaml.safe_load(f)

        if target_config_info == config_info['drl']:
            found_flg = True
            break
    
    if not found_flg:
        raise NotImplementedError('No matching config found.')
    
    session_file_path = config_dir_path / 'session' / 'session.csv'
    if not session_file_path.exists():
        raise NotImplementedError('No session file found.')
    session_df = pd.read_csv(session_file_path)
    return session_df

def plotRewardTransition(figure_info, session_df, save_dir_path):
    session_df['inflow_label'] = session_df['inflow'].apply(lambda x: figure_info['legend']['labels'][x])
    session_df['inflow_label'] = pd.Categorical(session_df['inflow_label'], categories=figure_info['legend']['labels'].values(), ordered=True)
    session_df['rel_episode'] = session_df.groupby(['layout', 'inflow']).cumcount() + 1
    

    y_min = session_df['total_reward'].min()
    y_max = session_df['total_reward'].max()
    x_min = session_df['rel_episode'].min()
    x_max = session_df['rel_episode'].max()

    for layout in session_df['layout'].unique():
        target_session_df = session_df[session_df['layout'] == layout]

        fig, ax = plt.subplots()
        sns.lineplot(
            data=target_session_df,
            x='rel_episode',
            y='total_reward',
            hue='inflow_label',
            ax=ax,
            marker='o',
        )
        ax.set_title(figure_info['title'][layout])
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

        plt.savefig(save_dir_path / f'reward_transition_{layout}.png')
    return

if __name__ == "__main__":
    main()
