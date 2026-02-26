import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

from libs.figure_config import init_figure_config

# load config.yaml
with open(root_dir_path / 'misc' / 'analysis' / 'calc_time_cdf_plots' / 'config.yaml', 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# set figure configuration
init_figure_config()

data_dir = root_dir_path / 'data'
performance_metrics_dir = data_dir / 'performance_metrics'
analysis_dir_path = data_dir / 'analysis'

# make calc_time_df
calc_time_df = None

for inflow_name, inflow_label in config_yaml['target']['inflows'].items():
    inflow_dir_path = performance_metrics_dir / config_yaml['target']['layout'] / inflow_name

    for simulator_dir_path in inflow_dir_path.glob('simulator_*'):
        with open(simulator_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
            simulator_config = yaml.safe_load(f)

        # if seed is fixed in the plot config, check it
        if config_yaml['target']['seed']['fix_flg'] and simulator_config['seed'] != config_yaml['target']['seed']['fix_value']:
            continue
        del simulator_config['seed'] 

        # check simulator config matches the plot config
        if simulator_config != config_yaml['simulator']:
            continue
        
        # regarding mpc
        mpc_dir_path = simulator_dir_path / 'mpc'
        if not mpc_dir_path.exists():
            continue

        for method_dir_path in mpc_dir_path.glob('config_*'):
            with open(method_dir_path / 'config.yaml', 'r', encoding='utf-8') as f:
                method_config = yaml.safe_load(f)

            # check num_phases
            num_phases = method_config['phases']['4-road']
            if not config_yaml['target']['control_method']['mpc'][f"{num_phases}-phase"]:
                continue 
            del method_config['phases']

            config_yaml['mpc']['objective_function']['signal_change']['weight'] = config_yaml['target']['signal_change_weight'][f"{num_phases}-phase"]

            if method_config != config_yaml['mpc']:
                continue

            for intersection_dir_path in method_dir_path.glob('intersection_*'):
                with open(intersection_dir_path / 'calc_time.csv', 'r', encoding='utf-8') as f:
                    tmp_calc_time_df = pd.read_csv(f).rename(columns={'value': 'calc_time'}).drop(columns=['time'])
                    tmp_calc_time_df['inflow'] = inflow_label
                    tmp_calc_time_df['num_phases'] = num_phases

                if calc_time_df is None:
                    calc_time_df = tmp_calc_time_df.copy()
                else:
                    calc_time_df = pd.concat([calc_time_df, tmp_calc_time_df], ignore_index=True)

# make save_dir_path
save_dir_path = analysis_dir_path / 'calc_time_cdf_plots' / config_yaml['target']['layout']
save_dir_path.mkdir(parents=True, exist_ok=True)

# make plot
max_time = calc_time_df['calc_time'].max()
fig, ax = plt.subplots()

sns.kdeplot(
    ax=ax,
    data=calc_time_df,
    x='calc_time',
    hue='inflow',
    cumulative=True,
    common_norm=False,
    palette=config_yaml['figure']['palette']
)

ax.set_title(config_yaml['figure']['title']['inflow'])
ax.set_xlabel(config_yaml['figure']['x_axis']['label'])
ax.set_xlim(0, max_time)
ax.set_ylabel(config_yaml['figure']['y_axis']['label'])
sns.move_legend(ax, loc='lower right', title=config_yaml['figure']['legend']['title']['inflow'], ncol=len(config_yaml['target']['inflows'])/2)
fig.tight_layout()
fig.savefig(save_dir_path / 'calc_time_cdf_plot_inflow.png')
plt.close('all')

fig, ax = plt.subplots()

sns.kdeplot(
    ax=ax,
    data=calc_time_df,
    x='calc_time',
    hue='num_phases',
    cumulative=True,
    common_norm=False,
    palette=config_yaml['figure']['palette']
)

ax.set_title(config_yaml['figure']['title']['num_phases'])
ax.set_xlabel(config_yaml['figure']['x_axis']['label'])
ax.set_xlim(0, max_time)
ax.set_ylabel(config_yaml['figure']['y_axis']['label'])
sns.move_legend(ax, loc='lower right', title=config_yaml['figure']['legend']['title']['num_phases'])
fig.tight_layout()
fig.savefig(save_dir_path / 'calc_time_cdf_plot_num_phases.png')
plt.close('all')


print('Finished!')
        

