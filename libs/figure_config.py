import matplotlib as mpl
mpl.use('Agg')

from pathlib import Path
import yaml

def initFigureConfig():
    root_dir_path = (Path(__file__).parent / '..').resolve()
    config_path = root_dir_path / 'config' / 'figure_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        figure_info = yaml.safe_load(f)

    mpl.rcParams.update({
                # figure
                'figure.figsize': tuple(figure_info['size']),
                'figure.dpi': figure_info['dpi']['display'],
                'savefig.dpi': figure_info['dpi']['save'],

                # font
                'font.size': figure_info['font']['size'],
                'font.family': figure_info['font']['family'],
                'mathtext.default': figure_info['font']['mathtext'],

                # lines
                'lines.linewidth': figure_info['lines']['width'],
                'lines.markersize': figure_info['lines']['markersize'],

                # axes
                'axes.titlesize': figure_info['axes']['title']['size'],
                'axes.titleweight': figure_info['axes']['title']['weight'],
                'axes.labelsize': figure_info['axes']['label']['size'],
                'axes.labelweight': figure_info['axes']['label']['weight'],
                'axes.linewidth': figure_info['axes']['linewidth'],
                'axes.grid': figure_info['axes']['grid']['flg'],
                'grid.linestyle': figure_info['axes']['grid']['style'],
                'grid.alpha': figure_info['axes']['grid']['alpha'],
                'axes.spines.top': figure_info['axes']['spines']['top'],
                'axes.spines.right': figure_info['axes']['spines']['right'],
                
                # legend
                'legend.fontsize': figure_info['legend']['fontsize'],
                
                # ticks
                'xtick.labelsize': figure_info['ticks']['label']['size'],
                'xtick.direction': figure_info['ticks']['direction'],
                'xtick.major.width': figure_info['ticks']['major']['width'],
                'xtick.major.size': figure_info['ticks']['major']['size'],
                'xtick.minor.width': figure_info['ticks']['minor']['width'],
                'xtick.minor.size': figure_info['ticks']['minor']['size'],
                'ytick.labelsize': figure_info['ticks']['label']['size'],
                'ytick.direction': figure_info['ticks']['direction'],
                'ytick.major.width': figure_info['ticks']['major']['width'],
                'ytick.major.size': figure_info['ticks']['major']['size'],
                'ytick.minor.width': figure_info['ticks']['minor']['width'],
                'ytick.minor.size': figure_info['ticks']['minor']['size'],
            })
    
    return