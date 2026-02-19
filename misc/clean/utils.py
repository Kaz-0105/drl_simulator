from pathlib import Path
import shutil
from itertools import chain
import re

root_dir_path = (Path(__file__).parent / '..' / '..').resolve()

def clean_performance_metric_dir():
    data_dir_path  = root_dir_path / 'data'

    # performance_metrics directory cleanup
    performance_metrics_dir_path = data_dir_path / 'performance_metrics'
    for tmp_config_dir_path in performance_metrics_dir_path.rglob('config_*'):
        intersection_dir_path_list = list(tmp_config_dir_path.glob('intersection_*'))
        if len(intersection_dir_path_list) == 0:
            shutil.rmtree(tmp_config_dir_path)

    # make config ids to be continuous
    for target_dir_path in chain(performance_metrics_dir_path.rglob('mpc'), performance_metrics_dir_path.rglob('scoot')):
        config_dir_path_map = {}
        for tmp_config_dir_path in target_dir_path.glob('config_*'):
            config_dir_path_map[int(re.match(rf"config_(\d+)", tmp_config_dir_path.name).group(1))] = tmp_config_dir_path
        
        new_config_id = 1
        for old_config_id in sorted(config_dir_path_map.keys()):
            if old_config_id == new_config_id:
                new_config_id += 1
                continue

            old_config_dir_path = config_dir_path_map[old_config_id]
            new_config_dir_path = target_dir_path / f'config_{new_config_id}'
            old_config_dir_path.rename(new_config_dir_path)
            new_config_id += 1

def clean_layout_dir():
    layout_dir_path = root_dir_path / 'layout'

    for tmp_layout_dir in layout_dir_path.iterdir():
        if not tmp_layout_dir.is_dir():
            continue

        # delete *.err files
        for err_file in tmp_layout_dir.glob('*.err'):
            try:
                err_file.unlink()
            except Exception as e:
                print(f"Exception : {err_file}")
                print(f"Reason : {e}")

        # delete .results folders
        for results_dir in tmp_layout_dir.glob('*.results'):
            if results_dir.is_dir():
                shutil.rmtree(results_dir)
        
        # delete .avi files
        for avi_file in tmp_layout_dir.glob('*.avi'):
            avi_file.unlink()


if __name__ == '__main__':
    clean_performance_metric_dir()
    clean_layout_dir()

        
        