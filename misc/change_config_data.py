from pathlib import Path
import yaml
import shutil

root_dir_path = (Path(__file__).parent / '..').resolve()


def main():
    performance_metrics_dir_path = root_dir_path / 'data' / 'performance_metrics'
    for drl_dir_path in performance_metrics_dir_path.rglob('drl'):
        for config_dir_path in drl_dir_path.glob('config_*'):
            with open(config_dir_path / 'config.yaml', 'r') as f:
                config_info = yaml.safe_load(f)

            if not isinstance(config_info['reward']['waiting_vehicles']['bonus'], dict):
                shutil.rmtree(config_dir_path)
                continue
            if 'spillback' in config_info['reward']['waiting_vehicles']['bonus'].keys(): continue
            
            config_info['reward']['waiting_vehicles']['bonus']['spillback'] = 0.0
            with open(config_dir_path / 'config.yaml', 'w') as f:
                yaml.safe_dump(config_info, f)

    drl_dir_path = root_dir_path / 'data' / 'drl'
    for config_dir_path in drl_dir_path.rglob('config_*'):
        with open(config_dir_path / 'config.yaml', 'r') as f:
            config_info = yaml.safe_load(f)

        if 'spillback' in config_info['reward']['waiting_vehicles']['bonus'].keys(): continue

        config_info['reward']['waiting_vehicles']['bonus']['spillback'] = 0.0
        with open(config_dir_path / 'config.yaml', 'w') as f:
            yaml.safe_dump(config_info, f)


if __name__ == '__main__':
    main()