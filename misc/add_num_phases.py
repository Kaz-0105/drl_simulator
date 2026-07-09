from pathlib import Path
import yaml

root_dir_path = (Path(__file__).parent / '..').resolve()


def main():
    performance_metrics_dir_path = root_dir_path / 'data' / 'performance_metrics'
    for drl_dir_path in performance_metrics_dir_path.rglob('drl'):
        for config_dir_path in drl_dir_path.glob('config_*'):
            with open(config_dir_path / 'config.yaml', 'r') as f:
                config_info = yaml.safe_load(f)

            if 'num_phases' in config_info:
                continue
            
            
            config_info['num_phases'] = 17
            with open(config_dir_path / 'config.yaml', 'w') as f:
                yaml.safe_dump(config_info, f)


if __name__ == '__main__':
    main()