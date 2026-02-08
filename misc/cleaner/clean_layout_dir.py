from pathlib import Path
import shutil

root_dir = (Path(__file__).parent / '..' / '..').resolve()
layout_dir_path = root_dir / 'layout'

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

    