import shutil
from pathlib import Path
from datetime import datetime

src_dirs = [Path('buffers'), Path('models')]
backup_root = Path('backup')

for src_dir in src_dirs:
    # バックアップ先ディレクトリに日時付きのフォルダを作る
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"{src_dir.name}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    # ファイルをコピー
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, backup_dir / item.name)

    # 5個以上のバックアップがある場合は古いものを削除
    backup_dirs = sorted(backup_root.glob(f"{src_dir.name}_*"), key=lambda x: x.name)
    if len(backup_dirs) > 5:
        for old_backup in backup_dirs[:-5]:
            shutil.rmtree(old_backup)

    print(f"Backup of {src_dir} completed to {backup_dir}")
