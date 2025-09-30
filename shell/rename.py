from pathlib import Path

target_dir = Path("backup/buffers_20250910_115946/apex/buffer_1")

if not target_dir.exists():
    raise FileNotFoundError(f"Directory not found: {target_dir}")

for file in target_dir.iterdir():
    stem_parts = file.stem.split('_')
    if stem_parts[2] == 'data':
        new_stem = '_'.join(stem_parts[:3] + [stem_parts[-1]])
    elif stem_parts[2] == 'tree':
        new_stem = '_'.join(stem_parts[:3])
    else:
        raise ValueError(f"Unexpected file name format: {file.name}")
    
    new_file = file.with_name(new_stem + file.suffix)
    file.rename(new_file)
    print(f"Renamed {file.name} to {new_file.name}")

   
