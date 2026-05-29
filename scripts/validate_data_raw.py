import os
import sys
from pathlib import Path

import numpy as np

# ensure repo root is on sys.path so `src` package can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

from src.data.dataset import discover_samples


def validate_raw(root: Path) -> int:
    samples = discover_samples(root)
    print('Discovered pairs:')
    for s in samples:
        print(s.image_path.as_posix(), '->', s.mask_path.as_posix())
    print('Total raw pairs:', len(samples))
    return len(samples)


def validate_prepared(project_root: Path) -> None:
    splits_dir = project_root / 'data' / 'splits'
    split_files = ['train.txt', 'val.txt', 'test.txt']

    for split_name in split_files:
        split_path = splits_dir / split_name
        if not split_path.exists():
            print(f'Missing split file: {split_path.as_posix()}')
            continue

        lines = [line.strip() for line in split_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        print(f'{split_name}: {len(lines)} entries')

        for line in lines[:3]:
            image_rel, mask_rel = line.split(',', maxsplit=1)
            image_path = project_root / image_rel
            mask_path = project_root / mask_rel
            if not image_path.exists() or not mask_path.exists():
                raise RuntimeError(f'Broken entry in {split_name}: {line}')

            image = np.load(image_path)
            mask = np.load(mask_path)
            if image.shape[:2] != mask.shape[:2]:
                raise RuntimeError(f'Shape mismatch for entry in {split_name}: {line}')


if __name__ == '__main__':
    root = Path('data/raw')
    total = validate_raw(root)
    if total == 0:
        raise SystemExit('No raw pairs found. Add data into data/raw first.')

    validate_prepared(Path(repo_root))
    print('Validation completed successfully.')
