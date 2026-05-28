from pathlib import Path
import sys
import os

# ensure repo root is on sys.path so `src` package can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

from src.data.dataset import discover_samples

root = Path('data/raw')
samples = discover_samples(root)
print('Discovered pairs:')
for s in samples:
    print(s.image_path.as_posix(), '->', s.mask_path.as_posix())
print('Total:', len(samples))
