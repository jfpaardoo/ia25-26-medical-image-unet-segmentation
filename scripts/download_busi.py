"""Download and organize the Breast Ultrasound Images (BUSI) dataset.

This script downloads the BUSI dataset from Kaggle and places the images
into ``data/raw/`` following the project's data contract.

Only **benign** and **malignant** classes are kept because the *normal*
class contains empty masks (all-black) which are not useful for
segmentation training.

When a sample has multiple mask files (e.g. ``_mask.png`` and
``_mask_1.png``), they are merged into a single binary mask via
logical-OR so that the full lesion extent is captured.

Usage
-----
    python scripts/download_busi.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "data" / "raw"

# Classes to keep (normal has empty masks → useless for segmentation)
CLASSES_TO_KEEP = ("benign", "malignant")


def _download_dataset() -> Path:
    """Download the BUSI dataset via kagglehub and return its local path."""
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed.  Run:\n"
            "    python -m pip install kagglehub\n"
            "then re-run this script."
        )

    print("Downloading BUSI dataset from Kaggle …")
    dataset_path = kagglehub.dataset_download(
        "aryashah2k/breast-ultrasound-images-dataset"
    )
    return Path(dataset_path)


def _find_busi_root(download_path: Path) -> Path:
    """Locate the actual folder that contains benign/ malignant/ normal/."""
    # kagglehub may nest the data one level deep
    for candidate in [download_path] + list(download_path.rglob("*")):
        if candidate.is_dir() and (candidate / "benign").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not find benign/ subdirectory inside {download_path}"
    )


def _merge_masks(mask_paths: list[Path]) -> np.ndarray:
    """Merge multiple mask files into a single binary mask (logical OR)."""
    merged = None
    for mp in mask_paths:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        binary = (m > 0).astype(np.uint8)
        if merged is None:
            merged = binary
        else:
            merged = np.maximum(merged, binary)
    if merged is None:
        raise ValueError(f"All mask files unreadable: {mask_paths}")
    return merged * 255  # store as 0/255 uint8 PNG


def _organize(busi_root: Path) -> dict[str, int]:
    """Copy and merge BUSI images+masks into data/raw/{images,masks}/."""

    # Clean previous raw data (keep the directory structure)
    for sub in ("images", "masks"):
        target = RAW_DIR / sub
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}

    for cls_name in CLASSES_TO_KEEP:
        cls_dir = busi_root / cls_name
        if not cls_dir.is_dir():
            print(f"[WARN] Class directory not found: {cls_dir}")
            continue

        # Collect all image files (not masks) in this class
        all_files = sorted(cls_dir.glob("*.png"))
        image_files = [
            f for f in all_files
            if "_mask" not in f.stem
        ]

        count = 0
        for img_file in image_files:
            stem = img_file.stem  # e.g. "benign (1)"

            # Find ALL mask files for this image
            mask_candidates = sorted(
                cls_dir.glob(f"{stem}_mask*.png")
            )
            if not mask_candidates:
                print(f"[WARN] No mask found for {img_file.name}, skipping")
                continue

            # Build a clean filename (replace spaces and parens)
            clean_name = (
                stem.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
            )
            # e.g. "benign_1"

            dst_image = RAW_DIR / "images" / f"{clean_name}.png"
            dst_mask = RAW_DIR / "masks" / f"{clean_name}.png"

            # Copy the original image
            shutil.copy2(img_file, dst_image)

            # Merge all masks and save
            merged = _merge_masks(mask_candidates)
            cv2.imwrite(str(dst_mask), merged)

            count += 1

        stats[cls_name] = count
        print(f"  {cls_name}: {count} image-mask pairs")

    return stats


def main() -> None:
    download_path = _download_dataset()
    busi_root = _find_busi_root(download_path)
    print(f"BUSI root found at: {busi_root}")

    stats = _organize(busi_root)
    total = sum(stats.values())

    print(f"\nTotal: {total} image-mask pairs copied to {RAW_DIR}")
    print("Structure:")
    print(f"  {RAW_DIR / 'images'}/  ({total} files)")
    print(f"  {RAW_DIR / 'masks'}/   ({total} files)")
    print("\nYou can now run:")
    print("    python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
