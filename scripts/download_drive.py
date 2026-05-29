"""Download and organize the DRIVE 2004 dataset.

This script downloads the DRIVE (Digital Retinal Images for Vessel Extraction)
dataset and places the files into ``data/raw/`` following the project's
data contract.

The official dataset contains 40 fundus images split into 20 training and
20 test images. The script keeps the official split and stores the primary
manual segmentation as the ground-truth mask.

Usage
-----
    python scripts/download_drive.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "data" / "raw"
KAGGLE_DATASET_ID = "andrewmvd/drive-digital-retinal-images-for-vessel-extraction"


def _download_dataset() -> Path:
    """Download DRIVE 2004 via kagglehub and return its local path."""
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Run:\n"
            "    python -m pip install kagglehub\n"
            "and then re-run this script."
        )

    print("Downloading DRIVE 2004 dataset from Kaggle …")
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    return Path(dataset_path)


def _find_drive_root(download_path: Path) -> Path:
    """Locate the directory that contains the official training/test folders."""
    candidates = [download_path] + list(download_path.rglob("*"))
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "training").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise RuntimeError(f"Could not find training/ and test/ directories inside {download_path}")


def _read_any_image(path: Path) -> np.ndarray:
    """Read an image file with a fallback through Pillow when OpenCV is unavailable."""
    array = np.array(Image.open(path))
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :4]
    return array


def _write_png(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array)).save(destination)


def _binary_mask(path: Path) -> np.ndarray:
    array = _read_any_image(path)
    if array.ndim == 3:
        array = array.max(axis=-1)
    return (array > 0).astype(np.uint8) * 255


def _drive_prefix(path: Path) -> str | None:
    match = re.match(r"^(\d+)", path.stem)
    if match is None:
        return None
    return match.group(1)


def _find_drive_file(directory: Path, prefix: str, *, include: tuple[str, ...] = ()) -> Path | None:
    if not directory.is_dir():
        return None

    candidates: list[Path] = []
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        if _drive_prefix(file_path) != prefix:
            continue
        lowered = file_path.stem.lower()
        if include and not any(token in lowered for token in include):
            continue
        candidates.append(file_path)

    if not candidates:
        return None
    return candidates[0]


def _organize_split(split_root: Path, split_name: str) -> dict[str, int]:
    images_src = split_root / "images"
    primary_masks_src = split_root / "1st_manual"
    fov_src = split_root / "mask"
    split_dir = RAW_DIR / split_name
    images_dst = split_dir / "images"
    masks_dst = split_dir / "masks"
    fov_dst = split_dir / "fov_masks"

    for directory in (images_dst, masks_dst, fov_dst):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    image_count = 0
    mask_count = 0
    missing_primary_masks: list[str] = []
    missing_fov_masks: list[str] = []

    for image_path in sorted(images_src.rglob("*")):
        if not image_path.is_file():
            continue
        prefix = _drive_prefix(image_path)
        if prefix is None:
            continue
        target = images_dst / image_path.name
        shutil.copy2(image_path, target)
        image_count += 1

        mask_path = _find_drive_file(primary_masks_src, prefix, include=("manual1", "manual"))
        if mask_path is not None:
            _write_png(_binary_mask(mask_path), masks_dst / f"{image_path.stem}.png")
            mask_count += 1
        else:
            if split_name == "training":
                missing_primary_masks.append(image_path.name)

        fov_path = _find_drive_file(fov_src, prefix, include=("mask",))
        if fov_path is not None:
            _write_png(_binary_mask(fov_path), fov_dst / f"{image_path.stem}.png")
        else:
            missing_fov_masks.append(image_path.name)

    if image_count != 20:
        raise RuntimeError(f"Expected 20 images in DRIVE {split_name}, found {image_count}")
    if split_name == "training" and mask_count != 20:
        raise RuntimeError(f"Expected 20 primary masks in DRIVE {split_name}, found {mask_count}")
    if missing_primary_masks:
        raise RuntimeError(
            f"Missing primary masks in DRIVE {split_name}: {', '.join(sorted(missing_primary_masks))}"
        )
    if missing_fov_masks:
        raise RuntimeError(
            f"Missing FOV masks in DRIVE {split_name}: {', '.join(sorted(missing_fov_masks))}"
        )

    return {"images": image_count, "masks": mask_count}


def main() -> None:
    download_path = _download_dataset()
    drive_root = _find_drive_root(download_path)
    print(f"DRIVE root found at: {drive_root}")

    for subdir in (RAW_DIR / "training", RAW_DIR / "test"):
        if subdir.exists():
            shutil.rmtree(subdir)

    training_stats = _organize_split(drive_root / "training", "training")
    test_stats = _organize_split(drive_root / "test", "test")

    total_images = training_stats["images"] + test_stats["images"]
    total_masks = training_stats["masks"] + test_stats["masks"]

    print(f"\nTotal: {total_images} images and {total_masks} primary masks copied to {RAW_DIR}")
    print("Structure:")
    print(f"  {RAW_DIR / 'training' / 'images'}/")
    print(f"  {RAW_DIR / 'training' / 'masks'}/")
    print(f"  {RAW_DIR / 'training' / 'fov_masks'}/")
    print(f"  {RAW_DIR / 'test' / 'images'}/")
    print(f"  {RAW_DIR / 'test' / 'fov_masks'}/")
    print("\nYou can now run:")
    print("    python scripts/prepare_data.py")


if __name__ == "__main__":
    main()