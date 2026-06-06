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
KAGGLE_DATASET_ID = "zionfuo/drive2004"


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


def _setup_directories(split_dir: Path, split_name: str) -> tuple[Path, Path, Path, Path | None]:
    images_dst = split_dir / "images"
    fov_dst = split_dir / "fov_masks"
    if split_name == "test":
        primary_dst = split_dir / "masks_expert1"
        secondary_dst = split_dir / "masks_expert2"
        dirs = [images_dst, fov_dst, primary_dst, secondary_dst]
    else:
        primary_dst = split_dir / "masks"
        secondary_dst = None
        dirs = [images_dst, fov_dst, primary_dst]

    for d in dirs:
        if d is not None:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    return images_dst, fov_dst, primary_dst, secondary_dst


def _validate_split(split_name: str, counts: dict, missing: dict) -> None:
    if counts["images"] != 20:
        raise RuntimeError(f"Expected 20 images in DRIVE {split_name}, found {counts['images']}")
    if counts["masks"] != 20:
        raise RuntimeError(f"Expected 20 primary masks in DRIVE {split_name}, found {counts['masks']}")
    if missing["primary"]:
        raise RuntimeError(f"Missing primary masks in DRIVE {split_name}: {', '.join(sorted(missing['primary']))}")
    if missing["fov"]:
        raise RuntimeError(f"Missing FOV masks in DRIVE {split_name}: {', '.join(sorted(missing['fov']))}")
    if split_name == "test" and missing["secondary"]:
        raise RuntimeError(f"Missing secondary masks in DRIVE {split_name}: {', '.join(sorted(missing['secondary']))}")


def _process_single_image(
    image_path: Path, prefix: str, split_name: str,
    src_dirs: dict[str, Path], dst_dirs: dict[str, Path | None],
    counts: dict[str, int], missing: dict[str, list[str]]
) -> None:
    shutil.copy2(image_path, dst_dirs["images"] / image_path.name) # type: ignore
    counts["images"] += 1

    mask_path = _find_drive_file(src_dirs["primary"], prefix, include=("manual1", "manual"))
    if mask_path is not None:
        _write_png(_binary_mask(mask_path), dst_dirs["primary"] / f"{image_path.stem}.png") # type: ignore
        counts["masks"] += 1
    else:
        missing["primary"].append(image_path.name)

    if split_name == "test" and dst_dirs.get("secondary"):
        mask2_path = _find_drive_file(src_dirs["secondary"], prefix, include=("manual2", "manual"))
        if mask2_path is not None:
            _write_png(_binary_mask(mask2_path), dst_dirs["secondary"] / f"{image_path.stem}.png")
        else:
            missing["secondary"].append(image_path.name)

    fov_path = _find_drive_file(src_dirs["fov"], prefix, include=("mask",))
    if fov_path is not None:
        _write_png(_binary_mask(fov_path), dst_dirs["fov"] / f"{image_path.stem}.png") # type: ignore
    else:
        missing["fov"].append(image_path.name)


def _organize_split(split_root: Path, split_name: str) -> dict[str, int]:
    src_dirs = {
        "images": split_root / "images",
        "primary": split_root / "1st_manual",
        "secondary": split_root / "2nd_manual",
        "fov": split_root / "mask",
    }
    
    split_dir = RAW_DIR / split_name
    images_dst, fov_dst, primary_dst, secondary_dst = _setup_directories(split_dir, split_name)
    dst_dirs: dict[str, Path | None] = {
        "images": images_dst,
        "primary": primary_dst,
        "secondary": secondary_dst,
        "fov": fov_dst,
    }

    counts = {"images": 0, "masks": 0}
    missing: dict[str, list[str]] = {"primary": [], "fov": [], "secondary": []}

    for image_path in sorted(src_dirs["images"].rglob("*")):
        if not image_path.is_file():
            continue
        prefix = _drive_prefix(image_path)
        if prefix is not None:
            _process_single_image(image_path, prefix, split_name, src_dirs, dst_dirs, counts, missing)

    _validate_split(split_name, counts, missing)
    return counts


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
    print(f"  {RAW_DIR / 'test' / 'masks_expert1'}/")
    print(f"  {RAW_DIR / 'test' / 'masks_expert2'}/")
    print("\nYou can now run:")
    print("    python scripts/prepare_data.py")


if __name__ == "__main__":
    main()