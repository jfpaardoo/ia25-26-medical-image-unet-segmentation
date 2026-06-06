"""Prepare raw medical images for training."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Ensure repo root is importable when running as: python scripts/prepare_data.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import SegmentationSample, discover_samples
from src.data.patching import extract_patches
from src.data.preprocessing import normalize_image, resize_pair


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_pair(sample: SegmentationSample) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Could not read image file: {sample.image_path}")
    if mask is None:
        raise ValueError(f"Could not read mask file: {sample.mask_path}")

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Shape mismatch for pair {sample.image_path} and {sample.mask_path}: "
            f"{image.shape[:2]} vs {mask.shape[:2]}"
        )

    return image, mask


def _augmentations(enabled: bool, names: list[str], rng: np.random.Generator) -> list[tuple[str, callable]]:
    base = [("orig", lambda image, mask: (image, mask))]
    if not enabled:
        return base

    supported = {
        "hflip": lambda image, mask: (np.fliplr(image).copy(), np.fliplr(mask).copy()),
        "vflip": lambda image, mask: (np.flipud(image).copy(), np.flipud(mask).copy()),
        "rot90": lambda image, mask: (np.rot90(image, 1).copy(), np.rot90(mask, 1).copy()),
        "noise": lambda image, mask: (np.clip(image + rng.normal(0, 0.05, image.shape), 0, 1).astype(np.float32), mask.copy()),
        "contrast": lambda image, mask: (np.clip(image * 1.2, 0, 1).astype(np.float32), mask.copy()),
    }

    out = list(base)
    unknown: list[str] = []
    for name in names:
        key = str(name).strip().lower()
        if key in supported:
            out.append((key, supported[key]))
        else:
            unknown.append(str(name))
    if unknown:
        raise ValueError(f"Unsupported augmentation transform(s): {', '.join(unknown)}")
    return out


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _infer_drive_split(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "training" in parts or "train" in parts:
        return "train"
    if "test" in parts or "testing" in parts:
        return "test"
    return None


def _write_split(split_path: Path, items: list[dict]) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item['image_path']},{item['mask_path']}\n")


def _split_source_ids(source_ids: list[str], ratios: dict, rng: np.random.Generator) -> dict[str, set[str]]:
    train_ratio = float(ratios.get("train", 0.7))
    val_ratio = float(ratios.get("val", 0.15))
    test_ratio = float(ratios.get("test", 0.15))
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    arr = np.array(sorted(set(source_ids)), dtype=object)
    rng.shuffle(arr)
    n = len(arr)
    n_train = int(round(n * (train_ratio / total)))
    n_val = int(round(n * (val_ratio / total)))

    if n_train > n:
        n_train = n
    if n_train + n_val > n:
        n_val = n - n_train

    n_test = n - n_train - n_val

    train_ids = set(arr[:n_train].tolist())
    val_ids = set(arr[n_train : n_train + n_val].tolist())
    test_ids = set(arr[n_train + n_val : n_train + n_val + n_test].tolist())

    if n > 0 and not train_ids:
        first = arr[0]
        train_ids.add(first)
        val_ids.discard(first)
        test_ids.discard(first)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _split_official_drive_ids(source_ids: list[str], ratios: dict, rng: np.random.Generator) -> dict[str, set[str]]:
    train_ratio = float(ratios.get("train", 0.8))
    val_ratio = float(ratios.get("val", 0.2))
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    total = train_ratio + val_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    arr = np.array(sorted(set(source_ids)), dtype=object)
    rng.shuffle(arr)
    n = len(arr)
    n_train = int(round(n * (train_ratio / total)))
    if n_train > n:
        n_train = n
    n_val = n - n_train

    train_ids = set(arr[:n_train].tolist())
    val_ids = set(arr[n_train : n_train + n_val].tolist())
    return {"train": train_ids, "val": val_ids, "test": set()}


def _validate_split_file(path: Path, name: str, project_root: Path, ph: int, pw: int) -> None:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        if name not in {"test", "val"}:
            raise RuntimeError(f"Split file is empty: {path}")
        return

    for line in lines:
        image_rel, mask_rel = line.split(",", maxsplit=1)
        image_path = project_root / image_rel
        mask_path = project_root / mask_rel
        if not image_path.exists() or not mask_path.exists():
            raise RuntimeError(f"Split entry points to missing file: {line}")

        image = np.load(image_path)
        mask = np.load(mask_path)
        if image.shape[0] != ph or image.shape[1] != pw:
            raise RuntimeError(f"Invalid image patch shape in {image_path}: {image.shape}")
        if mask.shape[0] != ph or mask.shape[1] != pw:
            raise RuntimeError(f"Invalid mask patch shape in {mask_path}: {mask.shape}")
        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"Image/mask patch mismatch in {line}")


def _validate_prepared_dataset(project_root: Path, processed_dir: Path, splits_dir: Path, patch_size: tuple[int, int]) -> None:
    images_dir = processed_dir / "images"
    masks_dir = processed_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise RuntimeError("Processed dataset missing images/ or masks/ directory")

    split_files = {
        "train": splits_dir / "train.txt",
        "val": splits_dir / "val.txt",
        "test": splits_dir / "test.txt",
    }
    for name, path in split_files.items():
        if not path.exists():
            raise RuntimeError(f"Missing split file: {path}")

    ph, pw = patch_size
    for name, path in split_files.items():
        _validate_split_file(path, name, project_root, ph, pw)


def _save_patch_if_valid(
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    patch_id: str,
    source_id: str,
    aug_name: str,
    official_split: str | None,
    mask_dir_name: str,
    processed_dir: Path,
    project_root: Path,
    min_foreground: float,
) -> dict | None:
    mask_arr = mask_patch
    if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
        mask_arr = mask_arr[..., 0]
    fg_ratio = float((mask_arr > 0).sum()) / float(mask_arr.shape[0] * mask_arr.shape[1])
    if fg_ratio < min_foreground:
        return None

    image_path = processed_dir / "images" / f"{patch_id}.npy"
    mask_path = processed_dir / mask_dir_name / f"{patch_id}.npy"

    mask_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        np.save(image_path, image_patch.astype(np.float32))
    np.save(mask_path, mask_patch.astype(np.uint8))

    return {
        "source_id": source_id,
        "augmentation": aug_name,
        "official_split": official_split,
        "mask_dir_name": mask_dir_name,
        "image_path": _relative(image_path, project_root),
        "mask_path": _relative(mask_path, project_root),
    }

def _process_single_sample(
    sample: SegmentationSample,
    raw_dir: Path,
    processed_dir: Path,
    project_root: Path,
    transforms: list[tuple[str, callable]],
    patch_size: tuple[int, int],
    patch_stride: tuple[int, int],
    mask_format: str,
    target_channels: int,
    min_foreground: float,
    use_official_drive_split: bool,
) -> list[dict]:
    try:
        image_raw, mask_raw = _read_pair(sample)
    except ValueError as exc:
        print(f"[WARN] Skipping sample: {exc}")
        return []

    mask_dir_name = sample.mask_path.parent.name

    image_resized, mask_resized = resize_pair(
        image_raw,
        mask_raw,
        target_size=image_raw.shape[:2],
        mask_format=mask_format,
        target_channels=target_channels,
    )
    image_resized = normalize_image(image_resized)
    if image_resized.ndim == 2:
        image_resized = image_resized[..., None]

    sample_rel = sample.image_path.relative_to(raw_dir)
    source_id = sample_rel.with_suffix("").as_posix().replace("/", "__")
    official_split = _infer_drive_split(sample.image_path) if use_official_drive_split else None

    sample_transforms = transforms if official_split != "test" else [transforms[0]]
    records = []

    for aug_name, aug_fn in sample_transforms:
        image_aug, mask_aug = aug_fn(image_resized, mask_resized)
        image_patches, mask_patches, _, _ = extract_patches(
            image_aug,
            mask_aug,
            patch_size=patch_size,
            stride=patch_stride,
        )

        for idx in range(image_patches.shape[0]):
            patch_id = f"{source_id}__{aug_name}__p{idx:04d}"
            record = _save_patch_if_valid(
                image_patch=image_patches[idx],
                mask_patch=mask_patches[idx],
                patch_id=patch_id,
                source_id=source_id,
                aug_name=aug_name,
                official_split=official_split,
                mask_dir_name=mask_dir_name,
                processed_dir=processed_dir,
                project_root=project_root,
                min_foreground=min_foreground,
            )
            if record:
                records.append(record)

    return records


def _create_splits(records: list[dict], split_cfg: dict, rng: np.random.Generator, use_official: bool) -> dict[str, list[dict]]:
    if use_official and any(r["official_split"] in {"train", "test"} for r in records):
        train_source_ids = [r["source_id"] for r in records if r["official_split"] == "train"]
        test_source_ids = [r["source_id"] for r in records if r["official_split"] == "test"]
        split_ids = _split_official_drive_ids(train_source_ids, split_cfg, rng)
        return {
            "train": [r for r in records if r["official_split"] == "train" and r["source_id"] in split_ids["train"]],
            "val": [r for r in records if r["official_split"] == "train" and r["source_id"] in split_ids["val"]],
            "test": [r for r in records if r["official_split"] == "test" and r["source_id"] in test_source_ids],
        }
    else:
        split_ids = _split_source_ids([r["source_id"] for r in records], split_cfg, rng)
        return {
            "train": [r for r in records if r["source_id"] in split_ids["train"]],
            "val": [r for r in records if r["source_id"] in split_ids["val"]],
            "test": [r for r in records if r["source_id"] in split_ids["test"]],
        }


def run_preparation(config_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = _load_config(config_path)

    seed = int(cfg.get("project", {}).get("seed", 394867))
    rng = np.random.default_rng(seed)

    data_cfg = cfg.get("data", {})
    raw_dir = project_root / str(data_cfg.get("raw_dir", "data/raw"))
    processed_dir = project_root / str(data_cfg.get("processed_dir", "data/processed"))
    splits_dir = project_root / str(data_cfg.get("splits_dir", "data/splits"))

    configured_image_size = data_cfg.get("image_size", [256, 256])
    patch_size = data_cfg.get("patch_size", [256, 256])
    patch_stride = data_cfg.get("patch_stride", patch_size)
    mask_format = str(data_cfg.get("mask_format", "binary"))
    min_foreground = float(data_cfg.get("patch_min_foreground", 0.0))

    model_cfg = cfg.get("model", {})
    target_channels = int(model_cfg.get("input_channels", 1))

    aug_cfg = data_cfg.get("augmentation", {})
    aug_enabled = bool(aug_cfg.get("enabled", True))
    aug_names = list(aug_cfg.get("transforms", ["hflip", "vflip", "rot90"]))
    split_cfg = data_cfg.get("split", {"train": 0.7, "val": 0.15, "test": 0.15})
    use_official_drive_split = bool(data_cfg.get("use_official_drive_split", True))

    samples = discover_samples(raw_dir)
    if not samples:
        raise RuntimeError(
            f"No image-mask pairs found under {raw_dir}. "
            "Run `python scripts/download_drive.py` first and make sure the DRIVE files are in data/raw."
        )

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    (processed_dir / "images").mkdir(parents=True, exist_ok=True)
    (processed_dir / "masks").mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    transforms = _augmentations(enabled=aug_enabled, names=aug_names, rng=rng)
    records: list[dict] = []

    skipped_samples = 0

    for sample in samples:
        sample_records = _process_single_sample(
            sample, raw_dir, processed_dir, project_root,
            transforms, patch_size, patch_stride, mask_format,
            target_channels, min_foreground, use_official_drive_split
        )
        if not sample_records:
            skipped_samples += 1
        else:
            records.extend(sample_records)

    if not records:
        raise RuntimeError(
            "No processed patches were generated. Check that data/raw contains readable image-mask pairs."
        )

    split_records = _create_splits(records, split_cfg, rng, use_official_drive_split)

    _write_split(splits_dir / "train.txt", split_records["train"])
    _write_split(splits_dir / "val.txt", split_records["val"])
    _write_split(splits_dir / "test.txt", split_records["test"])

    summary = {
        "num_raw_samples": len(samples),
        "num_skipped_samples": skipped_samples,
        "num_processed_patches": len(records),
        "num_train": len(split_records["train"]),
        "num_val": len(split_records["val"]),
        "num_test": len(split_records["test"]),
        "seed": seed,
        "image_size": list(configured_image_size),
        "patch_size": list(patch_size),
        "patch_stride": list(patch_stride),
        "use_official_drive_split": use_official_drive_split,
        "augmentation": {"enabled": aug_enabled, "transforms": aug_names},
    }
    (splits_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _validate_prepared_dataset(project_root, processed_dir, splits_dir, patch_size)

    print("Dataset preparado correctamente.")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for segmentation training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    run_preparation(args.config)


if __name__ == "__main__":
    main()