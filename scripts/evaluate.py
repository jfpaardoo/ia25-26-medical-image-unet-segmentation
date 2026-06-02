"""Evaluate a trained segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import CONFIGS_DIR, load_yaml_config
from src.evaluation.inference import evaluate_model


def _ensure_mask_shape(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        return mask[..., None]
    return mask


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray | tuple[np.ndarray, ...]]:
    data = np.load(path)
    if "images" not in data:
        raise ValueError("NPZ file must contain an 'images' array.")

    images = data["images"]
    expert_keys = [key for key in ("masks_expert1", "masks_expert2") if key in data]
    if expert_keys:
        masks = tuple(_ensure_mask_shape(np.asarray(data[key])) for key in expert_keys)
        return images, masks

    if "masks" not in data:
        raise ValueError("NPZ file must contain 'masks' or 'masks_expert1'/'masks_expert2' arrays.")

    return images, _ensure_mask_shape(np.asarray(data["masks"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "default.yaml")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    _ = load_yaml_config(args.config)
    images, masks = _load_npz(args.data_npz)
    results = evaluate_model(args.model, images, masks, threshold=args.threshold)

    for name, value in results.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()