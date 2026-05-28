"""Train the U-Net segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import CONFIGS_DIR, load_yaml_config
from src.training.train import train_model


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "images" not in data or "masks" not in data:
        raise ValueError("NPZ file must contain 'images' and 'masks' arrays.")
    return data["images"], data["masks"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the U-Net segmentation model.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "default.yaml")
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    train_images, train_masks = _load_npz(args.train_npz)
    val_data = None
    if args.val_npz:
        val_images, val_masks = _load_npz(args.val_npz)
        val_data = (val_images, val_masks)

    train_model(train_data=(train_images, train_masks), val_data=val_data, config=config)


if __name__ == "__main__":
    main()