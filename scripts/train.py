"""Train the U-Net segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from src.config import CONFIGS_DIR, FINAL_MODELS_DIR, load_yaml_config
from src.training.train import train_model


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "images" not in data or "masks" not in data:
        raise ValueError("NPZ file must contain 'images' and 'masks' arrays.")
    images = data["images"]
    masks = data["masks"]
    if images.ndim == 3:
        images = images[..., None]
    if masks.ndim == 3:
        masks = masks[..., None]
    return images, masks


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

    seed = config.get("project", {}).get("seed", 42)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    final_models_dir = config.get("outputs", {}).get("final_model_dir", FINAL_MODELS_DIR)
    if not isinstance(final_models_dir, Path):
        final_models_dir = Path(final_models_dir)
    final_models_dir.mkdir(parents=True, exist_ok=True)

    fold = 1
    for train_idx, val_idx in kf.split(train_images):
        print(f"--- Training Fold {fold} ---")
        x_train, y_train = train_images[train_idx], train_masks[train_idx]
        x_val, y_val = train_images[val_idx], train_masks[val_idx]

        model, history = train_model(train_data=(x_train, y_train), val_data=(x_val, y_val), config=config)

        model_path = final_models_dir / f"unet_fold_{fold}.keras"
        model.save(model_path)
        print(f"Saved model for fold {fold} at {model_path}")
        fold += 1


if __name__ == "__main__":
    main()