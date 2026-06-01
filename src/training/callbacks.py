"""Custom callbacks for segmentation training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import keras


def build_callbacks(
    checkpoints_dir: Path,
    logs_dir: Path,
    monitor: str = "val_dice",
    patience: int = 15,
    save_best_only: bool = True,
) -> list[keras.callbacks.Callback]:
    """Build callbacks for checkpointing and early stopping."""

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    mode = "max" if any(token in monitor for token in ("dice", "iou", "acc")) else "min"
    checkpoint_path = checkpoints_dir / "unet_best.keras"

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.TensorBoard(log_dir=str(logs_dir)),
    ]

    return callbacks