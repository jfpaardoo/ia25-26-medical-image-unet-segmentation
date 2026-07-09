"""Training entry points and orchestration helpers."""

from __future__ import annotations

from typing import Any, Optional

import keras

from src.config import PROJECT_ROOT, CHECKPOINTS_DIR, LOGS_DIR
from src.evaluation.metrics import DiceCoefficient, bce_dice_loss
from src.models.unet import build_unet
from src.training.callbacks import build_callbacks

def train_model(
    train_data,
    val_data=None,
    config: Optional[dict[str, Any]] = None,
    model: Optional[keras.Model] = None,
):
    """Train the model using prepared datasets or arrays."""
    
    config = config or {}
    training_cfg = config.get("training", {})

    if model is None:
        patch_size = tuple(config.get("data", {}).get("patch_size", [128, 128]))
        model = build_unet(
            input_shape=(patch_size[0], patch_size[1], 1), 
            num_classes=1,
            base_filters=config.get("model", {}).get("base_filters", 16),
            depth=config.get("model", {}).get("depth", 4),
            dropout_rate=config.get("model", {}).get("dropout_rate", 0.25),
            use_batch_norm=config.get("model", {}).get("use_batch_norm", True)
        )

    checkpoints_dir = config.get("outputs", {}).get("checkpoints_dir", CHECKPOINTS_DIR)
    logs_dir = config.get("outputs", {}).get("logs_dir", LOGS_DIR)
    
    if isinstance(checkpoints_dir, str):
        checkpoints_dir = PROJECT_ROOT / checkpoints_dir
    if isinstance(logs_dir, str):
        logs_dir = PROJECT_ROOT / logs_dir

    monitor_metric = "val_dice" if val_data is not None else "dice"
    callbacks = build_callbacks(
        checkpoints_dir=checkpoints_dir, 
        logs_dir=logs_dir, 
        monitor=monitor_metric
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(training_cfg.get("learning_rate", 1e-4))),
        loss=bce_dice_loss,
        metrics=[
            DiceCoefficient(name="dice"),
        ],
    )

    history = model.fit(
        x=train_data,
        validation_data=val_data,
        epochs=int(training_cfg.get("epochs", 100)),
        callbacks=callbacks
    )

    return model, history