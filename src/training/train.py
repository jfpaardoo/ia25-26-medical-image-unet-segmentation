"""Training entry points and orchestration helpers."""

from __future__ import annotations

from typing import Any, Optional

import keras

from src.config import PROJECT_ROOT
from src.evaluation.metrics import DiceCoefficient, Specificity, bce_dice_loss
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
            base_filters=16,
            depth=4,
            dropout_rate=0.0,
            use_batch_norm=True
        )

    checkpoints_dir = PROJECT_ROOT / config.get("outputs", {}).get("checkpoints_dir", "artifacts/checkpoints")
    logs_dir = PROJECT_ROOT / config.get("outputs", {}).get("logs_dir", "artifacts/logs")
    final_models_dir = PROJECT_ROOT / config.get("outputs", {}).get("final_model_dir", "artifacts/models")

    callbacks = build_callbacks(
        checkpoints_dir=checkpoints_dir, 
        logs_dir=logs_dir, 
        monitor="val_loss"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(training_cfg.get("learning_rate", 1e-4))),
        loss=bce_dice_loss,
        metrics=[
            DiceCoefficient(name="dice"),
            keras.metrics.BinaryIoU(target_class_ids=[1], name="iou"),
            keras.metrics.Recall(name="sensitivity"),
            Specificity(name="specificity"),
        ],
    )

    history = model.fit(
        x=train_data,
        validation_data=val_data,
        epochs=int(training_cfg.get("epochs", 100)),
        callbacks=callbacks
    )

    final_models_dir.mkdir(parents=True, exist_ok=True)
    model.save(final_models_dir / "unet_final.keras")

    return model, history