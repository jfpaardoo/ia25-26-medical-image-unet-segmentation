"""Training entry points and orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import keras

from src.config import CHECKPOINTS_DIR, FINAL_MODELS_DIR, LOGS_DIR, PROJECT_ROOT
from src.evaluation.metrics import DiceCoefficient, dice_loss, iou_score
from src.models.unet import build_unet
from src.training.callbacks import build_callbacks


def _resolve_input_shape(config: dict[str, Any]) -> tuple[int, int, int]:
    image_size = config.get("data", {}).get("image_size", (256, 256))
    channels = config.get("model", {}).get("input_channels", 1)
    return (int(image_size[0]), int(image_size[1]), int(channels))


def _resolve_output_channels(config: dict[str, Any]) -> int:
    return int(config.get("model", {}).get("output_channels", 1))


def _resolve_output_dir(config: dict[str, Any], key: str, default: Path) -> Path:
    outputs = config.get("outputs", {})
    raw_path = outputs.get(key)
    if not raw_path:
        return default
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def train_model(
    train_data,
    val_data=None,
    config: Optional[dict[str, Any]] = None,
    model: Optional[keras.Model] = None,
    loss: Optional[str] = None,
    monitor: Optional[str] = None,
):
    """Train the model using prepared datasets or arrays."""

    if train_data is None:
        raise ValueError("train_data must be provided. Pass a dataset or (x, y) tuple.")

    config = config or {}
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    base_filters = int(model_cfg.get("base_filters", 32))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.0))
    learning_rate = float(training_cfg.get("learning_rate", 1e-4))
    epochs = int(training_cfg.get("epochs", 100))
    batch_size = training_cfg.get("batch_size")
    validation_split = float(training_cfg.get("validation_split", 0.0))

    if model is None:
        input_shape = _resolve_input_shape(config)
        output_channels = _resolve_output_channels(config)
        model = build_unet(
            input_shape=input_shape,
            num_classes=output_channels,
            base_filters=base_filters,
            dropout_rate=dropout_rate,
        )

    if loss is None:
        output_channels = _resolve_output_channels(config)
        if output_channels == 1:
            bce = keras.losses.BinaryCrossentropy()

            def _combined_loss(y_true, y_pred):
                return 0.5 * bce(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

            loss = _combined_loss
        else:
            loss = "categorical_crossentropy"

    has_validation = val_data is not None or validation_split > 0.0
    if monitor is None:
        monitor = "val_dice" if has_validation else "loss"
    checkpoints_dir = _resolve_output_dir(config, "checkpoints_dir", CHECKPOINTS_DIR)
    logs_dir = _resolve_output_dir(config, "logs_dir", LOGS_DIR)
    final_models_dir = _resolve_output_dir(config, "final_model_dir", FINAL_MODELS_DIR)
    callbacks = build_callbacks(checkpoints_dir=checkpoints_dir, logs_dir=logs_dir, monitor=monitor)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[DiceCoefficient(name="dice"), iou_score],
    )

    fit_kwargs: dict[str, Any] = {
        "epochs": epochs,
        "callbacks": callbacks,
    }
    if batch_size is not None:
        fit_kwargs["batch_size"] = int(batch_size)

    if val_data is not None:
        fit_kwargs["validation_data"] = val_data
    elif validation_split > 0.0 and isinstance(train_data, (tuple, list)):
        fit_kwargs["validation_split"] = validation_split

    history = model.fit(train_data, **fit_kwargs)

    final_models_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_models_dir / "unet_final.keras"
    model.save(final_path)

    return model, history