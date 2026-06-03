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
    data_cfg = config.get("data", {})
    input_dims = data_cfg.get("patch_size", data_cfg.get("image_size", (256, 256)))
    channels = config.get("model", {}).get("input_channels", 1)
    return (int(input_dims[0]), int(input_dims[1]), int(channels))


def _resolve_output_channels(config: dict[str, Any]) -> int:
    return int(config.get("model", {}).get("output_channels", 1))


def _resolve_output_dir(config: dict[str, Any], key: str, default: Path) -> Path:
    raw_path = config.get("outputs", {}).get(key)
    return (PROJECT_ROOT / raw_path).resolve() if raw_path else default


def _build_model_if_needed(model: Optional[keras.Model], config: dict[str, Any]) -> keras.Model:
    if model is not None:
        return model
    model_cfg = config.get("model", {})
    return build_unet(
        input_shape=_resolve_input_shape(config),
        num_classes=_resolve_output_channels(config),
        base_filters=int(model_cfg.get("base_filters", 32)),
        depth=int(model_cfg.get("depth", 4)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.0)),
        use_batch_norm=bool(model_cfg.get("use_batch_norm", model_cfg.get("use_batchnorm", True))),
    )


def _resolve_loss(loss: Optional[Any], config: dict[str, Any], training_cfg: dict[str, Any]) -> Any:
    if loss is not None:
        return loss
    if _resolve_output_channels(config) != 1:
        return "categorical_crossentropy"
    
    loss_choice = str(training_cfg.get("loss", "bce_dice")).strip().lower()
    bce = keras.losses.BinaryCrossentropy()
    
    if loss_choice == "dice":
        return dice_loss
    if loss_choice == "bce":
        return bce
        
    def _combined_loss(y_true, y_pred):
        return 0.5 * bce(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)
    return _combined_loss


def _prepare_fit_kwargs(train_data, val_data, epochs: int, batch_size: Optional[int], val_split: float, callbacks: list) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"epochs": epochs, "callbacks": callbacks}
    if batch_size:
        kwargs["batch_size"] = int(batch_size)
    
    if val_data is not None:
        kwargs["validation_data"] = val_data
    elif val_split > 0.0 and isinstance(train_data, (tuple, list)):
        kwargs["validation_split"] = val_split

    if isinstance(train_data, (tuple, list)) and len(train_data) == 2:
        kwargs.update({"x": train_data[0], "y": train_data[1]})
    else:
        kwargs["x"] = train_data
    return kwargs


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
    
    model = _build_model_if_needed(model, config)
    loss_fn = _resolve_loss(loss, config, training_cfg)

    val_split = float(training_cfg.get("validation_split", 0.0))
    has_validation = val_data is not None or val_split > 0.0
    monitor = monitor or ("val_loss" if has_validation else "loss")
    
    checkpoints_dir = _resolve_output_dir(config, "checkpoints_dir", CHECKPOINTS_DIR)
    logs_dir = _resolve_output_dir(config, "logs_dir", LOGS_DIR)
    final_models_dir = _resolve_output_dir(config, "final_model_dir", FINAL_MODELS_DIR)
    
    callbacks = build_callbacks(checkpoints_dir=checkpoints_dir, logs_dir=logs_dir, monitor=monitor)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(training_cfg.get("learning_rate", 1e-4))),
        loss=loss_fn,
        metrics=[DiceCoefficient(name="dice"), iou_score],
    )

    fit_kwargs = _prepare_fit_kwargs(
        train_data, val_data,
        epochs=int(training_cfg.get("epochs", 100)),
        batch_size=training_cfg.get("batch_size"),
        val_split=val_split,
        callbacks=callbacks
    )

    history = model.fit(**fit_kwargs)

    final_models_dir.mkdir(parents=True, exist_ok=True)
    model.save(final_models_dir / "unet_final.keras")

    return model, history