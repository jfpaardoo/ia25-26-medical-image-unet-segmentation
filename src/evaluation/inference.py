"""Inference helpers for trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import keras

from .metrics import DiceCoefficient, dice_coefficient, dice_loss, iou_score


def _prepare_images(images: np.ndarray, model: keras.Model) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    input_shape = model.input_shape
    expected_channels = input_shape[-1] if input_shape is not None else None

    if images.ndim == 2:
        if expected_channels not in (None, 1):
            raise ValueError(
                "Input channels do not match model expectation. "
                f"Got 1, expected {expected_channels}."
            )
        images = images[..., np.newaxis]
        images = images[np.newaxis, ...]
        return images

    if images.ndim == 3:
        if expected_channels is not None and images.shape[-1] == expected_channels:
            images = images[np.newaxis, ...]
            return images
        if expected_channels not in (None, 1):
            raise ValueError(
                "Input channels do not match model expectation. "
                f"Got 1, expected {expected_channels}."
            )
        images = images[..., np.newaxis]
        return images

    if images.ndim == 4:
        if expected_channels is not None and images.shape[-1] != expected_channels:
            raise ValueError(
                "Input channels do not match model expectation. "
                f"Got {images.shape[-1]}, expected {expected_channels}."
            )
        return images

    raise ValueError("Images must have 2, 3, or 4 dimensions.")



def _postprocess_predictions(predictions: np.ndarray, threshold: float) -> np.ndarray:
    if predictions.shape[-1] == 1:
        return (predictions >= threshold).astype(np.uint8)
    return np.argmax(predictions, axis=-1).astype(np.uint8)


def load_model(model_path: Path | str, compile: bool = False) -> keras.Model:
    """Load a serialized Keras model from disk."""

    custom_objects = {
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "iou_score": iou_score,
        "DiceCoefficient": DiceCoefficient,
    }
    return keras.saving.load_model(str(model_path), compile=compile, custom_objects=custom_objects)


def predict_mask(
    model_or_path: keras.Model | Path | str,
    images: np.ndarray,
    threshold: float = 0.5,
    batch_size: Optional[int] = None,
    return_probabilities: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Predict segmentation masks from prepared images."""

    model = model_or_path
    if isinstance(model_or_path, (str, Path)):
        model = load_model(model_or_path)

    images = _prepare_images(images, model)
    probabilities = model.predict(images, batch_size=batch_size, verbose=0)
    masks = _postprocess_predictions(probabilities, threshold)
    if return_probabilities:
        return masks, probabilities
    return masks


def evaluate_model(
    model_or_path: keras.Model | Path | str,
    images: np.ndarray,
    masks: np.ndarray,
    threshold: float = 0.5,
    metrics: Optional[Iterable] = None,
) -> dict[str, float]:
    """Run inference and compute metrics over prepared arrays."""

    if metrics is None:
        metrics = [dice_coefficient, iou_score]

    masks = np.asarray(masks)
    pred_masks = predict_mask(model_or_path, images, threshold=threshold)
    results: dict[str, float] = {}
    for metric in metrics:
        name = getattr(metric, "__name__", metric.__class__.__name__)
        results[name] = float(metric(masks, pred_masks))
    return results