"""Inference helpers for trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import keras

from .metrics import DiceCoefficient, Specificity, dice_coefficient, dice_loss


def _get_expected_channels(model: keras.Model) -> Optional[int]:
    """Extract the expected number of channels from a Keras model's input shape."""
    shape = model.input_shape
    if isinstance(shape, (list, tuple)) and shape:
        if isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return shape[-1]
    return None


def _prepare_images(images: np.ndarray, model: keras.Model) -> np.ndarray:
    """Format input images to (batch_size, height, width, channels)."""
    images = np.asarray(images, dtype=np.float32)
    expected_channels = _get_expected_channels(model)

    if images.ndim not in (2, 3, 4):
        raise ValueError("Images must have 2, 3, or 4 dimensions.")

    if images.ndim == 2:
        # (H, W) -> (1, H, W, 1)
        images = images[np.newaxis, ..., np.newaxis]
    elif images.ndim == 3:
        # Determine if the 3D input is (H, W, C) or (B, H, W)
        is_hwc = (
            expected_channels is not None and images.shape[-1] == expected_channels
        ) or (
            expected_channels is None and images.shape[-1] in (1, 3, 4)
        )
        if is_hwc:
            images = images[np.newaxis, ...]  # (H, W, C) -> (1, H, W, C)
        else:
            images = images[..., np.newaxis]  # (B, H, W) -> (B, H, W, 1)

    # Validate final 4D shape (B, H, W, C)
    if expected_channels is not None and images.shape[-1] != expected_channels:
        raise ValueError(
            "Input channels do not match model expectation. "
            f"Got {images.shape[-1]}, expected {expected_channels}."
        )

    return images


def _postprocess_predictions(predictions: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold probabilities to generate categorical masks."""
    if predictions.shape[-1] == 1:
        return (predictions >= threshold).astype(np.uint8)
    return np.argmax(predictions, axis=-1).astype(np.uint8)


def load_model(model_path: Path | str, compile: bool = False) -> keras.Model:
    """Load a serialized Keras model from disk."""
    custom_objects = {
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "DiceCoefficient": DiceCoefficient,
        "Specificity": Specificity,
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
