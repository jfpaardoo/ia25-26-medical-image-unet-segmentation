"""Inference helpers for trained models."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import keras

from .metrics import DiceCoefficient, Specificity, dice_coefficient, dice_loss, bce_dice_loss


def load_model(model_path: Path | str, compile: bool = False) -> keras.Model:
    """Load a serialized Keras model from disk."""
    custom_objects = {
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "bce_dice_loss": bce_dice_loss,
        "DiceCoefficient": DiceCoefficient,
        "Specificity": Specificity,
    }
    return keras.saving.load_model(str(model_path), compile=compile, custom_objects=custom_objects)


def predict_mask(
    model: keras.Model,
    images: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 16,
    return_probabilities: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Predict segmentation masks from prepared images (B, 256, 256, 1)."""
    
    probabilities = model.predict(images, batch_size=batch_size, verbose=0)
    masks = (probabilities >= threshold).astype(np.uint8)
    
    if return_probabilities:
        return masks, probabilities
    return masks
