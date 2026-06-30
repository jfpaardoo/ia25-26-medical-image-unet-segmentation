"""Image and mask preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np


def load_grayscale_image(path: Path | str, normalize: bool = False) -> np.ndarray:
    """Loads an image in grayscale format. Optionally scales it to [0, 1]."""
    img = keras.utils.img_to_array(keras.utils.load_img(path, color_mode="grayscale"))
    if normalize:
        return img.astype(np.float32) / 255.0
    return img


def binarize_mask(mask: np.ndarray, threshold: float = 127) -> np.ndarray:
    """Binarizes a mask to 0 and 1 values."""
    return (mask > threshold).astype(np.uint8)
