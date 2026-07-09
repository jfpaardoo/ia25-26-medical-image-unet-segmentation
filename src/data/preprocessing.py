"""Image and mask preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np


def load_grayscale_image(path: Path | str) -> np.ndarray:
    """Loads an image in grayscale format as uint8."""
    img = keras.utils.img_to_array(keras.utils.load_img(path, color_mode="grayscale"))
    return img.astype(np.uint8)


def binarize_mask(mask: np.ndarray, threshold: float = 127) -> np.ndarray:
    """Binarizes a mask to 0 and 1 values."""
    return (mask > threshold).astype(np.uint8)
