"""Image and mask preprocessing helpers."""

from __future__ import annotations

import numpy as np
import cv2


def apply_mask_format(mask, mask_format="binary"):
    """Convert a mask array to the configured semantic format."""
    arr = np.asarray(mask)
    if mask_format == "binary":
        if arr.ndim == 3:
            arr = arr.max(axis=-1)
        return (arr > 0).astype(np.uint8)
    return arr


def to_grayscale(image):
    """Convert an image to single-channel grayscale.

    If the image is already 2-D or single-channel it is returned as-is
    (squeezed to 2-D).  RGB/BGR 3-channel images are converted using
    OpenCV (luminance-weighted).  4-channel images (BGRA) are also
    handled.
    """
    arr = np.asarray(image)

    if arr.ndim == 2:
        return arr

    channels = arr.shape[2] if arr.ndim == 3 else 1
    if channels == 1:
        return arr[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)

    # Fallback: take the mean across channels
    return arr.mean(axis=-1).astype(arr.dtype)


def normalize_image(image):
    """Normalize an image array to float32 in range [0, 1].

    If input is integer (e.g. uint8) it is divided by 255. Otherwise cast
    to float32 and return as-is.
    """
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)