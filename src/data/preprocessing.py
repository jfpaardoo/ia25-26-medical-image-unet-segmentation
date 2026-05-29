"""Image and mask preprocessing helpers."""

from __future__ import annotations

import numpy as np
import cv2


def apply_mask_format(mask, mask_format="binary"):
    """Convert a mask array to the configured semantic format."""
    arr = np.asarray(mask)
    normalized_format = str(mask_format).strip().lower()

    if normalized_format in {"binary", "grayscale", "identity", "raw"}:
        if normalized_format == "binary":
            if arr.ndim == 3:
                arr = arr.max(axis=-1)
            return (arr > 0).astype(np.uint8)
        return arr

    raise ValueError(f"Unsupported mask_format: {mask_format!r}")


def resize_pair(image, mask, target_size, mask_format="binary"):
    """Resize an image-mask pair to the target size.

    Args:
        image: HxW or HxWxC numpy array (uint8 or float).
        mask: HxW numpy array (integer labels).
        target_size: (height, width) tuple or int for square resize.
        mask_format: Semantic mask format expected by the pipeline.

    Returns:
        (resized_image, resized_mask)

    Notes:
        - Image resized with bilinear interpolation.
        - Mask resized with nearest neighbor to preserve labels.
    """
    img = np.asarray(image)
    msk = np.asarray(mask)

    if isinstance(target_size, int):
        h = w = int(target_size)
    else:
        h, w = int(target_size[0]), int(target_size[1])

    # OpenCV expects (width, height)
    resized_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
    resized_mask = apply_mask_format(resized_mask, mask_format=mask_format)

    return resized_img, resized_mask


def normalize_image(image):
    """Normalize an image array to float32 in range [0, 1].

    If input is integer (e.g. uint8) it is divided by 255. Otherwise cast
    to float32 and return as-is.
    """
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)