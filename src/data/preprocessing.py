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


def resize_pair(image, mask, target_size, mask_format="binary",
                target_channels=None):
    """Resize an image-mask pair to the target size.

    Args:
        image: HxW or HxWxC numpy array (uint8 or float).
        mask: HxW numpy array (integer labels).
        target_size: (height, width) tuple or int for square resize.
        mask_format: Semantic mask format expected by the pipeline.
        target_channels: If ``1``, convert image to grayscale before
            returning.  If ``3``, ensure 3 channels.  ``None`` leaves
            the image unchanged.

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

    # Channel adjustment
    if target_channels == 1:
        resized_img = to_grayscale(resized_img)
    elif target_channels == 3 and resized_img.ndim == 2:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

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