"""Image and mask preprocessing helpers."""

from __future__ import annotations

import numpy as np
import cv2


def resize_pair(image, mask, target_size):
    """Resize an image-mask pair to the target size.

    Args:
        image: HxW or HxWxC numpy array (uint8 or float).
        mask: HxW numpy array (integer labels).
        target_size: (height, width) tuple or int for square resize.

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