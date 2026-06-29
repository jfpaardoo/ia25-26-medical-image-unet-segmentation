"""Image and mask preprocessing helpers."""

from __future__ import annotations

import numpy as np

def apply_mask_format(mask, mask_format="binary"):
    arr = np.asarray(mask)
    if mask_format == "binary":
        if arr.ndim == 3:
            arr = arr.max(axis=-1)
        return (arr > 0).astype(np.uint8)
    return arr
