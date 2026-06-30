"""Patch extraction utilities for large medical images."""

from __future__ import annotations

import numpy as np


def _pad_to_fit(array: np.ndarray, patch_size: tuple[int, int], stride: tuple[int, int]) -> np.ndarray:
    """Pad array to neatly fit patch extraction."""
    h, w = array.shape[:2]
    ph, pw = patch_size
    sh, sw = stride

    out_h = ph if h <= ph else ph + int(np.ceil((h - ph) / sh)) * sh
    out_w = pw if w <= pw else pw + int(np.ceil((w - pw) / sw)) * sw

    pad_h = max(out_h - h, 0)
    pad_w = max(out_w - w, 0)
    
    # We assume array is (H, W, 1)
    pad_spec = ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(array, pad_spec, mode="reflect")


def extract_patches(image: np.ndarray, mask: np.ndarray, patch_size: tuple[int, int], stride: tuple[int, int]):
    """Split an image-mask pair into aligned patches."""
    ph, pw = patch_size
    sh, sw = stride

    img_pad = _pad_to_fit(image, patch_size, stride)
    msk_pad = _pad_to_fit(mask, patch_size, stride)

    padded_h, padded_w = img_pad.shape[:2]
    image_patches = []
    mask_patches = []
    positions = []

    for top in range(0, padded_h - ph + 1, sh):
        for left in range(0, padded_w - pw + 1, sw):
            image_patches.append(img_pad[top : top + ph, left : left + pw, ...])
            mask_patches.append(msk_pad[top : top + ph, left : left + pw, ...])
            positions.append((top, left))

    return (
        np.stack(image_patches, axis=0),
        np.stack(mask_patches, axis=0),
        positions,
        (padded_h, padded_w),
    )


def reconstruct_from_patches(patches: list | np.ndarray, positions: list[tuple[int, int]], output_shape: tuple[int, int]):
    """Reconstruct a 2D array from patches using overlap averaging."""
    patch_arr = np.asarray(patches)
    out_h, out_w = output_shape
    ph, pw = patch_arr.shape[1], patch_arr.shape[2]

    max_bottom = max(top + ph for top, _ in positions)
    max_right = max(left + pw for _, left in positions)
    
    accum = np.zeros((max_bottom, max_right, 1), dtype=np.float32)
    count = np.zeros((max_bottom, max_right, 1), dtype=np.float32)

    for idx, (top, left) in enumerate(positions):
        accum[top : top + ph, left : left + pw, ...] += patch_arr[idx].astype(np.float32)
        count[top : top + ph, left : left + pw, ...] += 1.0

    np.maximum(count, 1.0, out=count)
    rebuilt = accum / count
    return rebuilt[:out_h, :out_w, ...]