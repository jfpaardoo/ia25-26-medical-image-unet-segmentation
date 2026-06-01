"""Patch extraction utilities for large medical images."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_hw(size) -> tuple[int, int]:
    if isinstance(size, int):
        side = int(size)
        return side, side
    if isinstance(size, Sequence) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError("size must be an int or a (height, width) pair")


def _pad_to_fit(array: np.ndarray, patch_size: tuple[int, int], stride: tuple[int, int]) -> np.ndarray:
    h, w = array.shape[:2]
    ph, pw = patch_size
    sh, sw = stride

    out_h = ph if h <= ph else ph + int(np.ceil((h - ph) / sh)) * sh
    out_w = pw if w <= pw else pw + int(np.ceil((w - pw) / sw)) * sw

    pad_h = max(out_h - h, 0)
    pad_w = max(out_w - w, 0)
    pad_spec = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (array.ndim - 2)

    # Prefer reflected padding to avoid creating artificial zero borders.
    spatial_shape = array.shape[:2]
    if all(dim > 1 for dim in spatial_shape):
        try:
            return np.pad(array, pad_spec, mode="reflect")
        except ValueError:
            # Reflect padding can fail for tiny arrays; fall back to edge padding.
            return np.pad(array, pad_spec, mode="edge")

    return np.pad(array, pad_spec, mode="edge")


def extract_patches(image, mask, patch_size, stride=None):
    """Split an image-mask pair into aligned patches.

    Returns:
        image_patches: np.ndarray of shape (N, ph, pw, C?)
        mask_patches: np.ndarray of shape (N, ph, pw, C?)
        positions: list of (top, left) coordinates on padded arrays
        padded_shape: tuple (height, width)
    """

    img = np.asarray(image)
    msk = np.asarray(mask)

    if img.shape[:2] != msk.shape[:2]:
        raise ValueError("image and mask must have identical height and width")

    ph, pw = _to_hw(patch_size)
    sh, sw = _to_hw(stride if stride is not None else patch_size)

    if ph <= 0 or pw <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("patch_size and stride must be positive")

    img_pad = _pad_to_fit(img, (ph, pw), (sh, sw))
    msk_pad = _pad_to_fit(msk, (ph, pw), (sh, sw))

    padded_h, padded_w = img_pad.shape[:2]
    image_patches: list[np.ndarray] = []
    mask_patches: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []

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


def reconstruct_from_patches(patches, positions, output_shape):
    """Reconstruct a 2D/3D array from patches using overlap averaging."""

    patch_arr = np.asarray(patches)
    if patch_arr.ndim < 3:
        raise ValueError("patches must have shape (N, H, W, ...) or (N, H, W)")
    if len(positions) != patch_arr.shape[0]:
        raise ValueError("positions length must match number of patches")

    out_h, out_w = _to_hw(output_shape)
    ph, pw = patch_arr.shape[1], patch_arr.shape[2]
    tail_shape = patch_arr.shape[3:]

    max_bottom = max(top + ph for top, _ in positions)
    max_right = max(left + pw for _, left in positions)
    accum = np.zeros((max_bottom, max_right) + tail_shape, dtype=np.float32)
    count = np.zeros((max_bottom, max_right) + ((1,) if tail_shape else ()), dtype=np.float32)

    for idx, (top, left) in enumerate(positions):
        accum[top : top + ph, left : left + pw, ...] += patch_arr[idx].astype(np.float32)
        if tail_shape:
            count[top : top + ph, left : left + pw, 0] += 1.0
        else:
            count[top : top + ph, left : left + pw] += 1.0

    np.maximum(count, 1.0, out=count)
    rebuilt = accum / count
    return rebuilt[:out_h, :out_w, ...]