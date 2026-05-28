"""Image and mask preprocessing helpers."""

from __future__ import annotations


def resize_pair(image, mask, target_size):
    """Resize an image-mask pair to the target size."""

    raise NotImplementedError("Implement resizing, normalization and mask alignment here.")


def normalize_image(image):
    """Normalize an image tensor or array."""

    raise NotImplementedError("Implement image normalization here.")