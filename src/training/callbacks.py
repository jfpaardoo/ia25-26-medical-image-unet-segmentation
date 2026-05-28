"""Custom callbacks for segmentation training."""

from __future__ import annotations


def build_callbacks(*args, **kwargs):
    raise NotImplementedError("Implement checkpoint, early stopping and logging callbacks here.")