"""Dataset discovery helpers for paired image-mask data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentationSample:
    """Pair of image and mask paths."""

    image_path: Path
    mask_path: Path


def discover_samples(root_dir: Path) -> list[SegmentationSample]:
    """Return the dataset samples found under root_dir.

    The concrete discovery rules depend on the chosen dataset layout.
    """

    raise NotImplementedError("Define how image-mask pairs are discovered for the dataset.")