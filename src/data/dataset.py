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
    from typing import List

    root = Path(root_dir)
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def is_image(p: Path) -> bool:
        return p.suffix.lower() in image_exts

    samples: List[SegmentationSample] = []

    images_dir = root / "images"
    masks_dir = root / "masks"

    # Case A: explicit images/ and masks/ directories with mirrored structure
    if images_dir.is_dir() and masks_dir.is_dir():
        for img in images_dir.rglob("*"):
            if not img.is_file() or not is_image(img):
                continue
            rel = img.relative_to(images_dir)
            candidate = masks_dir / rel
            if candidate.exists():
                samples.append(SegmentationSample(image_path=img, mask_path=candidate))
                continue
            # try mask with _mask suffix next to the image
            mask_with_suffix = img.with_name(img.stem + "_mask" + img.suffix)
            if mask_with_suffix.exists():
                samples.append(SegmentationSample(image_path=img, mask_path=mask_with_suffix))

        return sorted(samples, key=lambda s: s.image_path.as_posix())

    # Case B: try to discover pairs in a single tree under root
    images = [p for p in root.rglob("*") if p.is_file() and is_image(p)]
    for img in images:
        mask_candidate = None

        # 1) mask in same directory with _mask suffix
        candidate = img.with_name(img.stem + "_mask" + img.suffix)
        if candidate.exists():
            mask_candidate = candidate

        # 2) mirror under root/masks/<relative path>
        if mask_candidate is None:
            try:
                rel = img.relative_to(root)
                candidate2 = root / "masks" / rel
                if candidate2.exists():
                    mask_candidate = candidate2
            except Exception:
                pass

        # 3) any file under root with same stem (closest match)
        if mask_candidate is None:
            for p in root.rglob("*"):
                if p.is_file() and p.stem == img.stem and p != img:
                    mask_candidate = p
                    break

        if mask_candidate is not None:
            samples.append(SegmentationSample(image_path=img, mask_path=mask_candidate))

    return sorted(samples, key=lambda s: s.image_path.as_posix())