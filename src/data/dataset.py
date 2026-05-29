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

    def matching_files(directory: Path, stem: str) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(
            [p for p in directory.rglob("*") if p.is_file() and p.stem == stem],
            key=lambda path: path.as_posix(),
        )

    def resolve_mask_for_image(img: Path) -> Path | None:
        candidate = img.with_name(img.stem + "_mask" + img.suffix)
        if candidate.exists():
            return candidate

        same_dir_matches = matching_files(img.parent, img.stem)
        for match in same_dir_matches:
            if match != img:
                return match

        try:
            rel = img.relative_to(root)
        except ValueError:
            return None

        mirrored_candidate = root / "masks" / rel
        if mirrored_candidate.exists():
            return mirrored_candidate

        mirrored_dir_matches = matching_files(root / "masks" / rel.parent, img.stem)
        for match in mirrored_dir_matches:
            return match

        return None

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

            mask_candidate = resolve_mask_for_image(img)
            if mask_candidate is not None:
                samples.append(SegmentationSample(image_path=img, mask_path=mask_candidate))

        return sorted(samples, key=lambda s: s.image_path.as_posix())

    # Case B: try to discover pairs in a single tree under root
    images = [p for p in root.rglob("*") if p.is_file() and is_image(p)]
    for img in images:
        mask_candidate = resolve_mask_for_image(img)
        if mask_candidate is not None:
            samples.append(SegmentationSample(image_path=img, mask_path=mask_candidate))

    return sorted(samples, key=lambda s: s.image_path.as_posix())