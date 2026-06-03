"""Dataset discovery helpers for paired image-mask data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentationSample:
    """Pair of image and mask paths."""

    image_path: Path
    mask_path: Path


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _matching_files(directory: Path, stem: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.rglob("*") if p.is_file() and p.stem == stem],
        key=lambda path: path.as_posix(),
    )


def _image_mask_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()

    explicit_images_dirs = [path for path in root.rglob("images") if path.is_dir()]
    for images_dir in explicit_images_dirs:
        masks_dir = images_dir.parent / "masks"
        if masks_dir.is_dir():
            pair = (images_dir, masks_dir)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)

    if (root / "images").is_dir() and (root / "masks").is_dir():
        pair = (root / "images", root / "masks")
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    return sorted(pairs, key=lambda pair: pair[0].as_posix())


def _resolve_mask_for_image(img: Path, root: Path) -> Path | None:
    candidate = img.with_name(img.stem + "_mask" + img.suffix)
    if candidate.exists():
        return candidate

    same_dir_matches = _matching_files(img.parent, img.stem)
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

    mirrored_dir_matches = _matching_files(root / "masks" / rel.parent, img.stem)
    for match in mirrored_dir_matches:
        return match

    return None


def discover_samples(root_dir: Path) -> list[SegmentationSample]:
    """Return the dataset samples found under root_dir.

    The concrete discovery rules depend on the chosen dataset layout.
    """
    root = Path(root_dir)
    samples: list[SegmentationSample] = []

    # Case A: explicit images/ and masks/ directories with mirrored structure
    for images_dir, masks_dir in _image_mask_pairs(root):
        for img in images_dir.rglob("*"):
            if not img.is_file() or not _is_image(img):
                continue
            
            rel = img.relative_to(images_dir)
            candidate = masks_dir / rel
            if candidate.exists():
                samples.append(SegmentationSample(image_path=img, mask_path=candidate))
                continue

            sibling_matches = sorted(
                [p for p in masks_dir.rglob("*") if p.is_file() and p.stem == img.stem],
                key=lambda path: path.as_posix(),
            )
            if sibling_matches:
                samples.append(SegmentationSample(image_path=img, mask_path=sibling_matches[0]))
                continue

            mask_candidate = _resolve_mask_for_image(img, root)
            if mask_candidate is not None:
                samples.append(SegmentationSample(image_path=img, mask_path=mask_candidate))

    if samples:
        return sorted(samples, key=lambda s: s.image_path.as_posix())

    # Case B: try to discover pairs in a single tree under root
    images = [p for p in root.rglob("*") if p.is_file() and _is_image(p)]
    for img in images:
        mask_candidate = _resolve_mask_for_image(img, root)
        if mask_candidate is not None:
            samples.append(SegmentationSample(image_path=img, mask_path=mask_candidate))

    return sorted(samples, key=lambda s: s.image_path.as_posix())