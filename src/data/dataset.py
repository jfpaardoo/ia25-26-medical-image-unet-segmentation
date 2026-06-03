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
    pairs = {(root / "images", root / "masks")} if (root / "images").is_dir() and (root / "masks").is_dir() else set()
    pairs.update((d, d.parent / "masks") for d in root.rglob("images") if d.is_dir() and (d.parent / "masks").is_dir())
    return sorted(pairs, key=lambda p: p[0].as_posix())


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


def _find_explicit_mask(img: Path, images_dir: Path, masks_dir: Path, root: Path) -> Path | None:
    candidate = masks_dir / img.relative_to(images_dir)
    if candidate.exists():
        return candidate

    sibling_matches = sorted(
        (p for p in masks_dir.rglob("*") if p.is_file() and p.stem == img.stem),
        key=lambda path: path.as_posix()
    )
    if sibling_matches:
        return sibling_matches[0]

    return _resolve_mask_for_image(img, root)


def _discover_explicit_layout(root: Path) -> list[SegmentationSample]:
    samples = []
    for images_dir, masks_dir in _image_mask_pairs(root):
        for img in (p for p in images_dir.rglob("*") if p.is_file() and _is_image(p)):
            mask_candidate = _find_explicit_mask(img, images_dir, masks_dir, root)
            if mask_candidate is not None:
                samples.append(SegmentationSample(image_path=img, mask_path=mask_candidate))
    return samples


def _discover_fallback_layout(root: Path) -> list[SegmentationSample]:
    samples = []
    for img in (p for p in root.rglob("*") if p.is_file() and _is_image(p)):
        mask_candidate = _resolve_mask_for_image(img, root)
        if mask_candidate is not None:
            samples.append(SegmentationSample(image_path=img, mask_path=mask_candidate))
    return samples


def discover_samples(root_dir: Path) -> list[SegmentationSample]:
    """Return the dataset samples found under root_dir.

    The concrete discovery rules depend on the chosen dataset layout.
    """
    root = Path(root_dir)
    samples = _discover_explicit_layout(root)
    if samples:
        return sorted(samples, key=lambda s: s.image_path.as_posix())
        
    samples = _discover_fallback_layout(root)
    return sorted(samples, key=lambda s: s.image_path.as_posix())