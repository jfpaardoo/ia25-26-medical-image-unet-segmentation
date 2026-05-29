import os
from pathlib import Path

import pytest

from src.data.dataset import discover_samples


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def test_discover_images_and_masks_dirs(tmp_path: Path):
    root = tmp_path / "dataset"
    images = root / "images"
    masks = root / "masks"

    # create image files and mirrored mask files
    touch(images / "img1.png")
    touch(images / "sub" / "img2.jpg")
    touch(masks / "img1.png")
    touch(masks / "sub" / "img2.jpg")

    samples = discover_samples(root)
    pairs = {(s.image_path.name, s.mask_path.name) for s in samples}
    assert ("img1.png", "img1.png") in pairs
    assert ("img2.jpg", "img2.jpg") in pairs


def test_discover_mask_suffix_and_root_masks(tmp_path: Path):
    root = tmp_path / "dataset2"

    # image with mask in same directory using _mask suffix
    touch(root / "case" / "imageA.png")
    touch(root / "case" / "imageA_mask.png")

    # image with mask in root/masks mirrored
    touch(root / "nested" / "imageB.tif")
    touch(root / "masks" / "nested" / "imageB.tif")

    samples = discover_samples(root)
    names = {(s.image_path.relative_to(root).as_posix(), s.mask_path.relative_to(root).as_posix()) for s in samples}

    assert ("case/imageA.png", "case/imageA_mask.png") in names
    assert ("nested/imageB.tif", "masks/nested/imageB.tif") in names


def test_discover_does_not_cross_match_repeated_stems(tmp_path: Path):
    root = tmp_path / "dataset3"

    touch(root / "case_a" / "image.png")
    touch(root / "case_a" / "image_mask.png")
    touch(root / "case_b" / "image.png")
    touch(root / "case_b" / "image_mask.png")

    samples = discover_samples(root)
    pairs = {
        (s.image_path.relative_to(root).as_posix(), s.mask_path.relative_to(root).as_posix())
        for s in samples
    }

    assert ("case_a/image.png", "case_a/image_mask.png") in pairs
    assert ("case_b/image.png", "case_b/image_mask.png") in pairs
