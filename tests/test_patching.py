import numpy as np
import pytest

from src.data.patching import extract_patches, reconstruct_from_patches


def test_extract_patches_with_padding_and_stride():
    image = np.zeros((5, 5, 1), dtype=np.float32)
    mask = np.zeros((5, 5), dtype=np.uint8)

    img_p, mask_p, positions, padded_shape = extract_patches(
        image,
        mask,
        patch_size=(4, 4),
        stride=(3, 3),
    )

    assert img_p.shape[0] == mask_p.shape[0]
    assert img_p.shape[1:3] == (4, 4)
    assert padded_shape == (7, 7)
    assert len(positions) == img_p.shape[0]


def test_reconstruct_from_overlapping_patches():
    image = np.arange(36, dtype=np.float32).reshape(6, 6)
    mask = image.astype(np.uint8)

    patches, _, positions, _ = extract_patches(image, mask, patch_size=(4, 4), stride=(2, 2))
    rebuilt = reconstruct_from_patches(patches, positions, output_shape=(6, 6))

    assert rebuilt.shape == (6, 6)
    assert np.allclose(rebuilt, image, atol=1e-6)


def test_extract_patches_rejects_mismatched_shapes():
    image = np.zeros((8, 8, 1), dtype=np.float32)
    mask = np.zeros((7, 8), dtype=np.uint8)

    with pytest.raises(ValueError):
        extract_patches(image, mask, patch_size=(4, 4))
