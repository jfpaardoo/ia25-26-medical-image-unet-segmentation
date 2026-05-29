import numpy as np
import pytest

from src.data.preprocessing import resize_pair, normalize_image, apply_mask_format, to_grayscale


def test_resize_and_mask_alignment():
    img = np.random.randint(0, 256, size=(100, 120, 3), dtype=np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[::10, ::10] = 1

    img_r, mask_r = resize_pair(img, mask, (64, 64))

    assert img_r.shape == (64, 64, 3)
    assert mask_r.shape == (64, 64)
    # mask values should remain labels (0 or 1)
    assert set(np.unique(mask_r)).issubset({0, 1})


def test_resize_pair_binary_mask_format():
    img = np.random.randint(0, 256, size=(20, 20), dtype=np.uint8)
    mask = np.array([[0, 2], [3, 0]], dtype=np.uint8)

    _, mask_r = resize_pair(img, mask, (10, 10), mask_format="binary")

    assert set(np.unique(mask_r)).issubset({0, 1})


def test_apply_mask_format_rejects_unknown_value():
    with pytest.raises(ValueError):
        apply_mask_format(np.zeros((2, 2), dtype=np.uint8), mask_format="palette")


def test_normalize_image():
    arr_uint8 = np.array([[0, 255]], dtype=np.uint8)
    norm = normalize_image(arr_uint8)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0 and norm.max() <= 1.0

    arr_float = np.array([[0.0, 1.0]], dtype=np.float32)
    norm2 = normalize_image(arr_float)
    assert norm2.dtype == np.float32
    assert np.allclose(norm2, arr_float)


def test_to_grayscale_from_rgb():
    rgb = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    gray = to_grayscale(rgb)
    assert gray.ndim == 2
    assert gray.shape == (10, 10)


def test_to_grayscale_already_gray():
    gray = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    result = to_grayscale(gray)
    assert result.ndim == 2
    np.testing.assert_array_equal(result, gray)


def test_to_grayscale_single_channel():
    single = np.random.randint(0, 256, size=(10, 10, 1), dtype=np.uint8)
    result = to_grayscale(single)
    assert result.ndim == 2
    np.testing.assert_array_equal(result, single[:, :, 0])


def test_resize_pair_target_channels_1():
    """RGB image should be converted to grayscale when target_channels=1."""
    img = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)

    img_r, mask_r = resize_pair(img, mask, (32, 32), target_channels=1)

    assert img_r.ndim == 2
    assert img_r.shape == (32, 32)
    assert mask_r.shape == (32, 32)


def test_resize_pair_target_channels_3():
    """Grayscale image should be converted to 3 channels when target_channels=3."""
    img = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)

    img_r, mask_r = resize_pair(img, mask, (32, 32), target_channels=3)

    assert img_r.ndim == 3
    assert img_r.shape == (32, 32, 3)
    assert mask_r.shape == (32, 32)


def test_resize_pair_target_channels_3_handles_single_channel_input():
    img = np.random.randint(0, 256, size=(50, 50, 1), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)

    img_r, _ = resize_pair(img, mask, (32, 32), target_channels=3)

    assert img_r.shape == (32, 32, 3)


def test_resize_pair_target_channels_3_handles_bgra_input():
    img = np.random.randint(0, 256, size=(50, 50, 4), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)

    img_r, _ = resize_pair(img, mask, (32, 32), target_channels=3)

    assert img_r.shape == (32, 32, 3)

