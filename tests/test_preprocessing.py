import numpy as np

from src.data.preprocessing import resize_pair, normalize_image


def test_resize_and_mask_alignment():
    img = np.random.randint(0, 256, size=(100, 120, 3), dtype=np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[::10, ::10] = 1

    img_r, mask_r = resize_pair(img, mask, (64, 64))

    assert img_r.shape == (64, 64, 3)
    assert mask_r.shape == (64, 64)
    # mask values should remain labels (0 or 1)
    assert set(np.unique(mask_r)).issubset({0, 1})


def test_normalize_image():
    arr_uint8 = np.array([[0, 255]], dtype=np.uint8)
    norm = normalize_image(arr_uint8)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0 and norm.max() <= 1.0

    arr_float = np.array([[0.0, 1.0]], dtype=np.float32)
    norm2 = normalize_image(arr_float)
    assert norm2.dtype == np.float32
    assert np.allclose(norm2, arr_float)
