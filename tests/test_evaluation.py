from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.evaluate import _load_npz
from src.evaluation.inference import evaluate_model


def test_evaluate_model_averages_multiple_reference_masks(monkeypatch):
    images = np.zeros((2, 4, 4, 1), dtype=np.float32)
    pred_masks = np.ones((2, 4, 4, 1), dtype=np.uint8)
    reference_masks = [np.ones((2, 4, 4, 1), dtype=np.uint8), np.zeros((2, 4, 4, 1), dtype=np.uint8)]

    monkeypatch.setattr("src.evaluation.inference.predict_mask", lambda *args, **kwargs: pred_masks)

    results = evaluate_model(object(), images, reference_masks, metrics=[lambda y_true, y_pred: float(np.mean(y_true))])

    assert results["<lambda>"] == 0.5


def test_load_npz_prefers_drive_expert_masks(tmp_path: Path):
    path = tmp_path / "drive_eval.npz"
    images = np.zeros((2, 8, 8, 1), dtype=np.float32)
    masks_expert1 = np.ones((2, 8, 8), dtype=np.uint8)
    masks_expert2 = np.zeros((2, 8, 8), dtype=np.uint8)
    np.savez(path, images=images, masks_expert1=masks_expert1, masks_expert2=masks_expert2)

    loaded_images, loaded_masks = _load_npz(path)

    assert loaded_images.shape == images.shape
    assert isinstance(loaded_masks, tuple)
    assert len(loaded_masks) == 2
    assert loaded_masks[0].shape == (2, 8, 8, 1)
    assert loaded_masks[1].shape == (2, 8, 8, 1)
