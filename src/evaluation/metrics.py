"""Segmentation metrics."""

from __future__ import annotations

import numpy as np


def dice_coefficient(y_true, y_pred, epsilon: float = 1e-7):
    y_true = np.asarray(y_true).astype(np.float32)
    y_pred = np.asarray(y_pred).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    return float((2.0 * intersection + epsilon) / (denominator + epsilon))


def iou_score(y_true, y_pred, epsilon: float = 1e-7):
    y_true = np.asarray(y_true).astype(np.float32)
    y_pred = np.asarray(y_pred).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + epsilon) / (union + epsilon))