"""Segmentation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import keras
from keras import ops


def _use_numpy(*values: Any) -> bool:
    return all(isinstance(value, np.ndarray) for value in values)


def _calculate_intersection_and_sums(y_true, y_pred):
    if _use_numpy(y_true, y_pred):
        y_true = np.asarray(y_true).astype(np.float32)
        y_pred = np.asarray(y_pred).astype(np.float32)
        intersection = np.sum(y_true * y_pred)
        sum_true_pred = np.sum(y_true) + np.sum(y_pred)
        return intersection, sum_true_pred, True

    y_true = ops.cast(ops.convert_to_tensor(y_true), "float32")
    y_pred = ops.cast(ops.convert_to_tensor(y_pred), "float32")
    intersection = ops.sum(y_true * y_pred)
    sum_true_pred = ops.sum(y_true) + ops.sum(y_pred)
    return intersection, sum_true_pred, False


@keras.saving.register_keras_serializable(package="segmentation")
def dice_coefficient(y_true, y_pred, epsilon: float = 1e-7):
    intersection, sum_true_pred, is_numpy = _calculate_intersection_and_sums(y_true, y_pred)
    res = (2.0 * intersection + epsilon) / (sum_true_pred + epsilon)
    return float(res) if is_numpy else res


@keras.saving.register_keras_serializable(package="segmentation")
def dice_loss(y_true, y_pred, epsilon: float = 1e-7):
    return 1.0 - dice_coefficient(y_true, y_pred, epsilon=epsilon)


@keras.saving.register_keras_serializable(package="segmentation")
def iou_score(y_true, y_pred, epsilon: float = 1e-7):
    intersection, sum_true_pred, is_numpy = _calculate_intersection_and_sums(y_true, y_pred)
    union = sum_true_pred - intersection
    res = (intersection + epsilon) / (union + epsilon)
    return float(res) if is_numpy else res


@keras.saving.register_keras_serializable(package="segmentation")
class DiceCoefficient(keras.metrics.Metric):
    """Dice coefficient as a streaming Keras metric."""

    def __init__(self, epsilon: float = 1e-7, name: str = "dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.denominator = self.add_weight(name="denominator", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(ops.convert_to_tensor(y_true), "float32")
        y_pred = ops.cast(ops.convert_to_tensor(y_pred), "float32")
        if sample_weight is not None:
            sample_weight = ops.cast(ops.convert_to_tensor(sample_weight), "float32")
            y_true = y_true * sample_weight
            y_pred = y_pred * sample_weight

        intersection = ops.sum(y_true * y_pred)
        denominator = ops.sum(y_true) + ops.sum(y_pred)
        self.intersection.assign_add(intersection)
        self.denominator.assign_add(denominator)

    def result(self):
        return (2.0 * self.intersection + self.epsilon) / (self.denominator + self.epsilon)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.denominator.assign(0.0)