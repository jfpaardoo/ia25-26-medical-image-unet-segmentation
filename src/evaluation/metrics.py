"""Segmentation metrics."""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="segmentation")
def dice_coefficient(y_true, y_pred, epsilon: float = 1e-7):
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    intersection = ops.sum(y_true * y_pred)
    sum_true_pred = ops.sum(y_true) + ops.sum(y_pred)
    return (2.0 * intersection + epsilon) / (sum_true_pred + epsilon)


@keras.saving.register_keras_serializable(package="segmentation")
def dice_loss(y_true, y_pred, epsilon: float = 1e-7):
    return 1.0 - dice_coefficient(y_true, y_pred, epsilon=epsilon)


@keras.saving.register_keras_serializable(package="segmentation")
def iou_score(y_true, y_pred, epsilon: float = 1e-7):
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    intersection = ops.sum(y_true * y_pred)
    union = ops.sum(y_true) + ops.sum(y_pred) - intersection
    return (intersection + epsilon) / (union + epsilon)


@keras.saving.register_keras_serializable(package="segmentation")
def sensitivity(y_true, y_pred, epsilon: float = 1e-7):
    """Sensitivity (Recall): TP / (TP + FN)."""
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    true_positives = ops.sum(y_true * y_pred)
    possible_positives = ops.sum(y_true)
    return (true_positives + epsilon) / (possible_positives + epsilon)


@keras.saving.register_keras_serializable(package="segmentation")
def specificity(y_true, y_pred, epsilon: float = 1e-7):
    """Specificity: TN / (TN + FP)."""
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    true_negatives = ops.sum((1.0 - y_true) * (1.0 - y_pred))
    possible_negatives = ops.sum(1.0 - y_true)
    return (true_negatives + epsilon) / (possible_negatives + epsilon)


@keras.saving.register_keras_serializable(package="segmentation")
class DiceCoefficient(keras.metrics.Metric):
    """Dice coefficient as a streaming Keras metric."""

    def __init__(self, epsilon: float = 1e-7, name: str = "dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.denominator = self.add_weight(name="denominator", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, "float32")
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