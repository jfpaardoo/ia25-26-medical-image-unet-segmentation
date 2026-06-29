"""Segmentation metrics — using native Keras components wherever possible."""

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


@keras.saving.register_keras_serializable(package="segmentation")
class Specificity(keras.metrics.Metric):
    """Specificity = TN / (TN + FP)"""

    def __init__(self, name: str = "specificity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_negatives = keras.metrics.TrueNegatives()
        self.false_positives = keras.metrics.FalsePositives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        tn = self.true_negatives.result()
        fp = self.false_positives.result()
        return tn / (tn + fp + keras.backend.epsilon())

    def reset_state(self):
        self.true_negatives.reset_state()
        self.false_positives.reset_state()
