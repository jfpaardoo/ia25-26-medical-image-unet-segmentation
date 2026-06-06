"""U-Net architecture definition."""

from __future__ import annotations

from typing import Optional

import keras
from keras import layers

from .blocks import conv_block, decoder_block, encoder_block

def build_unet(
    input_shape: tuple[int, int, int],
    num_classes: int = 1,
    base_filters: int = 32,
    depth: int = 4,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = True,
    final_activation: Optional[str] = None,
) -> keras.Model:
    """Build and return the segmentation model."""

    if final_activation is None:
        final_activation = "sigmoid" if num_classes == 1 else "softmax"

    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips: list[keras.KerasTensor] = []
    filters = base_filters

    for _ in range(depth):
        x, skip = encoder_block(x, filters, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)
        skips.append(skip)
        filters *= 2

    x = conv_block(x, filters, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)

    for skip in reversed(skips):
        filters //= 2
        x = decoder_block(x, skip, filters, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)

    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation=final_activation,
        padding="same",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs)