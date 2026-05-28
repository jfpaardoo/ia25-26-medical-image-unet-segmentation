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
    name: str = "unet",
) -> keras.Model:
    """Build and return the segmentation model."""

    if final_activation is None:
        final_activation = "sigmoid" if num_classes == 1 else "softmax"

    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    skips: list[keras.KerasTensor] = []
    filters = base_filters
    for idx in range(depth):
        x, skip = encoder_block(
            x,
            filters,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            name=f"enc{idx + 1}",
        )
        skips.append(skip)
        filters *= 2

    x = conv_block(
        x,
        filters,
        activation="relu",
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        name="bottleneck",
    )

    for idx in range(depth - 1, -1, -1):
        filters //= 2
        x = decoder_block(
            x,
            skips[idx],
            filters,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            name=f"dec{idx + 1}",
        )

    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation=final_activation,
        padding="same",
        name="segmentation_head",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)