"""Reusable convolutional blocks for U-Net."""

from __future__ import annotations

from typing import Optional

import keras
from keras import layers


def conv_block(
    x: keras.KerasTensor,
    filters: int,
    kernel_size: int = 3,
    activation: str = "relu",
    padding: str = "same",
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> keras.KerasTensor:
    """Apply two convolutions with optional batch norm and dropout."""

    for idx in range(2):
        conv_name = f"{name}_conv{idx + 1}" if name else None
        bn_name = f"{name}_bn{idx + 1}" if name else None
        act_name = f"{name}_act{idx + 1}" if name else None
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer="he_normal", name=conv_name)(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=bn_name)(x)
        x = layers.Activation(activation, name=act_name)(x)

    if dropout_rate > 0.0:
        drop_name = f"{name}_dropout" if name else None
        x = layers.Dropout(dropout_rate, name=drop_name)(x)

    return x


def encoder_block(
    x: keras.KerasTensor,
    filters: int,
    pool_size: int = 2,
    activation: str = "relu",
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Encoder block returning pooled output and skip connection."""

    block_name = f"{name}_conv" if name else None
    x = conv_block(
        x,
        filters,
        activation=activation,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        name=block_name,
    )
    skip = x
    pool_name = f"{name}_pool" if name else None
    x = layers.MaxPooling2D(pool_size=(pool_size, pool_size), name=pool_name)(x)
    return x, skip


def decoder_block(
    x: keras.KerasTensor,
    skip: keras.KerasTensor,
    filters: int,
    activation: str = "relu",
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
    use_transpose: bool = True,
    name: Optional[str] = None,
) -> keras.KerasTensor:
    """Decoder block with upsampling and skip concatenation."""

    up_name = f"{name}_up" if name else None
    if use_transpose:
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", kernel_initializer="he_normal", name=up_name)(x)
    else:
        x = layers.UpSampling2D(size=(2, 2), name=up_name)(x)
        conv_name = f"{name}_upconv" if name else None
        x = layers.Conv2D(filters, 2, padding="same", kernel_initializer="he_normal", name=conv_name)(x)

    concat_name = f"{name}_concat" if name else None
    x = layers.Concatenate(name=concat_name)([x, skip])

    block_name = f"{name}_conv" if name else None
    x = conv_block(
        x,
        filters,
        activation=activation,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        name=block_name,
    )
    return x