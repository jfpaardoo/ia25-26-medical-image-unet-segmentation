"""Reusable convolutional blocks for U-Net."""

from __future__ import annotations

import keras
from keras import layers

def conv_block(x: keras.KerasTensor, filters: int, kernel_size: int = 3, activation: str = "relu", padding: str = "same", use_batch_norm: bool = True, dropout_rate: float = 0.0) -> keras.KerasTensor:
    """Apply two convolutions with optional batch norm and dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer="he_normal")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        # Se utiliza ReLU (Rectified Linear Unit) como función de activación en las capas
        # ocultas porque introduce no linealidad en la red y ayuda a mitigar el problema 
        # del desvanecimiento del gradiente (vanishing gradient) que sufrirían redes profundas con Sigmoide.
        x = layers.Activation(activation)(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)
    return x

def encoder_block(x: keras.KerasTensor, filters: int, pool_size: int = 2, **kwargs) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Encoder block returning pooled output and skip connection."""
    x = conv_block(x, filters, **kwargs)
    return layers.MaxPooling2D(pool_size=(pool_size, pool_size))(x), x

def decoder_block(x: keras.KerasTensor, skip: keras.KerasTensor, filters: int, **kwargs) -> keras.KerasTensor:
    """Decoder block with upsampling and skip concatenation."""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", kernel_initializer="he_normal")(x)
    x = layers.Concatenate()([x, skip])
    return conv_block(x, filters, **kwargs)