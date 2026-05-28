"""Reusable convolutional blocks for U-Net."""

from __future__ import annotations


def conv_block(*args, **kwargs):
    raise NotImplementedError("Implement the convolutional block used by the U-Net encoder and decoder.")