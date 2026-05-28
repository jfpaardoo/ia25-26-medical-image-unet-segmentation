"""Training entry points and orchestration helpers."""

from __future__ import annotations


def train_model(*args, **kwargs):
    raise NotImplementedError("Implement the training loop, callbacks and checkpointing here.")