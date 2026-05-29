"""Visualization helpers for images, masks and predictions."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt


def _prepare_for_display(array: np.ndarray, is_label: bool) -> tuple[np.ndarray, Optional[str]]:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0], "gray"
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        return array, None
    if array.ndim == 3 and array.shape[-1] > 1 and is_label:
        return np.argmax(array, axis=-1), "gray"
    if array.ndim == 3 and array.shape[-1] > 1:
        return array[..., 0], "gray"
    return array, "gray"


def plot_sample(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    titles: Optional[Iterable[str] | dict[str, str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figure_size: tuple[int, int] = (12, 4),
):
    """Plot an image with optional mask and prediction side by side."""

    panels: list[tuple[str, np.ndarray, bool]] = [("Image", image, False)]
    if mask is not None:
        panels.append(("Mask", mask, True))
    if prediction is not None:
        panels.append(("Prediction", prediction, True))

    total = len(panels)
    if total == 0:
        raise ValueError("At least one panel is required for plotting.")

    fig, axes = plt.subplots(1, total, figsize=figure_size, squeeze=False)
    axes = axes[0]

    for idx, (label, array, is_label) in enumerate(panels):
        display, cmap = _prepare_for_display(array, is_label=is_label)
        axes[idx].imshow(display, cmap=cmap)
        axes[idx].axis("off")

        title = label
        if isinstance(titles, dict):
            title = titles.get(label, label)
        elif isinstance(titles, (list, tuple)) and idx < len(titles):
            title = titles[idx]
        axes[idx].set_title(title)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes