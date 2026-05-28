"""Evaluate a trained segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import CONFIGS_DIR, load_yaml_config
from src.evaluation.inference import evaluate_model


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "images" not in data or "masks" not in data:
        raise ValueError("NPZ file must contain 'images' and 'masks' arrays.")
    return data["images"], data["masks"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "default.yaml")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    _ = load_yaml_config(args.config)
    images, masks = _load_npz(args.data_npz)
    results = evaluate_model(args.model, images, masks, threshold=args.threshold)

    for name, value in results.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()