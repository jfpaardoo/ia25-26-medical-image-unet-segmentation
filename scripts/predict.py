"""Generate predictions for new medical images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import CONFIGS_DIR, PREDICTIONS_DIR, load_yaml_config, ensure_output_dirs
from src.evaluation.inference import predict_mask


def _load_images(path: Path) -> np.ndarray:
    data = np.load(path)
    if "images" not in data:
        raise ValueError("NPZ file must contain an 'images' array.")
    return data["images"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions for new medical images.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "default.yaml")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-probabilities", action="store_true")
    args = parser.parse_args()

    _ = load_yaml_config(args.config)
    ensure_output_dirs()

    images = _load_images(args.images_npz)
    output_dir = args.output or PREDICTIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    result = predict_mask(
        args.model,
        images,
        threshold=args.threshold,
        return_probabilities=args.save_probabilities,
    )

    if args.save_probabilities:
        masks, probabilities = result
        np.savez_compressed(output_dir / "predictions.npz", masks=masks, probabilities=probabilities)
    else:
        masks = result
        np.savez_compressed(output_dir / "predictions.npz", masks=masks)


if __name__ == "__main__":
    main()