"""Generate predictions for new medical images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import CONFIGS_DIR, PREDICTIONS_DIR, PROJECT_ROOT, load_yaml_config
from src.evaluation.inference import predict_mask


def _load_images(path: Path) -> np.ndarray:
    data = np.load(path)
    if "images" not in data:
        raise ValueError("NPZ file must contain an 'images' array.")
    return data["images"]


def _resolve_output_dir(config: dict, key: str, default: Path) -> Path:
    outputs = config.get("outputs", {})
    raw_path = outputs.get(key)
    if not raw_path:
        return default
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions for new medical images.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "default.yaml")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-probabilities", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    images = _load_images(args.images_npz)
    predictions_dir = _resolve_output_dir(config, "predictions_dir", PREDICTIONS_DIR)
    output_dir = args.output or predictions_dir
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

    for idx, mask in enumerate(masks):
        mask_img = (mask.squeeze() * 255).astype(np.uint8)
        file_path = output_dir / f"pred_{idx + 1:03d}.png"
        Image.fromarray(mask_img).save(file_path)
        
    print(f"Saved {len(masks)} prediction images to {output_dir}")


if __name__ == "__main__":
    main()