"""Train the U-Net segmentation model using DataGenerator and Cross Validation."""

from __future__ import annotations

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path

import numpy as np
from keras.utils import set_random_seed
from sklearn.model_selection import KFold

from src.config import CONFIGS_DIR, FINAL_MODELS_DIR, PROJECT_ROOT, load_yaml_config
from src.training.train import train_model
from src.data.dataset import discover_drive_samples
from src.data.generator import DataGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the U-Net segmentation model.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "default.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    # Descubrir las imágenes de entrenamiento (20 imágenes en DRIVE 2004)
    train_samples = discover_drive_samples(args.data_dir, split="training")
    if not train_samples:
        print("No se encontraron imágenes en el directorio de entrenamiento. ¿Descargaste el dataset?")
        return

    seed = config.get("project", {}).get("seed", 42)
    set_random_seed(seed)
    
    # K-Fold Cross Validation con k=5 (recomendado por el profesor)
    # Como tenemos 20 imágenes, 4 serán para validación y 16 para entrenamiento en cada fold
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    final_models_dir = config.get("outputs", {}).get("final_model_dir", FINAL_MODELS_DIR)
    if not isinstance(final_models_dir, Path):
        final_models_dir = Path(final_models_dir)
    final_models_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config.get("training", {}).get("batch_size", 16)
    patch_size = tuple(config.get("data", {}).get("patch_size", [128, 128]))

    fold = 1
    # Convertimos a np.array de objetos para usar los índices del KFold fácilmente
    samples_array = np.array(train_samples)
    
    for train_idx, val_idx in kf.split(samples_array):
        print(f"--- Training Fold {fold} ---")
        
        train_samples_fold = samples_array[train_idx].tolist()
        val_samples_fold = samples_array[val_idx].tolist()
        
        # Instanciar generadores para este fold
        train_gen = DataGenerator(
            samples=train_samples_fold,
            batch_size=batch_size,
            patch_size=patch_size,
            augment=True,
            shuffle=True,
            seed=seed + fold
        )
        
        val_gen = DataGenerator(
            samples=val_samples_fold,
            batch_size=batch_size,
            patch_size=patch_size,
            augment=False,
            shuffle=False,
            seed=seed
        )

        model, _ = train_model(train_data=train_gen, val_data=val_gen, config=config)

        model_path = final_models_dir / f"unet_fold_{fold}.keras"
        model.save(model_path)
        print(f"Saved model for fold {fold} at {model_path}")
        fold += 1


if __name__ == "__main__":
    main()