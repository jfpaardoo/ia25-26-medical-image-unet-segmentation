"""Generar predicciones para imágenes completas reconstruyendo los parches."""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.config import CONFIGS_DIR, PROJECT_ROOT, load_yaml_config
from src.evaluation.inference import load_model, predict_mask
from src.data.preprocessing import normalize_image, to_grayscale
from src.data.patching import extract_patches, reconstruct_from_patches

def main():
    parser = argparse.ArgumentParser(description="Predecir máscaras para imágenes completas.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "default.yaml")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "data/raw/test/images")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/predictions_full")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    patch_size = config.get("data", {}).get("patch_size", [128, 128])
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando modelo desde {args.model}...")
    model = load_model(args.model)
    
    image_paths = list(args.images_dir.glob("*.tif")) + list(args.images_dir.glob("*.png"))
    if not image_paths:
        print(f"No se encontraron imágenes en {args.images_dir}")
        return
        
    print(f"Se van a procesar {len(image_paths)} imágenes completas...")
    
    for img_path in image_paths:
        print(f" -> Reconstruyendo {img_path.name}...")
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        if img_raw is None:
            print(f"[WARN] No se pudo leer la imagen: {img_path}")
            continue
            
        # Convertimos a escala de grises y normalizamos a [0, 1]
        img_gray = to_grayscale(img_raw)
        img_processed = normalize_image(img_gray)
        img_processed = img_processed[..., None]
        dummy_mask = np.zeros(img_raw.shape[:2], dtype=np.uint8)[..., None]
        
        # Extraemos parches con solapamiento de la mitad para evitar bordes duros
        stride = [max(1, patch_size[0] // 2), max(1, patch_size[1] // 2)]
        image_patches, _, positions, _ = extract_patches(
            img_processed, 
            dummy_mask, 
            patch_size=patch_size, 
            stride=stride
        )
        
        # Predecimos la máscara para todos los parches de esta imagen
        _, probabilities = predict_mask(model, image_patches, threshold=args.threshold, return_probabilities=True)
        
        # Unimos el puzzle usando probabilidades y luego aplicamos umbral
        rebuilt_prob = reconstruct_from_patches(probabilities, positions, output_shape=img_raw.shape[:2])
        rebuilt_mask = (rebuilt_prob >= args.threshold).astype(np.uint8)
        
        # Guardamos la imagen final completa
        rebuilt_img = (rebuilt_mask.squeeze() * 255).astype(np.uint8)
        out_file = args.output / f"{img_path.stem}_pred.png"
        Image.fromarray(rebuilt_img).save(out_file)
        
    print(f"¡Listo! Se guardaron {len(image_paths)} fotos reconstruidas enteras en {args.output}")

if __name__ == "__main__":
    main()
