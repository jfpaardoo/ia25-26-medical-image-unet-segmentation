"""Download and organize the DRIVE 2004 dataset."""

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
# Usamos el ID exacto que has proporcionado
KAGGLE_ID = "andrewmvd/drive-digital-retinal-images-for-vessel-extraction"


def _binarize_and_save(src_path: Path, dst_path: Path) -> None:
    """Abre la máscara .gif esperada, la binariza a 0/255 y la guarda como .png."""
    if not src_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    mask_array = (np.array(Image.open(src_path)) > 0).astype(np.uint8) * 255
    Image.fromarray(mask_array).save(dst_path)


def main() -> None:
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub no está instalado. Ejecuta: python -m pip install kagglehub")

    print(f"Descargando el dataset DRIVE desde {KAGGLE_ID}...")
    dataset_path = Path(kagglehub.dataset_download(KAGGLE_ID))
    
    # Encontramos la carpeta raíz que contiene 'training' y 'test'
    root = next(p for p in dataset_path.rglob("*") if (p / "training").is_dir())

    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)

    total_images = 0

    for split in ["training", "test"]:
        src_split = root / split
        dst_split = RAW_DIR / split
        
        # Iteramos solo sobre los archivos .tif esperados
        for img_path in (src_split / "images").glob("*.tif"):
            prefix = img_path.name[:2]  # Ej: '01'
            stem = img_path.stem        # Ej: '01_training' o '01_test'
            
            # 1. Copiar imagen original (.tif)
            img_dst = dst_split / "images" / img_path.name
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, img_dst)
            
            # 2. Máscara FOV esperada: "01_training_mask.gif" o "01_test_mask.gif"
            fov_src = src_split / "mask" / f"{prefix}_{split}_mask.gif"
            _binarize_and_save(fov_src, dst_split / "fov_masks" / f"{stem}.png")
            
            # 3. Primera máscara manual esperada: "01_manual1.gif"
            primary_dir = "masks_expert1" if split == "test" else "masks"
            primary_src = src_split / "1st_manual" / f"{prefix}_manual1.gif"
            _binarize_and_save(primary_src, dst_split / primary_dir / f"{stem}.png")
            
            # 4. Segunda máscara manual esperada (solo en test): "01_manual2.gif"
            if split == "test":
                secondary_src = src_split / "2nd_manual" / f"{prefix}_manual2.gif"
                _binarize_and_save(secondary_src, dst_split / "masks_expert2" / f"{stem}.png")
            
            total_images += 1

    print(f"\nÉxito: {total_images} imágenes procesadas estrictamente bajo el formato DRIVE y guardadas en {RAW_DIR}")


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()