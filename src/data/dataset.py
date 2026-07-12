"""Funciones para cargar las rutas de las imágenes de la base de datos DRIVE 2004."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SegmentationSample:
    """Par que contiene la ruta a la imagen y su máscara (ground truth)."""
    image_path: Path
    mask_path: Path

def discover_drive_samples(root_dir: Path | str, split: str = "training") -> list[SegmentationSample]:
    """Descubre los archivos de imagen y máscara del dataset DRIVE 2004.
    
    El dataset DRIVE 2004 está estructurado de la siguiente forma:
    - training/
        - images/ (imágenes originales .tif)
        - masks/ (máscaras de segmentación .png)
    - test/
        - images/ (imágenes originales .tif)
        - masks_expert1/ (máscaras de segmentación del primer experto .png)
        - masks_expert2/ (máscaras del segundo experto - opcional)
    
    Args:
        root_dir: Directorio base de los datos (ej: data/raw)
        split: 'training' o 'test'
        
    Returns:
        Una lista de objetos SegmentationSample con las rutas encontradas.
    """
    root = Path(root_dir)
    images_dir = root / split / "images"
    
    masks_dir_name = "masks_expert1" if split == "test" else "masks"
    masks_dir = root / split / masks_dir_name
    mask_ext = ".png"
    
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(
            f"No se encuentra el dataset o las carpetas necesarias en {root}/{split}. "
            f"Asegúrate de haber descargado el dataset y de tener '{images_dir.name}' y '{masks_dir.name}'."
        )
        
    samples = []
    # Las imágenes en DRIVE son .tif
    for img_path in sorted(images_dir.glob("*.tif")):
        mask_name = f"{img_path.stem}{mask_ext}"
            
        mask_path = masks_dir / mask_name
        
        if mask_path.exists():
            samples.append(SegmentationSample(image_path=img_path, mask_path=mask_path))
        else:
            print(f"[WARN] No se encontró máscara para la imagen {img_path.name}: se esperaba {mask_name}")
            
    return samples