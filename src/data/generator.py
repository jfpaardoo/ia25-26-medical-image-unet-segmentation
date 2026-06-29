"""DataGenerator para cargar y procesar los datos bajo demanda (on-the-fly) durante el entrenamiento."""

from __future__ import annotations

import cv2
import numpy as np
import keras
from pathlib import Path

from src.data.dataset import SegmentationSample
from src.data.preprocessing import to_grayscale, normalize_image, apply_mask_format

class DataGenerator(keras.utils.Sequence):
    """Generador de datos para Keras que lee de disco, parchea y aumenta imágenes sobre la marcha.
    
    Al heredar de keras.utils.Sequence, garantizamos un funcionamiento seguro en
    multiprocesamiento (multiprocessing) al entrenar.
    """
    
    def __init__(
        self,
        samples: list[SegmentationSample],
        batch_size: int = 16,
        patch_size: tuple[int, int] = (128, 128),
        augment: bool = False,
        shuffle: bool = True,
        seed: int = 42
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        
        # Como DRIVE tiene sólo 20 imágenes de train, caben perfectamente en memoria.
        self.images_cache = []
        self.masks_cache = []
        
        for sample in self.samples:
            img = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
            
            if img is None or mask is None:
                continue
            
            # Reutilizamos las funciones de preprocessing.py para realizar la normalización
            img = normalize_image(to_grayscale(img))[..., np.newaxis]
            mask = apply_mask_format(mask, "binary")[..., np.newaxis]
            
            self.images_cache.append(img)
            self.masks_cache.append(mask)
            
        # Generar un índice virtual de parches
        # Por cada imagen, generaremos 'patches_per_image' aleatorios por epoch
        self.patches_per_image = 50 
        self.num_total_patches = len(self.images_cache) * self.patches_per_image
        self.indices = np.arange(self.num_total_patches)
        self.on_epoch_end()

    def __len__(self) -> int:
        """Número de batches por epoch."""
        return int(np.floor(self.num_total_patches / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Generar un batch de datos."""
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        X = np.empty((self.batch_size, *self.patch_size, 1), dtype=np.float32)
        y = np.empty((self.batch_size, *self.patch_size, 1), dtype=np.uint8)
        
        for i, idx in enumerate(batch_indices):
            # Seleccionar una imagen aleatoria (simplificación)
            img_idx = idx % len(self.images_cache)
            img = self.images_cache[img_idx]
            mask = self.masks_cache[img_idx]
            
            # Extraer parche, hacer padding y aumentar si procede
            img_patch, mask_patch = self._extract_random_patch(img, mask)
            img_patch, mask_patch = self._pad_if_needed(img_patch, mask_patch)
            
            if self.augment:
                img_patch, mask_patch = self._apply_augmentation(img_patch, mask_patch)
                    
            X[i,] = img_patch
            y[i,] = mask_patch
            
        return X, y

    def _extract_random_patch(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extrae un recorte aleatorio de la imagen del tamaño del parche objetivo."""
        h, w = img.shape[:2]
        ph, pw = self.patch_size
        
        max_y = h - ph
        max_x = w - pw
        
        y_start = self.rng.integers(0, max_y + 1) if max_y > 0 else 0
        x_start = self.rng.integers(0, max_x + 1) if max_x > 0 else 0
        
        return img[y_start:y_start+ph, x_start:x_start+pw], mask[y_start:y_start+ph, x_start:x_start+pw]

    def _pad_if_needed(self, img_patch: np.ndarray, mask_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Añade píxeles extra (padding) si la imagen original es más pequeña que el parche deseado."""
        ph, pw = self.patch_size
        if img_patch.shape[0] < ph or img_patch.shape[1] < pw:
            pad_y = max(0, ph - img_patch.shape[0])
            pad_x = max(0, pw - img_patch.shape[1])
            img_patch = np.pad(img_patch, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
            mask_patch = np.pad(mask_patch, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
        return img_patch, mask_patch

    def _apply_augmentation(self, img_patch: np.ndarray, mask_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Aplica aumentos de datos aleatorios (volteo horizontal y vertical)."""
        if self.rng.random() > 0.5:
            img_patch = np.fliplr(img_patch).copy()
            mask_patch = np.fliplr(mask_patch).copy()
        if self.rng.random() > 0.5:
            img_patch = np.flipud(img_patch).copy()
            mask_patch = np.flipud(mask_patch).copy()
        return img_patch, mask_patch

    def on_epoch_end(self) -> None:
        """Se ejecuta al final de cada epoch."""
        if self.shuffle:
            self.rng.shuffle(self.indices)
