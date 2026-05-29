# Estructura de datos esperada para el proyecto

## Dataset utilizado: BUSI (Breast Ultrasound Images)

El proyecto emplea el **Breast Ultrasound Images Dataset (BUSI)**, publicado por Al-Dhabyani *et al.* (2020) y disponible en [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

- **Imágenes originales:** 780 ecografías mamarias en formato PNG (RGB).
- **Clases utilizadas:** benigno (437) y maligno (210) → **647 pares** imagen-máscara.
- **Clase excluida:** *normal* (133 imágenes con máscaras vacías, sin utilidad para segmentación).
- **Fusión de máscaras:** cuando una imagen tiene varias máscaras (`_mask.png`, `_mask_1.png`, …), se combinan con OR lógico en una única máscara binaria.
- **Conversión:** las imágenes RGB se convierten a escala de grises (`input_channels=1`) durante el preprocesado.

### Descarga y preparación

```bash
# Descargar el dataset desde Kaggle (usa kagglehub)
python -m pip install kagglehub
python scripts/download_busi.py

# Preprocesar y generar parches + particiones
python scripts/prepare_data.py
```

> Los archivos de datos (`data/raw/*`, `data/processed/*`, `data/splits/*`) están excluidos en `.gitignore` y deben regenerarse localmente tras clonar el repositorio.

### Organización resultante

```text
data/
├── raw/                  # Dataset BUSI descargado (647 imágenes + máscaras)
│   ├── images/           # benign_1.png … malignant_210.png
│   └── masks/            # mismos nombres, máscaras binarias 0/255
├── processed/            # Parches 256×256 listos para entrenamiento
│   ├── images/           # float32 escala de grises (.npy)
│   └── masks/            # uint8 binarias (.npy)
└── splits/               # Particiones train/val/test
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── summary.json
```

---


Objetivo
- Definir y fijar la estructura de `data/raw/` para pares imagen-máscara usados en segmentación.

Convenciones (contrato de datos)
- Raíz de datos crudos: `data/raw/`.
- Estructura preferida (mirrored):
  - `data/raw/images/`  -> contiene imágenes de entrada (RGB o escala de grises).
  - `data/raw/masks/`   -> contiene máscaras con la misma estructura y nombres que `images/`.
  - Se permite estructura de subcarpetas; la correspondencia se hace por ruta relativa.

- Alternativa soportada:
  - Imágenes y máscaras en la misma carpeta, con máscaras nombradas usando el sufijo `_mask`, p.ej. `imageA.png` y `imageA_mask.png`.

- Extensiones admitidas: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`.
- Formato de máscara:
  - El contrato del pipeline se fija en `configs/default.yaml` con `mask_format`.
  - Valores soportados por el preprocesado actual: `binary` y `grayscale`.
  - Si se usan PNG con paleta o colores, documentar en `configs/default.yaml` cómo mapear colores→clases antes de llamar a `resize_pair(..., mask_format=...)`.

- Tamaño de imágenes y máscaras:
  - No obligatorio en crudo; el pipeline de preprocesado deberá reescalar o extraer parches según `configs/default.yaml`.

- Requisitos mínimos para integración con el pipeline:
  - `discover_samples(root)` debe devolver pares `(image_path, mask_path)` para el `root=data/raw`.
  - El pipeline asumirá que las máscaras están alineadas espacialmente con las imágenes; si no, añadir una nota en este README.

Ejemplo de uso

>>> from pathlib import Path
>>> from src.data.dataset import discover_samples
>>> samples = discover_samples(Path('data/raw'))
>>> for s in samples:
...     print(s.image_path, '->', s.mask_path)

Notas
- El dataset BUSI ya se distribuye en formato PNG, por lo que no requiere conversión adicional (DICOM → PNG, windowing, etc.).
- El script `scripts/download_busi.py` descarga el dataset automáticamente mediante `kagglehub`. Se necesita tener configuradas las credenciales de Kaggle.
- El script `scripts/prepare_data.py` ejecuta el pipeline completo: fusión de máscaras, conversión a escala de grises, normalización, parcheado 256×256, aumento de datos y generación de particiones.
- `configs/default.yaml` fija `image_size`, `mask_format` y `seed`.
