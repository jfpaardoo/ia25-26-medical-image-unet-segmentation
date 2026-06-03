# Estructura de datos del proyecto

## Dataset utilizado

El proyecto usa **DRIVE 2004** (Digital Retinal Images for Vessel Extraction), disponible en Kaggle. El conjunto contiene 40 imágenes de fondo de ojo: 20 en `training/` y 20 en `test/`.

## Organización esperada

El script [`scripts/download_drive.py`](../scripts/download_drive.py) deja `data/raw/` con esta estructura:

```text
data/raw/
├── training/
│   ├── images/
│   ├── masks/
│   └── fov_masks/
└── test/
    ├── images/
    ├── fov_masks/
    ├── masks_expert1/
    └── masks_expert2/
```

`training/masks/` contiene el ground truth usado para entrenar. En `test/` se conservan las dos segmentaciones manuales de expertos y se usan para evaluación comparando la predicción contra ambas máscaras por separado y promediando el DICE final.

## Flujo de preparación

```bash
python -m pip install kagglehub
python scripts/download_drive.py
python scripts/prepare_data.py
```

El preprocesado convierte las imágenes a escala de grises, normaliza, genera parches de 256×256 y escribe las particiones en `data/splits/`. La evaluación del split `test/` debe leer las dos máscaras expertas y calcular la media de sus métricas.

## Contrato del pipeline

- `discover_samples(Path('data/raw'))` debe encontrar pares válidos en `training/images/` y `training/masks/`.
- Las máscaras deben estar alineadas espacialmente con sus imágenes.
- `configs/default.yaml` controla `image_size`, `patch_size`, `patch_stride`, `mask_format` y la semilla.
- Los directorios `data/raw/`, `data/processed/` y `data/splits/` no se versionan y deben regenerarse localmente.
