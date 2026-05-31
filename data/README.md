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
    └── masks_expert2/
```

Solo `training/masks/` contiene ground truth de vasos. El split `test/` se conserva para inspección o inferencia, pero no genera muestras supervisadas para entrenamiento.

## Flujo de preparación

```bash
python -m pip install kagglehub
python scripts/download_drive.py
python scripts/prepare_data.py
python scripts/validate_data_raw.py
```

El preprocesado convierte las imágenes a escala de grises, normaliza, genera parches de 256×256 y escribe las particiones en `data/splits/`.

## Contrato del pipeline

- `discover_samples(Path('data/raw'))` debe encontrar pares válidos en `training/images/` y `training/masks/`.
- Las máscaras deben estar alineadas espacialmente con sus imágenes.
- `configs/default.yaml` controla `image_size`, `patch_size`, `patch_stride`, `mask_format` y la semilla.
- Los directorios `data/raw/`, `data/processed/` y `data/splits/` no se versionan y deben regenerarse localmente.
