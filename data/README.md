Estructura de datos esperada para el proyecto

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
  - Preferible: imágenes en escala de grises donde cada pixel es la etiqueta (0 para fondo, 1..N para clases).
  - Si se usan PNG con paleta o colores, documentar en `configs/default.yaml` cómo mapear colores→clases.

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
- Si el dataset original requiere pasos adicionales (DICOM -> PNG, windowing), documentar el script de conversión en `data/README.md` y añadirlo a `scripts/`.
- Actualiza `configs/default.yaml` con `image_size`, `mask_format` y `seed`.
