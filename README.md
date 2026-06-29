# ia25-26-medical-image-unet-segmentation

Repositorio base para el trabajo de segmentación de imágenes médicas con redes convolucionales y arquitectura U-Net.

## Alcance del proyecto

La documentación del repositorio y el enunciado de [docs/proyecto/segmentación_imágenes_médicas.pdf](docs/proyecto/segmentación_imágenes_médicas.pdf) fijan un objetivo claro: construir, entrenar y evaluar un modelo de segmentación médica capaz de generar máscaras con el mismo formato y tamaño que la imagen original, utilizando Keras 3 y TensorFlow.

Los documentos revisados implican, como mínimo, estas piezas de trabajo:

- Preprocesado de imágenes: normalización, padding y troceado en parches cuando sea necesario.
- Aumento de datos para conjuntos pequeños.
- Implementación de una U-Net con camino encoder/decoder y conexiones de salto.
- Entrenamiento reproducible, con validación y guardado del modelo en formato `.keras`.
- Evaluación con métricas de segmentación, especialmente DICE score.
- En DRIVE, comparación del test contra las dos máscaras expertas y promedio final de ambas métricas.
- Generación de resultados visuales y un informe final con formato de artículo.
- Trazabilidad del experimento: configuración reproducible, checkpoints y modelo final en `.keras`.

### Alineación con el enunciado

El enunciado del proyecto exige además varios puntos concretos que conviene tener presentes:

- Usar la base de datos DRIVE 2004 en la convocatoria de junio/julio.
- Trabajar con validación cruzada con al menos 5 pliegues.
- Implementar DICE score como métrica principal y aspirar a una media mínima de 0.75.
- Entregar segmentaciones generadas por el modelo en formato `.png`.
- Devolver salidas con el mismo tamaño que la imagen original tras el postprocesado.
- Aplicar padding o recorte cuando la forma de entrada no coincida con la que admite la U-Net.
- Manejar el preprocesado de forma reproducible; el repositorio implementa un `DataGenerator` que parchea y normaliza las imágenes sobre la marcha, alineado con la sugerencia del enunciado.

## Referencias analizadas

- [docs/proyecto/segmentación_imágenes_médicas.pdf](docs/proyecto/segmentación_imágenes_médicas.pdf)
- [docs/teoria/Redes_neuronales_(Contenido_teórico).pdf](docs/teoria/Redes_neuronales_(Contenido_teórico).pdf)
- [notebooks/Keras.ipynb](notebooks/Keras.ipynb)
- [DRIVE Digital Retinal Images for Vessel Extraction en Kaggle](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction)

### Resumen técnico de la documentación

- El PDF de segmentación define la entrega: modelo U-Net, data augmentation, padding, parcheado, métricas DICE, exportación en `.keras`, informe y presentación.
- El PDF teórico de redes neuronales aporta el marco matemático: perceptrón, descenso del gradiente, backpropagation, funciones de activación y criterios de salida según la tarea.
- El notebook `Keras.ipynb` muestra el estilo de trabajo esperado: notebooks educativos, reproducibilidad, `Sequential`, normalización, preprocesado con `scikit-learn` y ejemplos de clasificación/regresión con Keras.

## Estructura

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── default.json
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── docs/
│   ├── README.md
│   ├── proyecto/
│   │   ├── memoria.tex
│   │   ├── plantilla-trabajo.pdf
│   │   └── segmentación_imágenes_médicas.pdf
│   └── teoria/
│       └── Redes_neuronales_(Contenido_teórico).pdf
├── notebooks/
│   ├── README.md
│   └── Keras.ipynb
├── scripts/
├── src/
│   ├── config.py
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
├── models/
│   ├── checkpoints/
│   └── final/
├── artifacts/
│   ├── figures/
│   ├── logs/
│   ├── npz/
│   ├── predictions/
│   └── predictions_full/
└── tests/
```

## Flujo de trabajo previsto

1. Descargar y organizar el dataset DRIVE 2004 mediante `python scripts/download_drive.py` — binariza máscaras `.gif` a `.png` y las deposita directamente en `data/raw/`.
2. Entrenar la U-Net desde `scripts/train.py`, que ahora usa un `DataGenerator` (Keras `Sequence`) para cargar, parchear y aumentar los datos sobre la marcha sin empaquetado previo en `.npz`.
3. Guardar pesos, checkpoints y modelo final en `models/`.
4. Evaluar el modelo con DICE como métrica principal — el evaluador busca automáticamente las máscaras de ambos expertos con patrones de nombre flexibles.
5. Exportar las máscaras finales en `.png` y guardar figuras y predicciones en `artifacts/`.

## Trabajo en paralelo

La división está pensada para que cada persona pueda avanzar casi de forma independiente y solo compartir una interfaz común: estructura del dataset, nombre de ficheros, configuración y formato de salida.

### Juan

Responsable de la parte de datos y del preprocesado.

- Definir el formato de entrada del dataset y documentar cómo debe colocarse en `data/raw/`.
- Implementar la detección de pares imagen-máscara en `src/data/dataset.py`.
- Implementar limpieza, normalización, padding, redimensionado y parcheado en `src/data/preprocessing.py` y `src/data/patching.py`.
- Diseñar la estrategia de aumento de datos y la generación de particiones en `data/splits/`.
- Preparar un notebook de exploración inicial si hace falta validar el dataset.
- Entregar al final un flujo reproducible que produzca entradas listas para entrenamiento.

### Diego

Responsable de la parte de modelo, entrenamiento, evaluación y resultados.

- Implementar la arquitectura U-Net en `src/models/unet.py` y los bloques reutilizables en `src/models/blocks.py`.
- Implementar las métricas de segmentación y el proceso de evaluación en `src/evaluation/`.
- Crear el entrenamiento, callbacks y guardado del modelo en `src/training/` y `scripts/train.py`.
- Implementar inferencia y exportación de máscaras en `scripts/predict.py`.
- Preparar las figuras finales y la comparación de resultados en `artifacts/figures/` y `artifacts/predictions/`.
- Entregar un modelo final cargable con `load_model` y un informe técnico de resultados.

### Contrato mínimo entre ambos

1. Misma estructura de carpetas dentro de `data/raw/`, `data/processed/` y `data/splits/`.
2. Mismo tamaño de entrada de imagen y máscara definido en `configs/default.json`.
3. Mismo criterio de codificación de máscaras binaria o multiclase.
4. Misma definición de métrica principal, con DICE como referencia mínima.
5. Misma convención para nombres de checkpoints, modelo final y figuras.

### Ritmo de integración

- Primera integración: Juan entrega un lote pequeño de datos ya preparados y Diego valida que el entrenamiento arranca.
- Segunda integración: Diego entrega una U-Net funcional con entrada/salida acordadas y Juan valida que sus datos encajan sin cambios manuales.
- Tercera integración: ambos prueban evaluación, guardado del modelo y generación de máscaras sobre ejemplos reales.

Para mantener la dependencia mínima, cualquier cambio de formato debe pasar primero por `configs/default.json` y quedar reflejado en este README o en `docs/README.md`.

## Instalación rápida

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Si trabajas en PowerShell y `conda` no está inicializado en la sesión, arranca primero la shell de Conda y luego activa el entorno:

```powershell
& 'C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1'
conda activate ia25-medseg
```

## Comandos principales

Estos son los comandos que cubren el flujo completo del proyecto según el enunciado del PDF y la implementación actual.

Importante: ejecútalos desde la raíz del repositorio y usa siempre la ruta `scripts/...`. Los ficheros `train.py`, `evaluate.py`, `predict.py` y `download_drive.py` no están en la raíz.

```powershell
# 1. Preparar el entorno
& 'C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1'
conda activate ia25-medseg
python -m pip install -r requirements.txt
python -m pip install -e .

# 2. Descargar y organizar el dataset DRIVE 2004
python scripts/download_drive.py

# 3. Entrenar la U-Net (DataGenerator + 5-fold CV)
python scripts/train.py

# 4. Evaluar predicciones contra ambos expertos médicos
python scripts/evaluate.py

# 5. Generar máscaras de inferencia reconstruidas al tamaño original
python scripts/predict.py --model models/final/unet_fold_1.keras --images-dir data/raw/test/images --output artifacts/predictions
```

Para una comprobación rápida del entrenamiento, puedes acotar el número de épocas:

```powershell
python scripts/train.py --epochs 1 --batch-size 2
```

## Preparación del dataset

Este proyecto utiliza el **DRIVE 2004 (Digital Retinal Images for Vessel Extraction)**, un conjunto de 40 imágenes de fondo de ojo dividido oficialmente en 20 imágenes de entrenamiento y 20 de prueba. En este repositorio, las 20 imágenes de `training/` aportan ground truth supervisado de vasos; el conjunto `test/` conserva dos segmentaciones expertas y se usa para evaluación promediando ambas referencias.

La referencia clásica del dataset es: Staal J, Abramoff MD, Niemeijer M, Viergever MA, van Ginneken B. *Ridge-based vessel segmentation in color images of the retina.* IEEE Transactions on Medical Imaging. 2004;23(4):501-509.

### Requisitos que marca el enunciado

El PDF del proyecto pide una solución completa de segmentación con estos elementos mínimos:

- Arquitectura U-Net con encoder/decoder y conexiones de salto.
- Preprocesado con normalización, padding y, cuando haga falta, parcheado.
- Aumento de datos para datasets pequeños.
- Entrenamiento reproducible con validación y guardado del modelo.
- Validación cruzada con al menos 5 pliegues.
- Métricas de segmentación, con DICE como referencia mínima.
- Objetivo de rendimiento: media de DICE de al menos 0.75 en DRIVE 2004.
- Exportación del modelo en formato `.keras`.
- Figuras, predicciones en `.png` y un informe final con formato académico.

### Formato esperado de salida

Para cumplir con el enunciado, la salida de inferencia debe conservar el tamaño original de la imagen tras el postprocesado y generar una máscara exportable en `.png` por cada muestra procesada.

### Pasos para descargar y preparar los datos

> [!IMPORTANT]
> Los archivos de datos están excluidos del repositorio (`.gitignore`). Cualquier persona que clone el repo debe ejecutar estos comandos para regenerarlos localmente.

```bash
# Descargar el dataset DRIVE 2004 desde Kaggle (requiere kagglehub)
python -m pip install kagglehub
python scripts/download_drive.py
```

El script `download_drive.py` ha sido reestructurado por completo para que ahora esté enfocado exclusivamente en la estructura del dataset DRIVE 2004 de Kaggle (`andrewmvd/drive-digital-retinal-images-for-vessel-extraction`). Ahora itera directamente sobre los archivos `.tif` de `images/`, binariza a 0/255 las máscaras `.gif` originales (`1st_manual`, `2nd_manual`, `mask`) y las guarda como `.png` en los directorios correspondientes (`masks/`, `masks_expert1/`, `masks_expert2/`, `fov_masks/`).

El dataset `discover_drive_samples()` en `src/data/dataset.py` retorna objetos `SegmentationSample` con las rutas emparejadas imagen-máscara, consumidos directamente por el `DataGenerator`.

## Ejecución del modelo

El entrenamiento en [scripts/train.py](scripts/train.py) consume directamente las imágenes organizadas en `data/raw/` a través del `DataGenerator` (`src/data/generator.py`), que extiende `keras.utils.Sequence` para cargar, parchear (128×128) y aumentar datos sobre la marcha sin necesidad de archivos `.npz` intermedios.

```powershell
& 'C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1'
conda activate ia25-medseg
python scripts/train.py --config configs/default.json
```

Si quieres comprobar rápidamente que todo arranca, usa una pasada corta:

```powershell
python scripts/train.py --config configs/default.json --epochs 1 --batch-size 2
```

El resultado del entrenamiento se guarda en `models/final/` y los logs/checkpoints en `models/checkpoints/` y `artifacts/logs/`.

## Estado actual

El proyecto está completamente implementado para DRIVE 2004: descarga automática del dataset, preprocesado (normalización, conversión a escala de grises, parcheado 128×128), aumento de datos y generación de particiones reproducibles.

Recientemente se completó una refactorización estructural significativa:

- **Scripts eliminados:** `scripts/prepare_data.py` y `scripts/pack_npz.py` fueron suprimidos al optar por un pipeline de carga bajo demanda.
- **Nuevo DataGenerator:** `src/data/generator.py` implementa un generador Keras `Sequence` que parchea y aumenta imágenes sobre la marcha, eliminando la necesidad de pre-empaquetado `.npz`.
- **Simplificación de `download_drive.py`:** Reducido de 234 a 78 líneas, enfocado exclusivamente en la estructura DRIVE 2004. Binariza directamente las máscaras `.gif` originales a `.png` 0/255.
- **Refactorización de `evaluate.py`:** Extracción de lógica repetitiva en funciones auxiliares (`_get_expert_score`, `_print_expert_results`) y búsqueda flexible de máscaras mediante patrones de nombre.

El entrenamiento de la U-Net se ha realizado con validación cruzada de 5 pliegues, obteniendo un DICE promedio de 0.8115 sobre el conjunto de test evaluado contra ambos expertos médicos, superando las métricas exigidas. El informe en LaTeX con los resultados está disponible en `docs/proyecto/memoria.tex`.
