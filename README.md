# ia25-26-medical-image-unet-segmentation

Repositorio base para el trabajo de segmentación de imágenes médicas con redes convolucionales y arquitectura U-Net.

## Alcance del proyecto

La documentación del repositorio fija un objetivo claro: construir, entrenar y evaluar un modelo de segmentación médica capaz de generar máscaras con el mismo formato y tamaño que la imagen original, utilizando Keras 3 y TensorFlow.

Los documentos revisados implican, como mínimo, estas piezas de trabajo:

- Preprocesado de imágenes: normalización, padding y troceado en parches cuando sea necesario.
- Aumento de datos para conjuntos pequeños.
- Implementación de una U-Net con camino encoder/decoder y conexiones de salto.
- Entrenamiento reproducible, con validación y guardado del modelo en formato `.keras`.
- Evaluación con métricas de segmentación, especialmente DICE score.
- En DRIVE, comparación del test contra las dos máscaras expertas y promedio final de ambas métricas.
- Generación de resultados visuales y un informe final con formato de artículo.

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
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── docs/
│   ├── README.md
│   ├── proyecto/
│   │   └── segmentación_imágenes_médicas.pdf
│   └── teoria/
│       ├── Aprendizaje_automático_(Contenido_teórico).pdf
│       ├── Aprendizaje_por_refuerzo_(Contenido_teórico).pdf
│       ├── Planificación_automática_(Contenido_teórico).pdf
│       ├── Procesamiento_del_lenguaje_natural_(Contenido_teórico).pdf
│       └── Redes_neuronales_(Contenido_teórico).pdf
├── notebooks/
│   ├── README.md
│   ├── Gymnasium.ipynb
│   ├── Keras.ipynb
│   ├── NLTK.ipynb
│   ├── Numpy_Pandas.ipynb
│   ├── PDDL.ipynb
│   └── scikit-learn.ipynb
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
│   └── predictions/
└── tests/
```

## Flujo de trabajo previsto

1. Colocar el dataset original en `data/raw/`.
2. Ejecutar el preprocesado y generar particiones en `data/splits/`.
3. Entrenar la U-Net desde `scripts/train.py` o desde un notebook.
4. Guardar pesos, checkpoints y modelo final en `models/`.
5. Evaluar el modelo y exportar figuras y predicciones en `artifacts/`.

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
2. Mismo tamaño de entrada de imagen y máscara definido en `configs/default.yaml`.
3. Mismo criterio de codificación de máscaras binaria o multiclase.
4. Misma definición de métrica principal, con DICE como referencia mínima.
5. Misma convención para nombres de checkpoints, modelo final y figuras.

### Ritmo de integración

- Primera integración: Juan entrega un lote pequeño de datos ya preparados y Diego valida que el entrenamiento arranca.
- Segunda integración: Diego entrega una U-Net funcional con entrada/salida acordadas y Juan valida que sus datos encajan sin cambios manuales.
- Tercera integración: ambos prueban evaluación, guardado del modelo y generación de máscaras sobre ejemplos reales.

Para mantener la dependencia mínima, cualquier cambio de formato debe pasar primero por `configs/default.yaml` y quedar reflejado en este README o en `docs/README.md`.

## Instalación rápida

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Preparación del dataset

Este proyecto utiliza el **DRIVE 2004 (Digital Retinal Images for Vessel Extraction)**, un conjunto de 40 imágenes de fondo de ojo dividido oficialmente en 20 imágenes de entrenamiento y 20 de prueba. En este repositorio, las 20 imágenes de `training/` aportan ground truth supervisado de vasos; el conjunto `test/` conserva dos segmentaciones expertas y se usa para evaluación promediando ambas referencias.

La referencia clásica del dataset es: Staal J, Abramoff MD, Niemeijer M, Viergever MA, van Ginneken B. *Ridge-based vessel segmentation in color images of the retina.* IEEE Transactions on Medical Imaging. 2004;23(4):501-509.

### Pasos para descargar y preparar los datos

> [!IMPORTANT]
> Los archivos de datos están excluidos del repositorio (`.gitignore`). Cualquier persona que clone el repo debe ejecutar estos comandos para regenerarlos localmente.

```bash
# 1. Descargar el dataset DRIVE 2004 desde Kaggle (requiere kagglehub)
python -m pip install kagglehub
python scripts/download_drive.py

# 2. Preprocesar: normalización, parcheado 256×256 y particiones reproducibles
python scripts/prepare_data.py

# 3. (Opcional) Validar el pipeline con una demo visual
python scripts/demo_pipeline.py
```

Tras la ejecución, el pipeline genera parches de 256×256 píxeles (imágenes float32 en escala de grises, máscaras uint8 binarias) a partir del split etiquetado de `training/`. El test oficial de DRIVE mantiene sus dos máscaras expertas para evaluación, aunque no se usa como fuente de entrenamiento.

El aumento de datos (flip horizontal, flip vertical, rotación 90°) se aplica solo sobre las imágenes de entrenamiento, evitando fuga de datos entre particiones.

## Estado actual

El pipeline de datos está completamente implementado para DRIVE 2004: descarga automática del dataset, preprocesado (normalización, máscaras manuales de training, parcheado 256×256), aumento de datos y generación de particiones reproducibles. El siguiente paso es el entrenamiento de la U-Net.
