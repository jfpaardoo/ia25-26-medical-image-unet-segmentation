# ia25-26-medical-image-unet-segmentation

Repositorio base para el trabajo de segmentacion de imagenes medicas con redes convolucionales y arquitectura U-Net.

## Alcance del proyecto

La documentacion del repositorio fija un objetivo claro: construir, entrenar y evaluar un modelo de segmentacion medica capaz de generar mascaras con el mismo formato y tamano que la imagen original, utilizando Keras 3 y TensorFlow.

Los documentos revisados implican, como mínimo, estas piezas de trabajo:

- Preprocesado de imagenes: normalizacion, padding y troceado en parches cuando sea necesario.
- Aumento de datos para conjuntos pequenos.
- Implementacion de una U-Net con camino encoder/decoder y conexiones de salto.
- Entrenamiento reproducible, con validacion y guardado del modelo en formato `.keras`.
- Evaluacion con metricas de segmentacion, especialmente DICE score.
- Generacion de resultados visuales y un informe final con formato de articulo.

## Referencias analizadas

- [docs/proyecto/segmentación_imágenes_médicas.pdf](docs/proyecto/segmentación_imágenes_médicas.pdf)
- [docs/teoria/Redes_neuronales_(Contenido_teórico).pdf](docs/teoria/Redes_neuronales_(Contenido_teórico).pdf)
- [notebooks/Keras.ipynb](notebooks/Keras.ipynb)

### Resumen tecnico de la documentacion

- El PDF de segmentacion define la entrega: modelo U-Net, data augmentation, padding, parcheado, metricas DICE, exportacion en `.keras`, informe y presentacion.
- El PDF teorico de redes neuronales aporta el marco matematico: perceptron, descenso del gradiente, backpropagation, funciones de activacion y criterios de salida segun tarea.
- El notebook `Keras.ipynb` muestra el estilo de trabajo esperado: notebooks educativos, reproducibilidad, `Sequential`, normalizacion, preprocesado con `scikit-learn` y ejemplos de clasificacion/regresion con Keras.

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

## Instalacion rapida

```bash
python -m pip install -r requirements.txt
```

## Estado actual

Este repositorio contiene ya la base de carpetas y modulos para empezar a implementar el proyecto, pero todavia no incluye el pipeline completo de entrenamiento ni el dataset.
