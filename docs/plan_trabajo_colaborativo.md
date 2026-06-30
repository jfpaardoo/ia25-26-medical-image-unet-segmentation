# Plan de trabajo colaborativo

Este documento divide el proyecto en dos frentes para que Juan y Diego puedan avanzar en paralelo con la menor dependencia posible.

## Objetivo común

Construir una solución de segmentación médica con U-Net, capaz de entrenar, evaluar y exportar máscaras a partir de imágenes médicas.

## Juan: datos y preprocesado

Tareas principales:

- Revisar el dataset original y fijar su estructura en `data/raw/`.
- Implementar la carga de pares imagen-máscara en `src/data/dataset.py`.
- Implementar preprocesado en `src/data/preprocessing.py`.
- Implementar parcheado y reconstrucción en `src/data/patching.py`.
- Definir aumento de datos y particiones en `data/splits/`.
- Validar que el dataset queda listo para entrenamiento sin pasos manuales.

Entregable de Juan:

- Datos de entrenamiento, validación y prueba listos.
- Contrato de entrada documentado.
- Notebook de exploración si es necesario.

## Diego: modelo y entrenamiento

Tareas principales:

- Implementar la U-Net en `src/models/unet.py`.
- Crear bloques reutilizables en `src/models/blocks.py`.
- Implementar métricas en `src/evaluation/metrics.py`.
- Crear entrenamiento y callbacks en `src/training/`.
- Implementar inferencia y exportación en `scripts/predict.py`.
- Generar resultados finales y guardar el modelo en formato `.keras`.

Entregable de Diego:

- Modelo funcional entrenable y evaluable.
- Checkpoints y modelo final.
- Figuras y predicciones para el informe.

## Interfaz mínima compartida

1. Tamaño de imagen y máscara.
2. Formato de máscara: binario o multiclase.
3. Nombre y ubicación de los ficheros de salida.
4. Métrica principal para comparación.
5. Semilla y parámetros comunes en `configs/default.json`.

## Regla de integración

- Si Juan cambia el formato de datos, actualiza primero el contrato en `README.md` y `configs/default.json`.
- Si Diego cambia la forma de entrada del modelo, debe hacerlo solo si el contrato común lo permite.
- Ninguno debe asumir detalles internos del otro sin documentarlos primero.

## Orden recomendado

1. Cerrar el contrato de datos.
2. Validar que el pipeline de Juan produce un lote de ejemplo.
3. Implementar y probar la U-Net de Diego con ese lote.
4. Integrar métricas, entrenamiento y exportación.
5. Redactar el informe final con capturas y resultados.