"""Evalúa las predicciones generadas frente a los dos expertos médicos."""

import argparse
from pathlib import Path
import cv2
import numpy as np

from src.config import PROJECT_ROOT

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el DICE score entre dos máscaras binarias."""
    y_true_bin = (y_true > 127).astype(np.float32)
    y_pred_bin = (y_pred > 127).astype(np.float32)
    
    intersection = np.sum(y_true_bin * y_pred_bin)
    sum_true_pred = np.sum(y_true_bin) + np.sum(y_pred_bin)
    
    if sum_true_pred == 0:
        return 1.0
        
    return (2.0 * intersection) / sum_true_pred

def _evaluate_single_prediction(pred_path: Path, expert1_dir: Path, expert2_dir: Path) -> tuple[float | None, float | None]:
    """Evalúa una sola imagen de predicción contra ambos expertos."""
    # Nombre de la imagen base (ej. 01_test)
    base_name = pred_path.stem.replace("_pred", "")
    img_id = base_name.split("_")[0]
    
    # Buscar la máscara del experto 1
    mask1_path = expert1_dir / f"{img_id}_test.png"
    if not mask1_path.exists():
        mask1_path = expert1_dir / f"{img_id}_manual1.png"
        if not mask1_path.exists(): 
            mask1_path = expert1_dir / f"{img_id}_manual1.gif"
        
    # Buscar la máscara del experto 2
    mask2_path = expert2_dir / f"{img_id}_test.png"
    if not mask2_path.exists():
        mask2_path = expert2_dir / f"{img_id}_manual2.png"
        if not mask2_path.exists():
            mask2_path = expert2_dir / f"{img_id}_manual2.gif"

    pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    score1, score2 = None, None
    
    if mask1_path.exists():
        mask1 = cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE)
        score1 = dice_coefficient(mask1, pred_img)
        
    if mask2_path.exists():
        mask2 = cv2.imread(str(mask2_path), cv2.IMREAD_GRAYSCALE)
        score2 = dice_coefficient(mask2, pred_img)
        
    return score1, score2

def main():
    parser = argparse.ArgumentParser(description="Evaluar predicciones contra expertos.")
    parser.add_argument("--predictions-dir", type=Path, default=PROJECT_ROOT / "artifacts/predictions_full")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "data/raw/test")
    args = parser.parse_args()

    pred_dir = args.predictions_dir
    test_dir = args.test_dir
    
    expert1_dir = test_dir / "masks_expert1"
    expert2_dir = test_dir / "masks_expert2"
    
    if not pred_dir.exists():
        print(f"Error: No se encuentra la carpeta de predicciones {pred_dir}")
        return
        
    pred_files = list(pred_dir.glob("*.png"))
    if not pred_files:
        print(f"No hay predicciones en {pred_dir}")
        return

    print(f"Evaluando {len(pred_files)} predicciones...\n")
    
    scores_expert1 = []
    scores_expert2 = []

    for pred_path in sorted(pred_files):
        score1, score2 = _evaluate_single_prediction(pred_path, expert1_dir, expert2_dir)
        if score1 is not None:
            scores_expert1.append(score1)
        if score2 is not None:
            scores_expert2.append(score2)

    # Resultados Finales
    print("=" * 40)
    print("RESULTADOS DE LA EVALUACIÓN (DICE SCORE)")
    print("=" * 40)
    
    if scores_expert1:
        avg1 = np.mean(scores_expert1)
        print(f"Experto 1 (Media): {avg1:.4f}  (Evaluado en {len(scores_expert1)} imágenes)")
    else:
        print("Experto 1: No se encontraron máscaras de referencia.")
        
    if scores_expert2:
        avg2 = np.mean(scores_expert2)
        print(f"Experto 2 (Media): {avg2:.4f}  (Evaluado en {len(scores_expert2)} imágenes)")
    else:
        print("Experto 2: No se encontraron máscaras de referencia.")
        
    if scores_expert1 and scores_expert2:
        media_total = np.mean(scores_expert1 + scores_expert2)
        print("-" * 40)
        print(f"MEDIA GLOBAL:      {media_total:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()