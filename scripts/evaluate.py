"""Evalúa las predicciones generadas frente a los dos expertos médicos."""

import argparse
from pathlib import Path
import numpy as np
import keras

from src.config import PROJECT_ROOT

from src.evaluation.metrics import dice_coefficient
from src.data.preprocessing import load_grayscale_image, binarize_mask

def _get_expert_score(pred_img: np.ndarray, expert_dir: Path, img_id: str) -> float | None:
    """Busca la máscara de un experto, la carga y calcula el DICE score."""
    mask_path = expert_dir / f"{img_id}_test.png"
    
    if mask_path.exists():
        mask = load_grayscale_image(mask_path, normalize=False)
        mask_bin = binarize_mask(mask, threshold=127)
        pred_bin = binarize_mask(pred_img, threshold=127)
        return float(keras.ops.convert_to_numpy(dice_coefficient(mask_bin, pred_bin)))

    return None

def _evaluate_single_prediction(pred_path: Path, expert1_dir: Path, expert2_dir: Path) -> tuple[float | None, float | None]:
    """Evalúa una sola imagen de predicción contra ambos expertos."""
    base_name = pred_path.stem.replace("_pred", "")
    img_id = base_name.split("_")[0]

    pred_img = load_grayscale_image(pred_path, normalize=False)

    score1 = _get_expert_score(pred_img, expert1_dir, img_id)
    score2 = _get_expert_score(pred_img, expert2_dir, img_id)

    return score1, score2

def _print_expert_results(scores: list[float], expert_num: int):
    """Imprime la media de DICE score para un experto."""
    if scores:
        avg = np.mean(scores)
        print(f"Experto {expert_num} (Media): {avg:.4f}  (Evaluado en {len(scores)} imágenes)")
    else:
        print(f"Experto {expert_num}: No se encontraron máscaras de referencia.")

def main():
    parser = argparse.ArgumentParser(description="Evaluar predicciones contra expertos.")
    parser.add_argument("--predictions-dir", type=Path, default=PROJECT_ROOT / "artifacts/predictions")
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

    print("=" * 40)
    print("RESULTADOS DE LA EVALUACIÓN (DICE SCORE)")
    print("=" * 40)
    
    _print_expert_results(scores_expert1, expert_num=1)
    _print_expert_results(scores_expert2, expert_num=2)
        
    if scores_expert1 and scores_expert2:
        media_total = np.mean(scores_expert1 + scores_expert2)
        print("-" * 40)
        print(f"MEDIA GLOBAL:      {media_total:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()