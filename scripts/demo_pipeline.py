"""Demostración visual del pipeline de datos (parte de Juan).

Este script genera figuras que demuestran que cada paso del pipeline
funciona correctamente, **sin depender del modelo U-Net** (parte de Diego).

Genera las siguientes figuras en ``artifacts/figures/``:

1. ``demo_01_raw_samples.png``     — Muestras crudas del dataset DRIVE 2004
2. ``demo_02_preprocessing.png``   — Redimensionado + grayscale + normalización
3. ``demo_03_augmentations.png``   — Aumentos de datos aplicados
4. ``demo_04_patching.png``        — Parcheado y reconstrucción
5. ``demo_05_split_stats.png``     — Estadísticas de las particiones

Uso::

    python scripts/demo_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # backend sin GUI
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import discover_samples
from src.data.patching import extract_patches, reconstruct_from_patches
from src.data.preprocessing import (
    apply_mask_format,
    normalize_image,
    resize_pair,
    to_grayscale,
)

FIGURES_DIR = REPO_ROOT / "artifacts" / "figures"
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw(sample):
    img = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
    # OpenCV loads as BGR -> convert to RGB for display
    if img is not None and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, msk


# ── 1. Muestras crudas ──────────────────────────────────────────────
def demo_raw_samples(samples, n: int = 6) -> None:
    print("  -> Generando demo_01_raw_samples.png ...")
    subset = samples[:n]
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
    fig.suptitle("1 - Muestras crudas del dataset DRIVE 2004", fontsize=14, y=0.98)

    for i, sample in enumerate(subset):
        img, msk = _load_raw(sample)
        name = sample.image_path.stem

        axes[0, i].imshow(img)
        axes[0, i].set_title(name, fontsize=8)
        axes[0, i].axis("off")

        if msk is not None and msk.ndim == 3:
            msk = msk.max(axis=-1)
        axes[1, i].imshow(msk, cmap="gray")
        axes[1, i].set_title("Mascara", fontsize=8)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Imagen", fontsize=10)
    axes[1, 0].set_ylabel("Mascara", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "demo_01_raw_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 2. Preprocesado ─────────────────────────────────────────────────
def demo_preprocessing(samples) -> None:
    print("  -> Generando demo_02_preprocessing.png ...")
    sample = samples[0]
    img_raw, msk_raw = _load_raw(sample)

    # resize + grayscale
    img_resized, msk_resized = resize_pair(
        img_raw, msk_raw, (256, 256), mask_format="binary", target_channels=1
    )
    # normalize
    img_norm = normalize_image(img_resized)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("2 - Preprocesado (resize -> grayscale -> normalize)", fontsize=14, y=0.98)

    # Fila 1: imagenes
    axes[0, 0].imshow(img_raw)
    axes[0, 0].set_title(f"Original\n{img_raw.shape}", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_resized, cmap="gray")
    axes[0, 1].set_title(f"Resized + Grayscale\n{img_resized.shape}", fontsize=9)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_norm, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Normalizado [0,1]\n{img_norm.shape} {img_norm.dtype}", fontsize=9)
    axes[0, 2].axis("off")

    # Fila 2: mascaras
    if msk_raw.ndim == 3:
        msk_display = msk_raw.max(axis=-1)
    else:
        msk_display = msk_raw
    axes[1, 0].imshow(msk_display, cmap="gray")
    axes[1, 0].set_title(f"Mascara original\n{msk_raw.shape}", fontsize=9)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(msk_resized, cmap="gray")
    axes[1, 1].set_title(f"Mascara resized\n{msk_resized.shape} uniq={np.unique(msk_resized)}", fontsize=9)
    axes[1, 1].axis("off")

    # Overlay
    axes[1, 2].imshow(img_norm, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].imshow(msk_resized, cmap="Reds", alpha=0.4)
    axes[1, 2].set_title("Overlay imagen + mascara", fontsize=9)
    axes[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "demo_02_preprocessing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 3. Aumentos de datos ────────────────────────────────────────────
def demo_augmentations(samples) -> None:
    print("  -> Generando demo_03_augmentations.png ...")
    sample = samples[2]  # pick one with visible lesion
    img_raw, msk_raw = _load_raw(sample)
    img_r, msk_r = resize_pair(img_raw, msk_raw, (256, 256), mask_format="binary", target_channels=1)
    img_n = normalize_image(img_r)

    augmentations = {
        "Original": (img_n, msk_r),
        "H-Flip": (np.fliplr(img_n).copy(), np.fliplr(msk_r).copy()),
        "V-Flip": (np.flipud(img_n).copy(), np.flipud(msk_r).copy()),
        "Rot90": (np.rot90(img_n, 1).copy(), np.rot90(msk_r, 1).copy()),
    }

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("3 - Aumentos de datos", fontsize=14, y=0.98)

    for i, (name, (img, msk)) in enumerate(augmentations.items()):
        axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(name, fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(msk, cmap="Reds", alpha=0.4)
        axes[1, i].set_title(f"{name} + mascara", fontsize=10)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Imagen", fontsize=10)
    axes[1, 0].set_ylabel("Overlay", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "demo_03_augmentations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 4. Parcheado y reconstrucción ────────────────────────────────────
def demo_patching(samples) -> None:
    print("  -> Generando demo_04_patching.png ...")
    sample = samples[1]
    img_raw, msk_raw = _load_raw(sample)

    # Use a size that is NOT a multiple of patch to show padding
    img_r, msk_r = resize_pair(img_raw, msk_raw, (300, 400), mask_format="binary", target_channels=1)
    img_n = normalize_image(img_r)

    patch_size = (128, 128)
    stride = (100, 100)  # overlapping

    img_patches, msk_patches, positions, padded_shape = extract_patches(
        img_n, msk_r, patch_size=patch_size, stride=stride
    )

    rebuilt = reconstruct_from_patches(img_patches, positions, output_shape=img_n.shape[:2])

    n_patches = img_patches.shape[0]
    cols = min(n_patches, 6)

    fig, axes = plt.subplots(3, max(cols, 3), figsize=(3.5 * max(cols, 3), 10))
    fig.suptitle(
        f"4 - Parcheado ({img_n.shape[:2]} -> {n_patches} parches de {patch_size}, stride={stride})",
        fontsize=13, y=0.98,
    )

    # Fila 1: parches individuales
    for i in range(cols):
        p = img_patches[i]
        if p.ndim == 3 and p.shape[-1] == 1:
            p = p[:, :, 0]
        axes[0, i].imshow(p, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Parche {i}\npos={positions[i]}", fontsize=8)
        axes[0, i].axis("off")
    for i in range(cols, axes.shape[1]):
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("Parches", fontsize=10)

    # Fila 2: original vs reconstruida
    axes[1, 0].imshow(img_n, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Original", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rebuilt, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Reconstruida", fontsize=10)
    axes[1, 1].axis("off")

    diff = np.abs(img_n.squeeze() - rebuilt.squeeze())
    axes[1, 2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max() + 1e-8)
    axes[1, 2].set_title(f"Diferencia (max={diff.max():.6f})", fontsize=10)
    axes[1, 2].axis("off")

    for i in range(3, axes.shape[1]):
        axes[1, i].axis("off")
    axes[1, 0].set_ylabel("Reconstrucción", fontsize=10)

    # Fila 3: mascara parches
    for i in range(min(cols, n_patches)):
        p = msk_patches[i]
        if p.ndim == 3:
            p = p[:, :, 0]
        axes[2, i].imshow(p, cmap="gray")
        axes[2, i].set_title(f"Mascara parche {i}", fontsize=8)
        axes[2, i].axis("off")
    for i in range(cols, axes.shape[1]):
        axes[2, i].axis("off")
    axes[2, 0].set_ylabel("Mascara parches", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "demo_04_patching.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 5. Estadísticas de splits ────────────────────────────────────────
def demo_split_stats() -> None:
    print("  -> Generando demo_05_split_stats.png ...")
    summary_path = SPLITS_DIR / "summary.json"
    if not summary_path.exists():
        print("    [SKIP] No se encontró summary.json — ejecuta prepare_data.py primero")
        return

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    splits = {
        "Train": summary["num_train"],
        "Validation": summary["num_val"],
        "Test": summary["num_test"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("5 - Estadísticas del dataset preparado", fontsize=14, y=1.02)

    # Grafico de barras
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = axes[0].bar(splits.keys(), splits.values(), color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Parches por partición", fontsize=11)
    axes[0].set_ylabel("Nº de parches")
    for bar, val in zip(bars, splits.values()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                     str(val), ha="center", fontsize=11, fontweight="bold")

    # Grafico circular
    axes[1].pie(
        splits.values(),
        labels=splits.keys(),
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Distribución de particiones", fontsize=11)

    # Tabla resumen
    axes[2].axis("off")
    table_data = [
        ["Muestras crudas", str(summary["num_raw_samples"])],
        ["Saltadas", str(summary["num_skipped_samples"])],
        ["Parches totales", str(summary["num_processed_patches"])],
        ["Train", str(summary["num_train"])],
        ["Validation", str(summary["num_val"])],
        ["Test", str(summary["num_test"])],
        ["Tamaño imagen", str(summary["image_size"])],
        ["Tamaño parche", str(summary["patch_size"])],
        ["Stride", str(summary["patch_stride"])],
        ["Augmentations", ", ".join(summary["augmentation"]["transforms"])],
        ["Seed", str(summary["seed"])],
    ]
    table = axes[2].table(
        cellText=table_data,
        colLabels=["Parametro", "Valor"],
        cellLoc="left",
        loc="center",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    axes[2].set_title("Resumen del pipeline", fontsize=11)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "demo_05_split_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 6. Muestras procesadas (.npy) ────────────────────────────────────
def demo_processed_samples() -> None:
    print("  -> Generando demo_06_processed_samples.png ...")
    processed_dir = REPO_ROOT / "data" / "processed"
    img_dir = processed_dir / "images"
    msk_dir = processed_dir / "masks"

    if not img_dir.exists():
        print("    [SKIP] No hay datos procesados — ejecuta prepare_data.py primero")
        return

    # Pick 8 random processed samples
    all_images = sorted(img_dir.glob("*.npy"))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(all_images), size=min(8, len(all_images)), replace=False)
    chosen = [all_images[i] for i in sorted(indices)]

    n = len(chosen)
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 5.6))
    fig.suptitle("6 - Muestras procesadas (parches .npy listos para entrenar)", fontsize=13, y=0.98)

    for i, img_path in enumerate(chosen):
        img = np.load(img_path)
        mask_path = msk_dir / img_path.name
        msk = np.load(mask_path) if mask_path.exists() else np.zeros(img.shape[:2])

        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[:, :, 0]

        axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(img_path.stem.split("__")[1], fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(msk, cmap="Reds", alpha=0.45)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Imagen", fontsize=9)
    axes[1, 0].set_ylabel("+ Máscara", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "demo_06_processed_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    _ensure_dirs()

    print("=" * 60)
    print("  Demo del pipeline de datos — parte de Juan")
    print("=" * 60)

    samples = discover_samples(RAW_DIR)
    if not samples:
        raise SystemExit(
            "No se encontraron pares imagen-máscara en data/raw/.\n"
            "Ejecuta primero: python scripts/download_drive.py"
        )
    print(f"\nDescubiertos {len(samples)} pares imagen-máscara en data/raw/\n")

    demo_raw_samples(samples)
    demo_preprocessing(samples)
    demo_augmentations(samples)
    demo_patching(samples)
    demo_split_stats()
    demo_processed_samples()

    print(f"\n{'=' * 60}")
    print(f"  OK Todas las figuras guardadas en {FIGURES_DIR}/")
    print(f"{'=' * 60}")
    print("\nFiguras generadas:")
    for fig_path in sorted(FIGURES_DIR.glob("demo_*.png")):
        print(f"  • {fig_path.name}")


if __name__ == "__main__":
    main()
