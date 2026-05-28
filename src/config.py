"""Project-level paths and defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIGS_DIR = PROJECT_ROOT / "configs"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODELS_DIR = MODELS_DIR / "final"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ARTIFACTS_DIR / "logs"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 100
DEFAULT_SEED = 394867


def load_yaml_config(config_path: Path) -> dict[str, Any]:
	"""Load a YAML config file into a dictionary."""

	with config_path.open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle) or {}


def ensure_output_dirs() -> None:
	"""Ensure output directories exist for models and artifacts."""

	CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
	FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
	FIGURES_DIR.mkdir(parents=True, exist_ok=True)
	LOGS_DIR.mkdir(parents=True, exist_ok=True)
	PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)