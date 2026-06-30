"""Project-level paths and defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIGS_DIR = PROJECT_ROOT / "configs"

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODELS_DIR = MODELS_DIR / "final"
LOGS_DIR = ARTIFACTS_DIR / "logs"


def load_json_config(config_path: Path) -> dict[str, Any]:
	"""Load a JSON config file into a dictionary."""
	if not config_path.exists():
		return {}
	with open(config_path, "r", encoding="utf-8") as handle:
		return json.load(handle) or {}