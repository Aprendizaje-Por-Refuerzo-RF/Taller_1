"""Rutas del proyecto"""

from __future__ import annotations

from pathlib import Path

# src/rl_project/utils/paths.py -> sube 3 niveles hasta la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIGS_DIR = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RUNS_DIR = PROJECT_ROOT / "runs"
