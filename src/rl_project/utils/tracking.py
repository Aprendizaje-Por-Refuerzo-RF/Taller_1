"""Registro liviano de métricas por corrida (sin dependencias externas tipo MLflow).

Si más adelante quieren sumar puntos en la matriz de evaluación del taller
("herramientas adicionales... MLflow, Hydra, logging"), este es el punto natural para
integrar MLflow: bastaría con que `RunTracker` escriba también a `mlflow.log_metric`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunTracker:
    """Escribe métricas (una línea JSON por registro) y un resumen final en un run_dir."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.run_dir / "metrics.jsonl"

    def log(self, step: int, **metrics: Any) -> None:
        record = {"step": step, **metrics}
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_summary(self, summary: dict[str, Any]) -> None:
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
