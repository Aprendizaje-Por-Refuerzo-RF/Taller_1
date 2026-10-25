"""Funciones para crear directorios de ejecucion y guardar resultados."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rl_project.utils.paths import RUNS_DIR


def create_run_dir(algorithm: str, base_dir: Path | str = RUNS_DIR) -> Path:
    """Crea el directorio donde se almacenaran los resultados de una ejecucion.

    La estructura creada es:

        base_dir/
        └── algorithm/
            └── timestamp/

    El timestamp permite identificar cada ejecucion de forma independiente.

    Args:
        algorithm: Nombre del algoritmo utilizado.
        base_dir: Directorio base donde se almacenaran los resultados.

    Returns:
        Ruta del directorio creado para la ejecucion.
    """
    # Generar un identificador basado en la fecha y hora actual.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construir la ruta del directorio de resultados.
    run_dir = Path(base_dir) / algorithm / timestamp

    # Crear el directorio y sus carpetas padre si no existen.
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir
