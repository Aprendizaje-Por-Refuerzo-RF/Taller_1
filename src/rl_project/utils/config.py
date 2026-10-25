"""Funciones para cargar archivos de configuracion YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rl_project.utils.paths import CONFIGS_DIR


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Carga una configuracion desde un archivo YAML.

    El archivo puede especificarse de dos formas:

    - Como nombre de archivo dentro de `configs/`.
      Ejemplo: `"policy_iteration_milan.yaml"`.
    - Como ruta a un archivo YAML.
      Ejemplo: `"/ruta/config.yaml"`.

    Si no se indica la extension `.yaml`, se agrega automaticamente.

    Retorna:
        Diccionario con los parametros definidos en el archivo YAML.

    Raises:
        FileNotFoundError: si no se encuentra el archivo de configuracion.
    """

    # Se convierte el nombre o ruta recibida en un objeto Path.
    path = Path(name_or_path)

    # Se agrega la extension `.yaml` cuando no se especifica.
    if not path.suffix:
        path = path.with_suffix(".yaml")

    # Si no es una ruta absoluta y no existe en la ubicacion actual,
    # buscar el archivo dentro del directorio de configuraciones.
    if not path.is_absolute() and not path.exists():
        path = CONFIGS_DIR / path.name

    # Se verifica que el archivo exista antes de intentar abrirlo.
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuracion: {path}"
        )

    # Se lee el archivo y se convierte su contenido YAML en un diccionario.
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Si el archivo esta vacio se devuelve un diccionario vacio.
    return config or {}
