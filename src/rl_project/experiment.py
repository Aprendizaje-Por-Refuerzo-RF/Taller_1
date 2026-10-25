"""Funciones para ejecutar varias corridas de entrenamiento.

Cada configuracion se ejecuta de forma independiente mediante `train()`.
Los resultados de cada corrida son almacenados por `train()` en su propio
directorio dentro de `runs/`.

Este modulo permite ejecutar varias configuraciones de forma secuencial,
por ejemplo, para comparar diferentes algoritmos o parametros.
"""

from __future__ import annotations

from rl_project.train import train


def run_experiment(config_paths: list[str]) -> list[dict]:
    """Ejecuta una corrida de entrenamiento por cada configuracion.

    Args:
        config_paths: Lista de nombres o rutas de configuraciones.

    Returns:
        Lista con los resultados devueltos por cada llamada a `train()`.
    """
    # Guarda los resultados de cada corrida.    
    results = []

    # Se ejecuta cada configuracion de forma independiente.
    for config_path in config_paths:
        print(f"=== Corriendo {config_path} ===")
        # Se ejecuta el entrenamiento y se guarda su resultado.
        result = train(config_path)
        results.append(result)

    return results


if __name__ == "__main__":
    # Ejecutar las configuraciones de ejemplo cuando este archivo
    # se ejecuta directamente.
    run_experiment(["policy_iteration_milan", "value_iteration_milan"])
