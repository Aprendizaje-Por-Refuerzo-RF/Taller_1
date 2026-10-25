"""Entrenamiento de un agente de Dynamic Programming a partir de una configuracion.

Este modulo ejecuta una corrida completa de entrenamiento:

1. Carga la configuracion.
2. Crea el ambiente.
3. Construye el modelo tabular del MDP.
4. Selecciona el algoritmo configurado.
5. Resuelve el MDP.
6. Guarda la funcion de valor, la politica y un resumen de la corrida.

Los resultados se almacenan en:

    runs/<algorithm>/<timestamp>/

Los parametros definidos en el YAML se distribuyen entre el ambiente,
el agente y la salida segun corresponda.
"""

from __future__ import annotations

import numpy as np

from rl_project.agents import PolicyIterationAgent, ValueIterationAgent
from rl_project.envs import make_env
from rl_project.models import build_model
from rl_project.utils import RunTracker, create_run_dir, load_config

# Algoritmos disponibles para resolver el MDP.
AGENTS = {
    "policy_iteration": PolicyIterationAgent,
    "value_iteration": ValueIterationAgent,
}

# Claves de configuracion que no se envian al constructor del agente.
_RESERVED_KEYS = {"algorithm", "env", "gamma", "seed", "output"}


def train(config_path: str) -> dict:
    """Ejecuta una corrida completa a partir de una configuracion.

    Args:
        config_path: Nombre o ruta del archivo YAML de configuracion.

    Returns:
        Diccionario con los resultados de la corrida:
        `V`, `pi`, `run_dir`, `agent`, `state_space` y `config`.
    """
    # Se carga la configuracion del experimento.    
    config = load_config(config_path)

    # Se crea el ambiente utilizando los parametros definidos en `env`.
    env = make_env(**config.get("env", {}))

    # Se construyen las matrices de transicion y recompensa del MDP.
    P, R, state_space = build_model(env)

    # Se selecciona el algoritmo configurado.
    algorithm = config["algorithm"]
    agent_cls = AGENTS[algorithm]

    # Se separan los parametros que deben enviarse al agente.
    agent_kwargs = {k: v for k, v in config.items() if k not in _RESERVED_KEYS}

    # Se crea el agente usando los parametros de configuracion y el factor de descuento.
    agent = agent_cls(gamma=config["gamma"], **agent_kwargs)

    # Se resuelve el MDP y se obtiene la funcion de valor y la politica.
    V, pi = agent.solve(P, R)

    # Se crea el directorio donde se almacenaran los resultados.
    run_dir = create_run_dir(algorithm, base_dir=config.get("output", {}).get("runs_dir"))

    # Se crea el gestor encargado de guardar el resumen de la corridad
    tracker = RunTracker(run_dir)

    # Se guardan la funcion de valor y la politica en archivos
    np.save(run_dir / "V.npy", V)
    np.save(run_dir / "pi.npy", pi)

    # Se guardan los parametros y resultados de la corrida en un archivo de resumen
    tracker.save_summary({
        "algorithm": algorithm,
        "gamma": config["gamma"],
        "seed": config.get("seed"),
        "env": config.get("env", {}),
        "n_iterations": agent.n_iterations,
        **agent_kwargs,
    })

    # Se retornan los resultados para que puedan ser utilizados
    # por otros modulos sin tener que volver a resolver el MDP.
    return {
        "V": V,
        "pi": pi,
        "run_dir": run_dir,
        "agent": agent,
        "state_space": state_space,
        "config": config,
    }


if __name__ == "__main__":
    import sys
    
    # Permitir ejecutar una configuracion directamente desde la terminal.
    config_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "policy_iteration_milan"
    )

    train(config_path)
