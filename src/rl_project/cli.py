"""Interfaz de linea de comandos para ejecutar experimentos.

Permite ejecutar tres operaciones:

    train
        Resuelve un MDP utilizando una configuracion.

    experiment
        Ejecuta varias configuraciones de forma secuencial.

    visualize
        Resuelve un MDP, genera un GIF de un rollout y evalua
        la politica mediante simulacion.

Ejemplos:

    uv run python -m rl_project.cli train --config policy_iteration_milan

    uv run python -m rl_project.cli experiment \
        --configs policy_iteration_milan value_iteration_milan

    uv run python -m rl_project.cli visualize \
        --config policy_iteration_milan \
        --compare-random
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl_project.envs import make_env
from rl_project.evaluation import evaluate_policy, rollout
from rl_project.experiment import run_experiment
from rl_project.policies import RandomPolicy, TabularPolicy
from rl_project.train import train
from rl_project.utils import save_animation


def visualize(
    config_path: str,
    out: str | Path | None = None,
    seed: int | None = None,
    n_runs: int = 100,
    compare_random: bool = False,
) -> dict:
    """Resuelve, visualiza y evalua una politica.

    La funcion reutiliza `train()` para resolver el MDP y obtener la
    politica. Luego ejecuta un rollout para generar el GIF y varios
    episodios para medir el retorno promedio.

    Args:
        config_path: Nombre o ruta de la configuracion.
        out: Ruta donde se guardara el GIF.
        seed: Semilla utilizada para el rollout renderizado.
        n_runs: Numero de episodios utilizados para la evaluacion.
        compare_random: Indica si tambien se debe evaluar una politica
            aleatoria como referencia.

    Returns:
        Diccionario con las rutas y metricas generadas durante la corrida.
    """
    # Se resuelve el MDP utilizando la configuracion indicada.
    result = train(config_path)

    # Se obtienen los parametros necesarios para la evaluacion.
    config = result["config"]
    gamma = config["gamma"]

    # Se crea el ambiente para ejecutar la politica.
    env = make_env(**config.get("env", {}))

    # CSe convierte la matriz de politica en un objeto que pueda
    # seleccionar acciones a partir de los estados del ambiente.    
    policy = TabularPolicy(result["pi"], result["state_space"].state_to_idx)


    # Se define la ruta del GIF.
    if out is not None:
        out_path = Path(out)
    else:
        out_path = result["run_dir"] / "rollout.gif"

    # Se utiliza la semilla recibida por CLI o, si no existe,
    # la definida en la configuracion.    
    effective_seed = seed if seed is not None else config.get("seed", 0)

    # Se ejecuta un episodio y se guardan  los frames para generar el GIF.
    frames, _, actions, rewards = rollout(env, policy, render=True, seed=effective_seed)

    print(f"Rollout renderizado (seed={effective_seed}): {len(actions)} pasos, retorno total = {sum(rewards)}")

    # Se crea y se guarda la animacion    
    save_animation(frames, out_path)

    print(f"GIF guardado en: {out_path}")

    # Se evalua la politica durante varios episodios
    print("Política óptima:")

    mean_opt, margin_opt = evaluate_policy(
        env, policy, n_runs=n_runs, discount_factor=gamma, display_result=True
    )

    summary = {
        "run_dir": str(result["run_dir"]),
        "gif_path": str(out_path),
        "seed": effective_seed,
        "n_episode_steps": len(actions),
        "episode_return": float(sum(rewards)),
        "mean_discounted_return": mean_opt,
        "margin_of_error": margin_opt,
    }

    # Se evalua tambien una politica aleatoria si fue solicitado
    if compare_random:
        print("Política aleatoria (baseline):")
        mean_rand, margin_rand = evaluate_policy(
            env, RandomPolicy(), n_runs=n_runs, discount_factor=gamma, display_result=True
        )
        summary["random_mean_discounted_return"] = mean_rand
        summary["random_margin_of_error"] = margin_rand

    return summary


def main() -> None:
    """Configura los comandos de la interfaz de linea y los ejecuta."""
    # Crea el parser principal de la aplicacion.
    parser = argparse.ArgumentParser(prog="rl_project", description=__doc__)

    # Registra los subcomandos disponibles y sus argumentos.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Se configura el comando 'train'
    train_parser = subparsers.add_parser("train", help="Resuelve el MDP con una config.")

    train_parser.add_argument("--config", required=True, help="Nombre o ruta del YAML en configs/.")

    # Se configura el comando 'experiment'
    experiment_parser = subparsers.add_parser("experiment", help="Corre varias configs seguidas.")

    experiment_parser.add_argument("--configs", nargs="+", required=True)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Resuelve el MDP, renderiza un rollout de la política óptima (GIF) y evalúa su desempeño.",
    )
    visualize_parser.add_argument("--config", required=True, help="Nombre o ruta del YAML en configs/.")

    # Se configuran los argumentos del comando 'visualize'
    visualize_parser.add_argument(
        "--out", default=None,
        help="Ruta del .gif de salida (default: runs/<algoritmo>/<timestamp>/rollout.gif).",
    )
    visualize_parser.add_argument(
        "--seed", type=int, default=None,
        help="Semilla para el rollout renderizado (default: seed de la config, o 0 si no tiene).",
    )
    visualize_parser.add_argument(
        "--n-runs", type=int, default=100,
        help="Cantidad de episodios para evaluate_policy (default: 100).",
    )
    visualize_parser.add_argument(
        "--compare-random", action="store_true",
        help="También evalúa una política aleatoria uniforme como referencia.",
    )

    # Se leen los argumentos proporcionados desde la terminal
    args = parser.parse_args()

    # Se ejecuta el comando correspondiente
    if args.command == "train":
        train(args.config)
    elif args.command == "experiment":
        run_experiment(args.configs)
    elif args.command == "visualize":
        visualize(
            args.config,
            out=args.out,
            seed=args.seed,
            n_runs=args.n_runs,
            compare_random=args.compare_random,
        )


if __name__ == "__main__":
    main()
