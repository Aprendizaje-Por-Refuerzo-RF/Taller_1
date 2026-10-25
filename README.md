# Milan Taxi — Dynamic Programming

Resuelve el ambiente **Milan Taxi** (un taxi en una grilla que debe recoger un pasajero
en un punto y dejarlo en otro) con **Policy Iteration** y **Value Iteration**. El
ambiente es configurable (grilla, recompensas, estocasticidad), así que el mismo código
sirve para correr y comparar distintas variantes del problema.

La formulación completa del MDP (estados, acciones, recompensas, dinámica de transición)
está en la sección **Ambiente** de [`notebooks/milan_taxi.ipynb`](notebooks/milan_taxi.ipynb).

## Estado actual

- **Ambiente** (`src/rl_project/envs/milan_taxi.py`): `MilanTaxiEnv`, configurable por
  instancia: grilla (`num_rows`/`num_cols`), landmarks (`locs`), paredes
  (`internal_walls`), recompensas (`step_reward`/`pickup_reward`/`invalid_reward`/`dropoff_reward`)
  y estocasticidad (`slip_prob`; `0.0` = determinista). Ver "Cómo correr un experimento"
  más abajo.
- **Modelo** (`src/rl_project/models/mdp.py`): `build_model(env)` construye `P, R` para
  cualquier configuración del ambiente anterior (grilla y cantidad de landmarks
  variables), con un estado absorbente para las transiciones terminales.
- **Policy Iteration / Value Iteration** (`src/rl_project/agents/dynamic_programming.py`):
  implementados y validados. Sobre la configuración por defecto (Milan Taxi
  determinista): PI converge en 3 iteraciones, VI en 19 sweeps, ambos dan exactamente el
  mismo $V^*$ y la misma política óptima greedy.
- **CLI** (`src/rl_project/cli.py`): `train`, `experiment` y `visualize` (renderiza un
  GIF de la política óptima y evalúa su desempeño por simulación) corren de punta a
  punta.

## Estructura del proyecto

```
configs/                        # Configuración de cada algoritmo (gamma, theta, env, ...)
├── policy_iteration_milan.yaml
├── value_iteration_milan.yaml
└── value_iteration_milan_custom_rewards.yaml
notebooks/
└── milan_taxi.ipynb            # Formulación del MDP + validación interactiva
runs/                            # Resultados generados por cada corrida (no versionado)
src/rl_project/
├── envs/                       # milan_taxi.py: MilanTaxiEnv (configurable), registro y make_env
├── models/                     # mdp.py: StateSpace, P, R y operadores de Bellman
├── agents/                     # base.py (interfaz) + dynamic_programming.py (PI/VI)
├── policies.py                 # π(a|s): TabularPolicy (greedy/estocástica) y RandomPolicy
├── utils/                      # config.py, paths.py, run_dir.py, tracking.py, rendering.py
├── train.py                    # Encapsula la lógica de entrenamiento (resolver el MDP)
├── evaluation.py                # Evaluar una política por simulación (rollouts)
├── experiment.py                # Varias corridas / configuraciones en secuencia
└── cli.py                      # Comandos para ejecutar y visualizar experimentos
```

## Setup

Este proyecto usa [uv](https://docs.astral.sh/uv/). `uv sync` lee `requires-python` y
las `dependencies` de `pyproject.toml`, descarga un intérprete de Python compatible si
hace falta, crea/actualiza `.venv/` e instala `rl_project` en modo editable (gracias a
`[tool.hatch.build.targets.wheel] packages = ["src/rl_project"]`) -- no hay que instalar
Python ni las dependencias a mano.

```powershell
uv sync
.venv\Scripts\Activate.ps1
uv run jupyter lab
```

## Cómo correr un experimento

Cada archivo en `configs/*.yaml` es una corrida: algoritmo + configuración del ambiente +
hiperparámetros del agente. `train.py` reenvía cada clave a donde corresponde (ninguna es
decorativa -- una clave que el agente no reconozca hace fallar la corrida con `TypeError`,
a propósito, para que nunca quede una clave del YAML ignorada en silencio):

```yaml
algorithm: value_iteration          # selecciona la clase de agente

env:                                 # kwargs de MilanTaxiEnv(...) -- todos opcionales
  slip_prob: 0.1                    # probabilidad de que un movimiento resbale
  # num_rows: 7                     # grilla distinta al 5x5 original
  # num_cols: 7
  # locs: [[0, 0], [0, 6], [6, 0], [6, 6]]
  # internal_walls: []
  # step_reward: -1
  # pickup_reward: -1                # default: igual a step_reward si no se pasa
  # invalid_reward: -10
  # dropoff_reward: 20

gamma: 0.99
theta: 1.0e-8
max_iterations: 10000               # (Policy Iteration usa max_eval_iterations / max_policy_iterations en su lugar)
seed: 0                             # semilla por defecto para los rollouts de cli.py visualize

output:
  runs_dir: runs
```

Para un experimento nuevo, copiar uno de los YAML existentes, ajustar `env:`/`gamma`/etc.,
y correrlo:

```powershell
uv run python -m rl_project.cli train --config mi_experimento
uv run python -m rl_project.cli visualize --config mi_experimento --compare-random
```

También se puede correr una variante directamente en Python/notebook, sin YAML, pasando
los kwargs a `make_env(...)` (ver la sección **Ambiente** de `notebooks/milan_taxi.ipynb`
para ejemplos completos con `build_model` + `PolicyIterationAgent`/`ValueIterationAgent`).

Nota sobre el renderizado visual (`env.render()`, pygame / `cli.py visualize`): solo
funciona con la grilla y los landmarks originales de Milán -- cualquier otra `env:` (otra
grilla, otros `locs`) sigue resolviéndose y evaluándose igual, pero para verla hay que usar
`env.render_text()` en vez de la animación/GIF.

## Nota sobre el ambiente

`MilanTaxiEnv` (`src/rl_project/envs/milan_taxi.py`) modela: grilla configurable (5×5 por
defecto), landmarks configurables (4 por defecto), acciones `{SOUTH, NORTH, EAST, WEST,
PICKUP, DROPOFF}`, recompensas configurables (`-1 / -1 / -10 / +20` por defecto) y
estocasticidad opcional (`slip_prob`, `0.0` por defecto = determinista). El renderizado
con `pygame` vive en `src/rl_project/utils/rendering.py`, separado del ambiente.
`env.render()` devuelve un frame RGB (solo para la configuración original, ver arriba);
`env.render_text()` es un sustituto en texto sin dependencias, que funciona para
cualquier configuración. `utils.animate_frames(frames)` anima una lista de frames en un
notebook; `utils.save_animation(frames, path)` la guarda como `.gif` fuera de un notebook
(usado por `cli.py visualize`).

**Dinámica del episodio:** un `DROPOFF` exitoso termina el episodio (`terminated=True`);
el próximo pasajero y destino se sortean recién en el siguiente `reset()`. La tarea es
episódica: cada episodio empieza con un pasajero y termina cuando se lo deja en su
destino. `models/mdp.py::build_model` agrega un estado absorbente para representar esas
transiciones terminales dentro de las ecuaciones de Bellman. Detalle completo, con la
justificación y los chequeos, en la sección **Ambiente** de `notebooks/milan_taxi.ipynb`.

**Estocasticidad:** el parámetro `slip_prob` de `MilanTaxiEnv` (default `0.0` =
determinista) hace que, con esa probabilidad, una acción de movimiento resbale hacia una
de las dos direcciones perpendiculares (estilo `FrozenLake`) en vez de la intencionada;
`PICKUP`/`DROPOFF` nunca resbalan.

**Reproducibilidad:** `env.reset(seed=...)` usa el generador propio de la instancia
(`self.np_random`, el que fija Gymnasium), no el generador global `np.random` -- dos
`MilanTaxiEnv` con el mismo `seed` producen exactamente la misma secuencia (posición
inicial, pasajero/destino, y los resbalones si `slip_prob > 0`), sin depender de si algo
más en el proceso llamó a `np.random.seed(...)` antes.
