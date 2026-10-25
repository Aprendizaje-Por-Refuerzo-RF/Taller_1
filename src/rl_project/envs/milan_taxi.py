"""Ambiente Milan Taxi -- parametrizable para correr distintos experimentos por config.

Estado: `(row, col, pass_idx, dest_idx)`. Acciones: `{SOUTH, NORTH, EAST, WEST, PICKUP,
DROPOFF}` (fijas -- 6 en todo estado). Un episodio termina (`terminated=True`) cuando un
`DROPOFF` es exitoso; el próximo pasajero y destino se sortean recién en el siguiente
`reset()` (tarea episódica, no continua). `step()` también trunca a `self.horizon` pasos
como respaldo para episodios que nunca llegan a `DROPOFF`.

Todo lo que sigue es configurable por instancia (vía `MilanTaxiEnv(**kwargs)` / el bloque
`env:` de un YAML en `configs/`, reenviado tal cual por `make_env`); los valores por
defecto reproducen el ambiente original de `RL00_Problem_Formulation.ipynb`:

- `num_rows`, `num_cols`: tamaño de la grilla (default 5x5).
- `locs`: lista de landmarks (posiciones `(row, col)`) -- define cuántos pickup/destino
  hay y dónde. `pass_idx == len(locs)` siempre significa "pasajero en el taxi".
- `internal_walls`: pares de celdas vecinas entre las que no se puede pasar (solo
  bloquean movimiento EAST/WEST en el layout original, pero eso es una propiedad de la
  lista que se pase, no una restricción del código).
- `step_reward`: recompensa por toda acción de movimiento (`SOUTH/NORTH/EAST/WEST`),
  se mueva o no (choque contra borde/pared).
- `pickup_reward`: recompensa por un `PICKUP` válido. Si no se pasa, usa el mismo valor
  que `step_reward` (comportamiento original: un pickup válido costaba lo mismo que un
  paso).
- `invalid_reward`: recompensa por `PICKUP` o `DROPOFF` inválido (mismo valor para
  ambos casos -- no hay parámetros separados para "pickup erróneo" vs "dropoff
  erróneo").
- `dropoff_reward`: recompensa por `DROPOFF` exitoso.
- `slip_prob`: probabilidad de que una acción de movimiento (`SOUTH/NORTH/EAST/WEST`)
  "resbale" a una dirección perpendicular en vez de la intencionada (mitad de
  probabilidad para cada lado) -- estilo `FrozenLake`. `slip_prob=0.0` (default) es
  determinista. `PICKUP`/`DROPOFF` nunca resbalan.
- `horizon`: límite de pasos para truncar (no participa en el modelo P, R de
  `models/mdp.py`, solo en `step()`).

El renderizado con `pygame` (`rl_project.utils.rendering`) está dibujado a mano para la
grilla 5x5 y los 4 landmarks de Milán del layout original -- `render()` lo usa solo si la
instancia coincide exactamente con esa configuración por defecto; para cualquier otra
grilla/landmarks/paredes, usar `render_text()` (no requiere pygame ni assets).

Nota de reproducibilidad: `reset(seed=...)` usa `self.np_random` (el generador de
Gymnasium, fijado por `super().reset(seed=seed)`) para sortear posición inicial del taxi,
pasajero y destino; `step()` usa el mismo generador para el resbalón de `slip_prob`. Es
decir, dos `MilanTaxiEnv` con el mismo `seed` en `reset()` producen exactamente la misma
secuencia, sin depender de si algo más en el proceso llamó a `np.random.seed(...)`.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

NUM_ROWS, NUM_COLS = 5, 5
HORIZON = 100

SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF = 0, 1, 2, 3, 4, 5
ACTION_NAMES = ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF"]
MOVEMENT_ACTIONS = (SOUTH, NORTH, EAST, WEST)

# Para slip_prob > 0: qué dos acciones son "perpendiculares" a cada movimiento (estilo
# FrozenLake -- resbalar hacia cualquiera de los dos lados con probabilidad slip_prob/2
# cada uno; PICKUP/DROPOFF no tienen perpendiculares porque nunca resbalan).
PERPENDICULAR_ACTIONS = {
    SOUTH: (EAST, WEST),
    NORTH: (EAST, WEST),
    EAST: (SOUTH, NORTH),
    WEST: (SOUTH, NORTH),
}

INTERNAL_WALLS = [
    ((0, 1), (0, 2)), ((1, 1), (1, 2)),
    ((3, 0), (3, 1)), ((4, 0), (4, 1)),
    ((3, 2), (3, 3)), ((4, 2), (4, 3)),
]

LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

STEP_REWARD = -1
INVALID_REWARD = -10
DROPOFF_REWARD = 20
SLIP_PROB = 0.0

ENV_ID = "MilanTaxi-v0"


def check_wall(row, col, new_row, new_col, internal_walls) -> bool:
    """True si moverse de (row, col) a (new_row, new_col) cruza una pared de `internal_walls`."""
    return ((row, col), (new_row, new_col)) in internal_walls or \
        ((new_row, new_col), (row, col)) in internal_walls


class MilanTaxiEnv(gym.Env):
    """MDP configurable: estado = (row, col, pass_idx, dest_idx). Ver docstring del módulo."""

    def __init__(
        self,
        num_rows: int = NUM_ROWS,
        num_cols: int = NUM_COLS,
        locs: list[tuple[int, int]] | None = None,
        internal_walls: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
        step_reward: float = STEP_REWARD,
        pickup_reward: float | None = None,
        invalid_reward: float = INVALID_REWARD,
        dropoff_reward: float = DROPOFF_REWARD,
        slip_prob: float = SLIP_PROB,
        horizon: int = HORIZON,
    ):
        super().__init__()

        if not (0.0 <= slip_prob < 1.0):
            raise ValueError(f"slip_prob debe estar en [0, 1), recibido: {slip_prob}")

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.locs = list(locs) if locs is not None else list(LOCS)
        self.internal_walls = list(internal_walls) if internal_walls is not None else list(INTERNAL_WALLS)
        self.step_reward = step_reward
        # Si no se pasa `pickup_reward`, un PICKUP válido cuesta lo mismo que un paso
        # normal (comportamiento original, antes de que este parámetro existiera).
        self.pickup_reward = pickup_reward if pickup_reward is not None else step_reward
        self.invalid_reward = invalid_reward
        self.dropoff_reward = dropoff_reward
        self.slip_prob = slip_prob
        self.horizon = horizon

        # "pasajero en el taxi" es siempre el índice justo después del último landmark.
        self.pass_in_taxi = len(self.locs)

        self.state = None
        self.time_step = 0
        self.lastaction = None
        self._delivered_passengers = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # fija self.np_random si seed no es None

        self._delivered_passengers = 0
        self.time_step = 0
        self.lastaction = None

        taxi_row = int(self.np_random.integers(0, self.num_rows))
        taxi_col = int(self.np_random.integers(0, self.num_cols))
        pass_idx, dest_idx = self._spawn_new_passenger()

        self.state = (taxi_row, taxi_col, pass_idx, dest_idx)

        return self.state, {}

    def step(self, action):
        row, col, pass_idx, dest_idx = self.state

        outcomes = self._movement_outcomes(row, col, pass_idx, dest_idx, action)
        if len(outcomes) == 1:
            new_row, new_col, new_pass_idx, new_dest_idx, reward, dest_reached = outcomes[0][1:]
        else:
            probs = [prob for prob, *_ in outcomes]
            chosen = self.np_random.choice(len(outcomes), p=probs)
            new_row, new_col, new_pass_idx, new_dest_idx, reward, dest_reached = outcomes[chosen][1:]

        self.lastaction = action

        self.time_step += 1
        truncated = self.time_step >= self.horizon
        terminated = dest_reached  # DROPOFF exitoso termina el episodio (modificación del taller)

        if dest_reached:
            self._delivered_passengers += 1

        self.state = (new_row, new_col, new_pass_idx, new_dest_idx)

        return self.state, reward, terminated, truncated, {
            "delivered_passengers": self._delivered_passengers
        }

    def _spawn_new_passenger(self):
        pickup, dropoff = self.np_random.choice(len(self.locs), size=2, replace=False)
        return int(pickup), int(dropoff)

    def _transitions(self, row, col, pass_idx, dest_idx, action):
        """Resultado DETERMINISTA de ejecutar `action` en `(row,col,pass_idx,dest_idx)`
        -- es decir, la acción intencionada, sin resbalar. `_movement_outcomes` construye
        la distribución completa (con slip) a partir de esto; `models/mdp.py::build_model`
        también llama a `_movement_outcomes`, nunca reimplementa esta lógica."""
        new_row, new_col = row, col
        new_pass_idx = pass_idx
        dest_reached = False
        reward = self.step_reward

        if action == NORTH:
            new_row = max(row - 1, 0)
        elif action == SOUTH:
            new_row = min(row + 1, self.num_rows - 1)
        elif action == EAST:
            new_col = min(col + 1, self.num_cols - 1)
        elif action == WEST:
            new_col = max(col - 1, 0)
        elif action == PICKUP:
            if pass_idx != self.pass_in_taxi and (row, col) == self.locs[pass_idx]:
                new_pass_idx = self.pass_in_taxi
                reward = self.pickup_reward
            else:
                reward = self.invalid_reward
        elif action == DROPOFF:
            if pass_idx == self.pass_in_taxi and (row, col) == self.locs[dest_idx]:
                reward = self.dropoff_reward
                dest_reached = True
            else:
                reward = self.invalid_reward

        if check_wall(row, col, new_row, new_col, self.internal_walls):
            new_row, new_col = row, col

        return new_row, new_col, new_pass_idx, dest_idx, reward, dest_reached

    def _movement_outcomes(self, row, col, pass_idx, dest_idx, action):
        """Distribución completa de resultados de `action` en `(row,col,pass_idx,dest_idx)`:
        lista de `(prob, new_row, new_col, new_pass_idx, dest_idx, reward, dest_reached)`.

        Con `slip_prob == 0` o `action` no de movimiento (`PICKUP`/`DROPOFF` nunca
        resbalan), es un único resultado con `prob=1.0` -- el mismo comportamiento
        (determinista) de siempre. Con `slip_prob > 0`, la acción intencionada ocurre con
        probabilidad `1 - slip_prob` y cada una de las dos direcciones perpendiculares con
        `slip_prob / 2` (estilo FrozenLake).
        """
        if self.slip_prob == 0.0 or action not in MOVEMENT_ACTIONS:
            return [(1.0, *self._transitions(row, col, pass_idx, dest_idx, action))]

        left, right = PERPENDICULAR_ACTIONS[action]
        return [
            (1.0 - self.slip_prob, *self._transitions(row, col, pass_idx, dest_idx, action)),
            (self.slip_prob / 2, *self._transitions(row, col, pass_idx, dest_idx, left)),
            (self.slip_prob / 2, *self._transitions(row, col, pass_idx, dest_idx, right)),
        ]

    def render(self):
        """Renderiza el frame actual (RGB array) con pygame, vía `rl_project.utils.rendering`.

        Solo funciona para la configuración original de Milan Taxi (grilla 5x5, paredes y
        landmarks por defecto): los assets de pygame están dibujados a mano para esas 4
        ubicaciones exactas. Para cualquier otra configuración, usar `render_text()`.
        """
        is_default_layout = (
            self.num_rows == NUM_ROWS and self.num_cols == NUM_COLS
            and self.locs == LOCS and self.internal_walls == INTERNAL_WALLS
        )
        if not is_default_layout:
            raise NotImplementedError(
                "El renderizado visual (pygame) esta dibujado a mano para la grilla 5x5 "
                "y los landmarks originales de Milan Taxi. Esta instancia usa una "
                "configuracion distinta (grilla/landmarks/paredes) -- usa "
                "env.render_text() en su lugar."
            )

        from rl_project.utils.rendering import render_taxi

        return render_taxi(self.state, self.lastaction)

    def render_text(self):
        """Sustituto en texto, sin pygame ni assets -- funciona para cualquier configuración."""
        if self.state is None:
            return None
        row, col, pass_idx, dest_idx = self.state
        pass_str = "en el taxi" if pass_idx == self.pass_in_taxi else f"en {self.locs[pass_idx]}"
        action_str = ACTION_NAMES[self.lastaction] if self.lastaction is not None else "-"
        return (
            f"t={self.time_step:3d} | taxi=({row},{col}) | pasajero {pass_str} "
            f"| destino {self.locs[dest_idx]} | última acción={action_str}"
        )

    def close(self):
        from rl_project.utils.rendering import close_taxi_render

        close_taxi_render()


def make_env(**kwargs) -> gym.Env:
    """Factory estándar para instanciar el ambiente (útil para train.py / experiment.py).

    `kwargs` se reenvía tal cual a `MilanTaxiEnv(...)` -- viene directo del bloque `env:`
    de un YAML en `configs/` (ver `train.py::train`). Vacío (`env: {}`) equivale a los
    defaults del módulo, es decir, el Milan Taxi original.
    """
    return MilanTaxiEnv(**kwargs)


try:
    register(id=ENV_ID, entry_point="rl_project.envs.milan_taxi:MilanTaxiEnv")
except gym.error.Error:
    # Ya estaba registrado (p.ej. al recargar el módulo en un notebook con autoreload).
    pass
