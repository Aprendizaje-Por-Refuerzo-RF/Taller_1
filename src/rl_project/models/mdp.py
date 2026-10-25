"""El MDP como matrices: P, R y los operadores de Bellman.

La formulación completa (estados, acciones, recompensas, dinámica de transición y su
justificación) está en la sección "Ambiente" de `notebooks/milan_taxi.ipynb` -- este
archivo es la implementación de esa formulación, no una segunda fuente de verdad.
Referencia: Sutton & Barto, Cap. 3.6 (ecs. 3.6-3.9).

`MilanTaxiEnv` es configurable por instancia (grilla, landmarks, recompensas, slip_prob
-- ver `envs/milan_taxi.py`), así que el espacio de estados ya no es un tamaño fijo: cada
llamada a `build_model(env)` construye su propio `StateSpace` a partir de `env.num_rows`,
`env.num_cols` y `len(env.locs)`. `state_to_idx`/`idx_to_state` viven en ese objeto, no
como funciones sueltas de módulo, para que nunca queden desincronizadas de la instancia
de `env` con la que se construyeron `P, R`.

Diseño deliberadamente simple: `build_model` recorre los estados con loops explícitos
(nada de vectorización/einsum) y reusa `env._movement_outcomes(...)` -- la MISMA
distribución de resultados que usa `env.step()` para muestrear -- en vez de reimplementar
la dinámica del ambiente. Con hasta unos pocos miles de combinaciones (estados x
acciones), el costo de no vectorizar es irrelevante, y el código queda mucho más fácil de
leer y de verificar contra la formulación del notebook.

Estado absorbente: un `DROPOFF` exitoso termina el episodio (ver `envs/milan_taxi.py`).
Igual que hace `RL01_Dynamic_Programming.ipynb` con CliffWalking, `build_model` agrega un
estado extra (`StateSpace.absorbing_state`) al que van todas esas transiciones
terminales; ese estado se repite a sí mismo con recompensa 0 para siempre. Por eso `P, R`
tienen tamaño `StateSpace.n_states_model = StateSpace.n_states + 1`, no `n_states`.
"""

from __future__ import annotations

import itertools

import numpy as np


class StateSpace:
    """Espacio de estados de una instancia concreta de `MilanTaxiEnv`.

    Un estado "real" es la tupla `(row, col, pass_idx, dest_idx)`; además se agrega un
    estado absorbente (índice `n_states`, no tiene tupla equivalente) para las
    transiciones donde el episodio termina. `state_to_idx`/`idx_to_state` traducen entre
    esa tupla y la fila/columna que usan `P, R` -- deben ser las MISMAS funciones (esta
    instancia) usadas en cualquier lugar que interprete esos índices, p.ej.
    `policies.TabularPolicy`.
    """

    def __init__(self, num_rows: int, num_cols: int, n_locs: int):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.n_pass_states = n_locs + 1    # 4 destinos + "pasajero en el taxi"
        self.n_dest_states = n_locs        # un destino posible por landmark
        
        # Posibles estados "reales" (los que reporta el ambiente) -- no incluye el estado absorbente.
        self.states = list(itertools.product(
            range(num_rows), 
            range(num_cols), 
            range(self.n_pass_states), 
            range(self.n_dest_states)
        ))
        # Diccionario para traducir de tupla de estado a índice de fila/columna en P, R.
        self._state_to_idx = {}
        for idx, state in enumerate(self.states):
            self._state_to_idx[state] = idx
        

        self.n_states = len(self.states)          # numero de estados "reales" 
        self.absorbing_state = self.n_states      # estado extra: episodio ya terminado
        self.n_states_model = self.n_states + 1   # tamaño real de P, R

    def state_to_idx(self, state: tuple[int, int, int, int]) -> int:
        """Convierte un estado real (row, col, pass_idx, dest_idx) en su índice de fila.
        """
        return self._state_to_idx[state]

    def idx_to_state(self, idx: int) -> tuple[int, int, int, int]:
        """Convierte un índice de fila/columna en su estado real equivalente."""
        return self.states[idx]


def build_model(env) -> tuple[np.ndarray, np.ndarray, StateSpace]:
    """Construye (P, R, state_space) enumerando el espacio de estados de `env`.

    Devuelve:
        P:           array (n_states_model, n_actions, n_states_model) con
                     P[s, a, s'] = p(s' | s, a).
        R:           array (n_states_model, n_actions) con
                     R[s, a] = E[recompensa | s, a] (esperanza sobre `env._movement_outcomes`;
                     coincide con un valor único cuando `env.slip_prob == 0`).
        state_space: el `StateSpace` de esta instancia -- necesario para traducir estados
                     reales en índices de fila/columna de P, R.
    """
    n_actions = 6  # SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF
    state_space = StateSpace(env.num_rows, env.num_cols, len(env.locs))

    P = np.zeros((state_space.n_states_model, n_actions, state_space.n_states_model))
    R = np.zeros((state_space.n_states_model, n_actions))

    # se recorre el espacio de estados "reales" 
    for state in state_space.states:
        s = state_space.state_to_idx(state)
        row, col, pass_idx, dest_idx = state
        # Se recorre el espacio de acciones (6 acciones posibles)
        for a in range(n_actions):
            expected_reward = 0.0
            # Se recorre la distribución de resultados de la acción a en el estado s
            for prob, new_row, new_col, new_pass_idx, _, reward, dest_reached in env._movement_outcomes(
                row, col, pass_idx, dest_idx, a
            ):
                expected_reward += prob * reward

                if dest_reached:
                    # DROPOFF exitoso: el episodio termina 
                    P[s, a, state_space.absorbing_state] += prob
                    R[s, a] = expected_reward
                    
                else:
                    s_next = state_space.state_to_idx((new_row, new_col, new_pass_idx, dest_idx))
                    P[s, a, s_next] += prob
                    R[s, a] = expected_reward
                    

            
    # El estado absorbente se repite a sí mismo para siempre, sin más recompensa.
    P[state_space.absorbing_state, :, state_space.absorbing_state] = 1.0
    R[state_space.absorbing_state, :] = 0.0

    return P, R, state_space


def bellman_expectation_backup(P: np.ndarray, R: np.ndarray, V: np.ndarray, pi: np.ndarray, gamma: float) -> np.ndarray:
    """Un backup de la ecuación de Bellman para v_pi, vectorizado sobre todos los estados.

    V_new(s) = sum_a pi(a|s) sum_{s'} P[s,a,s'] * (R[s,a] + gamma * V(s'))
    """
    # Probabilidad de transicion entre estados bajo la politica pi
    # P_pi[s, s'] = sum_a pi(a|s) P[s,a,s']
    P_pi = np.einsum('sa,sat->st', pi, P)  
    # Recompensa esperada de cada estado bajo la politica pi
    # R_pi[s]     = sum_a pi(a|s) R[s,a]
    R_pi = np.einsum('sa,sa->s', pi, R)   

    # Ecuación de Bellman:
    # V_new = R_pi + gamma * P_pi * V
    V_new = R_pi + gamma * (P_pi @ V)

    return V_new


def bellman_optimality_backup(P: np.ndarray, R: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    """Un backup de la ecuación de optimalidad de Bellman vectorizado.

    V_new(s) = max_a sum_{s'} P[s,a,s'] * (R[s,a] + gamma * V(s'))

    """
    # Se calcula el valor esperado de cada acción en cada estado.
    Q = R + gamma * (P @ V)

    # Para cada estado se selecciona la acción con mayor valor.
    V_new = np.max(Q, axis=1)

    return V_new
