"""Clase base para los agentes de Dynamic Programming.

Los agentes trabajan directamente con el modelo conocido del MDP,
representado mediante las matrices de transicion P y recompensa R.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """Clase base para agentes que resuelven un MDP tabular conocido."""

    def __init__(self, gamma: float):
        # Factor de descuento utilizado para valorar las recompensas futuras.
        self.gamma = gamma

        # Funcion de valor de los estados.
        self.V: np.ndarray | None = None

        # Politica del agente.
        self.pi: np.ndarray | None = None

        # Numero de iteraciones realizadas durante la solución del MDP.
        self.n_iterations: int = 0

    @abstractmethod
    def solve(
        self,
        P: np.ndarray,
        R: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resuelve el MDP utilizando el modelo de transicion y recompensa.

        Cada implementacion debe calcular la funcion de valor y la politica,
        almacenarlas en `self.V` y `self.pi`, actualizar `self.n_iterations`
        y devolver ambas como `(V, pi)`.
        """
        raise NotImplementedError

    def get_action(self, state_idx: int) -> int:
        """Obtiene la accion seleccionada por la politica para un estado."""

        # Verifica que el agente haya resuelto previamente el MDP.
        if self.pi is None:
            raise RuntimeError(
                "Agent.solve(...) debe llamarse antes de get_action(...)."
            )

        # Seleccionar la accion con mayor probabilidad segun la politica.
        return int(np.argmax(self.pi[state_idx]))