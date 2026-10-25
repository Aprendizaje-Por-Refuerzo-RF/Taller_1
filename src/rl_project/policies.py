"""Politicas para interactuar con el ambiente.

Este modulo contiene:
- TabularPolicy: permite ejecutar una politica almacenada como matriz.
- RandomPolicy: selecciona acciones de forma aleatoria como referencia.

Las politicas generadas por Policy Iteration y Value Iteration se
almacenan como matrices de tamaño (n_states, n_actions).

Para utilizar estas politicas con un estado del ambiente, se necesita
convertir el estado a su indice correspondiente mediante `state_to_idx`.
"""

from __future__ import annotations

import numpy as np


class TabularPolicy:
    """Politica basada en una matriz de probabilidades por estado y accion.

    La matriz `pi` tiene una fila por estado y una columna por accion:

        pi[estado, accion] = probabilidad de seleccionar la accion

    La politica puede ser determinista o estocastica.

    Args:
        pi: Matriz de la politica con forma (n_states, n_actions).
        state_to_idx: Funcion que convierte un estado del ambiente
            en el indice correspondiente dentro de `pi`.
    """

    def __init__(self, pi: np.ndarray, state_to_idx):

        # Guarda la matriz de la politica.
        self.pi = pi

        # Guarda la funcion que convierte un estado en su indice.
        self.state_to_idx = state_to_idx

    def get_action(self, state) -> int:
        """Selecciona una accion de acuerdo con la politica.

        Para una politica estocastica, la accion se selecciona
        respetando las probabilidades definidas en `pi`.

        Para una politica determinista, se selecciona la accion
        con mayor probabilidad.
        """

        # Se convierte el estado del ambiente en el indice de la politica.  
        s = self.state_to_idx(state)

        # Se obtien las probabilidades de las acciones para ese estado.
        probs = self.pi[s]

        # Si la politica es estocastica, seleccionar una accion
        # respetando las probabilidades definidas.
        if np.isclose(probs.sum(), 1.0) and (probs.max() < 1.0):
            return int(np.random.choice(len(probs), p=probs))

        # Para una politica determinista, seleccionar la accion
        # con mayor probabilidad.      
        return int(np.argmax(probs))


class RandomPolicy:
    """Politica aleatoria uniforme para usar como referencia.

    Selecciona cualquiera de las acciones disponibles con la misma
    probabilidad.
    """

    def __init__(self, n_actions: int = 6):

        # Guarda el numero de acciones disponibles.
        self.n_actions = n_actions


    def get_action(self, state) -> int:
        """Selecciona una accion aleatoria."""

        # Elegir una accion entre todas las acciones disponibles.
        return int(np.random.randint(self.n_actions))
