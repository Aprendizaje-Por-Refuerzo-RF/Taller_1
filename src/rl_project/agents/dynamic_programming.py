"""Algoritmos de Dynamic Programming para resolver MDPs tabulares.

Incluye:

- Evaluación iterativa de una política.
- Mejora de una política.
- Policy Iteration.
- Value Iteration.

Los algoritmos trabajan directamente sobre el modelo conocido del MDP,
representado por las matrices de transición P y recompensa R.

"""

from __future__ import annotations

import numpy as np

from rl_project.agents.base import Agent
from rl_project.models.mdp import bellman_expectation_backup, bellman_optimality_backup


def iterative_policy_evaluation(
    P: np.ndarray, 
    R: np.ndarray, 
    pi: np.ndarray, 
    gamma: float,
    theta: float = 1e-8, 
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, int]:
    """Calcula V_pi mediante evaluacion iterativa de una politica.

    Comienza con V = 0 y actualiza los valores de los estados utilizando
    la ecuacion de expectativa de Bellman hasta alcanzar la convergencia.

    Retorna:
        V: funcion de valor de la politica.
        n_sweeps: numero de iteraciones realizadas.
    """
    # Inicializa el valor de todos los estados en cero.
    V = np.zeros(P.shape[0])

    # Actualiza la funcion de valor hasta alcanzar la convergencia.
    for k in range(max_iterations):
        # Se realiza un backup de Bellman usando la politica actual.
        V_new = bellman_expectation_backup(P, R, V, pi, gamma)
        # Se mide el cambio maximo entre la iteracion actual y la anterior.
        delta = np.max(np.abs(V_new - V))
        # Se actualiza la funcion de valor.
        V = V_new
        # Si el cambio es suficientemente pequeño, se considera convergencia.
        if delta < theta:
            break

    return V, k + 1


def policy_improvement(P: np.ndarray, R: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    """Construye una politica greedy respecto a una funcion de valor.

    Para cada estado, calcula el valor de cada accion y selecciona
    la accion con mayor valor.

    La politica retornada tiene tamaño (n_states, n_actions) y utiliza
    una representacion one-hot: la accion seleccionada tiene probabilidad
    1 y las demas tienen probabilidad 0.
    """
    # Se obtiene el numero de estados y acciones a partir de la matriz de recompensas.
    n_states, n_actions = R.shape

    # Se calcula el valor esperado de cada accion en cada estado. 
    Q = R + gamma * P @ V

    # Se inicializa una politica con probabilidad cero para todas las acciones.
    pi = np.zeros((n_states, n_actions))

    # Se selecciona la accion con mayor valor para cada estado y se asigna probabilidad 1.
    pi[np.arange(n_states), np.argmax(Q, axis=1)] = 1.0

    return pi


class PolicyIterationAgent(Agent):
    """Agente que resuelve el MDP mediante Policy Iteration.

    El algoritmo alterna dos etapas:

    1. Evaluar la politica actual hasta la convergencia.
    2. Mejorar la politica seleccionando las mejores acciones.

    El proceso termina cuando la politica deja de cambiar.
    """

    def __init__(self, gamma: float, 
                 theta: float = 1e-8,
                 max_eval_iterations: int = 10_000, 
                 max_policy_iterations: int = 1_000
                ):
        
        super().__init__(gamma)

        # Parametros utilizados durante la evaluacion de la politica.
        self.theta = theta
        self.max_eval_iterations = max_eval_iterations

        # Numero maximo de iteraciones de Policy Iteration.
        self.max_policy_iterations = max_policy_iterations

    def solve(self, 
              P: np.ndarray, 
              R: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:

        # Se obtiene el numero de estados y acciones a partir de la matriz de recompensas
        n_states, n_actions = R.shape

        # Comenzar con una politica que asigna la misma probabilidad
        # a todas las acciones.        
        pi = np.ones((n_states, n_actions)) / n_actions

        # se itera la evauación y mejora de la politica hasta que la politica deje de cambiar
        for k in range(self.max_policy_iterations):

            # Se evalua la politica actual hasta alcanzar la convergencia.
            V, _ = iterative_policy_evaluation(
                P, R, pi, self.gamma, self.theta, self.max_eval_iterations
            )

            # se crea una nueva politica greedy respecto a V.
            pi_new = policy_improvement(P, R, V, self.gamma)

            # Si la politica no cambio, se alcanzo la politica estable.
            if np.array_equal(pi_new, pi):
                break

            # Actualizar la politica mejorada.
            pi = pi_new  

        # Se guardan los resultados obtenidos por el agente.
        self.V = V
        self.pi = pi
        self.n_iterations = k + 1

        return self.V, self.pi


class ValueIterationAgent(Agent):
    """Agente que resuelve el MDP mediante Value Iteration.

    Actualiza directamente la funcion de valor utilizando la ecuacion
    de optimalidad de Bellman. Cuando alcanza la convergencia, obtiene
    la politica greedy respecto a la funcion de valor resultante.
    """

    def __init__(self, 
                 gamma: float, 
                 theta: float = 1e-8, 
                 max_iterations: int = 10_000
                 ):
        
        super().__init__(gamma)

        # Criterio de convergencia
        self.theta = theta

        # Numero maximo de iteraciones
        self.max_iterations = max_iterations

    def solve(self, 
              P: np.ndarray, 
              R: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        
        # Inicializar el valor de todos los estados en cero.
        V = np.zeros(P.shape[0])

        # Se actualiza la funcion de valor hasta alcanzar la convergencia
        for k in range(self.max_iterations):

            # Realizar un backup utilizando la ecuacion de optimalidad
            # de Bellman, seleccionando la mejor accion en cada estado.            
            V_new = bellman_optimality_backup(P, R, V, self.gamma)

            # Se mide el cambio maximo entre iteraciones.
            delta = np.max(np.abs(V_new - V))

            # Se actualiza la funcion de valor.
            V = V_new

            # Se detiene el proceso cuando la funcion de valor haya convergido.
            if delta < self.theta:
                break

        # Se obtiene la politica greedy respecto a la funcion de valor resultante.
        pi = policy_improvement(P, R, V, self.gamma)

        # Se guardan los resultados obtenidos por el agente.
        self.V = V
        self.pi = pi
        self.n_iterations = k + 1
        
        return self.V, self.pi
