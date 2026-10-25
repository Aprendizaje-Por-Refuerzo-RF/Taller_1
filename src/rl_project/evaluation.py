"""Funciones para evaluar politicas mediante simulacion.

Este modulo permite ejecutar una politica directamente sobre el ambiente
y medir su rendimiento a partir de varios episodios.

`rollout` ejecuta un episodio completo y registra los estados, acciones
y recompensas obtenidas.

`evaluate_policy` ejecuta varios episodios y calcula el retorno descontado
promedio junto con su margen de error.
"""

from __future__ import annotations

import numpy as np


def rollout(env, policy, render: bool = False, seed: int | None = None):
    """Ejecuta un episodio completo utilizando una politica.

    La politica debe proporcionar el metodo `get_action(state)` para
    seleccionar una accion a partir del estado actual.

    Args:
        env: Ambiente donde se ejecutara el episodio.
        policy: Politica utilizada para seleccionar las acciones.
        render: Indica si se deben guardar los frames del episodio.
        seed: Semilla utilizada al reiniciar el ambiente.

    Returns:
        Si `render` es False:
            (states, actions, rewards)

        Si `render` es True:
            (frames, states, actions, rewards)
    """  

    # Se reinicia el ambiente y se obtiene el estado inicial.
    initial_state, _ = env.reset(seed=seed)

    # Se guarda el estado inicial y se preparan las listas del episodio.
    states = [initial_state]
    actions = []
    rewards = []
    frames = []

    done = False

    # Se ejecutan acciones hasta que el episodio termine.
    while not done:
        # Se guarda el estado visual antes de ejecutar la siguiente accion.
        if render:
            frames.append(env.render())

        # La politica selecciona una accion usando el estado actual.
        action = policy.get_action(states[-1])

        # Se ejecuta la accion y se obtiene el resultado.
        next_state, reward, terminated, truncated, _ = env.step(action)

        # El episodio termina al alcanzar un estado terminal
        # o al llegar al limite de pasos.
        done = terminated or truncated

        # Registrar la accion, recompensa y nuevo estado.
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)

    # Devolver los frames cuando se solicito renderizado.
    if render:
        return frames, states, actions, rewards
    return states, actions, rewards


def evaluate_policy(env, policy, n_runs: int = 100, discount_factor: float = 1.0,
                     display_result: bool = False) -> tuple[float, float]:
    """Evalua una politica mediante multiples episodios.

    Para cada episodio calcula el retorno descontado. Luego obtiene
    el promedio de los retornos y su margen de error para un intervalo
    de confianza del 95%.

    Args:
        env: Ambiente donde se ejecutara la politica.
        policy: Politica que se desea evaluar.
        n_runs: Numero de episodios que se ejecutaran.
        discount_factor: Factor de descuento utilizado para calcular el retorno.            
        display_result: Indica si se debe mostrar el intervalo obtenido.

    Returns:
        Una tupla con:
            - Retorno descontado promedio.
            - Margen de error del intervalo de confianza.
    """
    # Se inicializan los valores utilizados para calcular la media y
    # la varianza de forma incremental.    
    mean_discounted_return = 0.0
    sum_squared_diffs = 0.0

    # Se ejecutan varios episodios para evaluar la politica.
    for i in range(n_runs):
        _, _, rewards = rollout(env, policy, render=False)

         # Se calcula el retorno descontado del episodio.
        discounted_return = sum(
            (discount_factor ** h) * r for h, r in enumerate(rewards)
        )

        # Se actualiza la media y la varianza utilizando Welford.
        diff_from_old_mean = discounted_return - mean_discounted_return
        mean_discounted_return += diff_from_old_mean / (i + 1)
        diff_from_new_mean = discounted_return - mean_discounted_return
        sum_squared_diffs += diff_from_old_mean * diff_from_new_mean

    # Se calcula la desviacion estandar de los retornos
    if n_runs > 1:
        std_dev = (sum_squared_diffs / (n_runs - 1)) ** 0.5
    else:
        std_dev = 0.0    

    # Se calcula el margen de error para un intervalo de confianza del 95%.
    if n_runs > 0:
        margin_of_error = 1.96 * (std_dev / np.sqrt(n_runs))
    else:
        margin_of_error = 0.0   

    # Se muestra el intervalo si fue solicitado.
    if display_result:
        lo, hi = mean_discounted_return - margin_of_error, mean_discounted_return + margin_of_error
        print(f"J^pi_gamma in [{lo:.2f}, {hi:.2f}] w.p. 0.95  (n_runs={n_runs})")

    return mean_discounted_return, margin_of_error
