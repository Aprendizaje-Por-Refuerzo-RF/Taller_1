"""Funciones de renderizado y animacion para el ambiente Milan Taxi.

Este modulo se encarga de:

- Dibujar el ambiente utilizando pygame.
- Generar frames RGB a partir del estado del taxi.
- Cerrar los recursos utilizados por pygame.
- Crear y mostrar animaciones de una secuencia de frames.
- Guardar una secuencia de frames como GIF.

El renderizado visual utiliza el mapa y los recursos graficos de la
configuracion original de Milan Taxi.
"""

from __future__ import annotations

from os import path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
from gymnasium.error import DependencyNotInstalled

# Se aumenta el limite permitido para animaciones embebidas en notebooks.
mpl.rcParams['animation.embed_limit'] = 500.0

# ---------------------------------------------------------------------------
# Configuracion del mapa y recursos graficos
# ---------------------------------------------------------------------------
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

WINDOW_SIZE = (550, 350)
# Posiciones de los cuatro puntos de interes del mapa.
LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

# Ajustes individuales para el tamaño y posicion de cada landmark.
# Formato: (escala, desplazamiento_x, desplazamiento_y).
LANDMARK_TWEAKS = [
    (1.2, 0, -50),   # [0] Politecnico
    (1.5, 0, -50),   # [1] Duomo
    (1.5, 0, 10),    # [2] Castello
    (2.0, 0, 10),    # [3] San Siro
]

# Indice utilizado para representar que el pasajero esta dentro del taxi.
PASS_IN_TAXI = 4

# Directorio donde se encuentran los recursos graficos.
IMGS_DIR = path.join(path.dirname(__file__), "imgs", "custom_taxi")

# ---------------------------------------------------------------------------
# Cache de recursos graficos
# ---------------------------------------------------------------------------
_RENDER_CACHE = {
    "window": None,
    "assets_loaded": False,
    "taxi_imgs": None,
    "passenger_img": None,
    "median_horiz": None,
    "median_vert": None,
    "background_img": None,
    "loc_imgs": None,  # los 4 landmarks de Milán
    "taxi_orientation": 0,
}


def _get_surf_loc(map_loc, cell_size):
    """Convierte una posición de la grilla a una posición en píxeles."""
    return ((map_loc[1] * 2 + 1) * cell_size[0], (map_loc[0] + 1) * cell_size[1])


def render_taxi_frame(state, desc, lastaction, window_size, locs):
    """Genera un frame RGB del estado actual utilizando pygame.

    Crea la ventana y carga los recursos graficos la primera vez que
    se ejecuta. Los recursos se almacenan en cache para reutilizarlos
    en los siguientes frames.

    Args:
        state: estado del taxi con posicion y pasajero.
        desc: representacion de la grilla.
        lastaction: ultima acción ejecutada.
        window_size: tamaño de la superficie de pygame.
        locs: posiciones de los puntos de interes.

    Returns:
        Array NumPy con el frame RGB.
    """
    # pygame se importa aqui para que el resto del proyecto pueda
    # utilizarse aunque pygame no este instalado.
    try:
        import pygame
    except ImportError as e:
        raise DependencyNotInstalled(
            "pygame no está instalado. Agrega 'pygame' a las dependencias "
            "(pyproject.toml) y corre `uv sync`."
        ) from e
    
    # Se crea la superficie de pygame la primera vez que se renderiza.
    if _RENDER_CACHE["window"] is None:
        pygame.init()
        pygame.display.set_caption("Taxi - Milan Edition")
        _RENDER_CACHE["window"] = pygame.Surface(window_size)

    window = _RENDER_CACHE["window"]

    # Se calcula el tamaño en pixeles de cada celda de la grilla.
    cell_size = (window_size[0] / desc.shape[1], window_size[1] / desc.shape[0])

    # Se cargan los recursos graficos una sola vez.
    if not _RENDER_CACHE["assets_loaded"]:
        load = lambda name: pygame.transform.scale(
            pygame.image.load(path.join(IMGS_DIR, name)), cell_size
        )

        # Imagenes del taxi para cada orientacion.
        _RENDER_CACHE["taxi_imgs"] = [
            load(n) for n in ("cab_front.png", "cab_rear.png", "cab_right.png", "cab_left.png")
        ]

        # Imagenes del pasajero y del fondo.
        _RENDER_CACHE["passenger_img"] = load("passenger.png")
        _RENDER_CACHE["background_img"] = load("taxi_background.png")

        # Elementos graficos de las paredes horizontales
        _RENDER_CACHE["median_horiz"] = [
            load(n) for n in ("gridworld_median_left.png", "gridworld_median_horiz.png", "gridworld_median_right.png")
        ]

        # Elementos graficos de las paredes verticales.
        _RENDER_CACHE["median_vert"] = [
            load(n) for n in ("gridworld_median_top.png", "gridworld_median_vert.png", "gridworld_median_bottom.png")
        ]

        raw_loc_names = ["polimi_pixel.png", "duomo_pixel.png", "castello_pixel.png", "sansiro_pixel.png"]
        _RENDER_CACHE["loc_imgs"] = []
        for i, name in enumerate(raw_loc_names):
            scale_factor = LANDMARK_TWEAKS[i][0]
            raw_img = pygame.image.load(path.join(IMGS_DIR, name))
            new_size = (int(cell_size[0] * scale_factor), int(cell_size[1] * scale_factor))
            _RENDER_CACHE["loc_imgs"].append(pygame.transform.scale(raw_img, new_size))

        _RENDER_CACHE["assets_loaded"] = True

    # -----------------------------------------------------------------------
    # Dibujar fondo y paredes
    # -----------------------------------------------------------------------
    for y in range(desc.shape[0]):
        for x in range(desc.shape[1]):
            cell = (x * cell_size[0], y * cell_size[1])
            window.blit(_RENDER_CACHE["background_img"], cell)
            ch = desc[y][x]
            if ch == b"|":
                if y == 0 or desc[y - 1][x] != b"|":
                    window.blit(_RENDER_CACHE["median_vert"][0], cell)
                elif y == desc.shape[0] - 1 or desc[y + 1][x] != b"|":
                    window.blit(_RENDER_CACHE["median_vert"][2], cell)
                else:
                    window.blit(_RENDER_CACHE["median_vert"][1], cell)
            elif ch == b"-":
                if x == 0 or desc[y][x - 1] != b"-":
                    window.blit(_RENDER_CACHE["median_horiz"][0], cell)
                elif x == desc.shape[1] - 1 or desc[y][x + 1] != b"-":
                    window.blit(_RENDER_CACHE["median_horiz"][2], cell)
                else:
                    window.blit(_RENDER_CACHE["median_horiz"][1], cell)

    # -----------------------------------------------------------------------
    # Dibujar landmarks
    # -----------------------------------------------------------------------
    for i, loc_cell in enumerate(locs):
        sx, sy = _get_surf_loc(loc_cell, cell_size)
        img_to_draw = _RENDER_CACHE["loc_imgs"][i]

        center_x_offset = (cell_size[0] - img_to_draw.get_width()) / 2
        center_y_offset = (cell_size[1] - img_to_draw.get_height()) / 2

        final_x = sx + center_x_offset + LANDMARK_TWEAKS[i][1]
        final_y = sy + center_y_offset + LANDMARK_TWEAKS[i][2]

        window.blit(img_to_draw, (final_x, final_y))

    # -----------------------------------------------------------------------
    # Dibujar pasajero y taxi
    # -----------------------------------------------------------------------
    taxi_row, taxi_col = state["row"], state["col"]
    pass_idx = state["pass_idx"]

    if pass_idx is not None and pass_idx < 4:
        window.blit(_RENDER_CACHE["passenger_img"], _get_surf_loc(locs[pass_idx], cell_size))

    if lastaction in (0, 1, 2, 3):
        _RENDER_CACHE["taxi_orientation"] = lastaction

    taxi_location = _get_surf_loc((taxi_row, taxi_col), cell_size)
    taxi_img_to_draw = _RENDER_CACHE["taxi_imgs"][_RENDER_CACHE["taxi_orientation"]]

    window.blit(taxi_img_to_draw, taxi_location)

    return np.transpose(np.array(pygame.surfarray.pixels3d(window)), axes=(1, 0, 2))


def close_taxi_render():
    """Cierra pygame y libera los recursos utilizados por el renderizado."""
    if _RENDER_CACHE["window"] is None:
        return
    import pygame

    pygame.display.quit()
    pygame.quit()
    _RENDER_CACHE["window"] = None
    _RENDER_CACHE["assets_loaded"] = False


def render_taxi(state, lastaction):
    """Genera el frame RGB correspondiente al estado del taxi.

    Convierte el estado utilizado por `MilanTaxiEnv` al formato esperado
    por `render_taxi_frame` y adapta las acciones del entorno a las
    orientaciones utilizadas por los sprites del taxi.

    Si no existe un estado actual, retorna `None`.
    """
    if state is None:
        return None

    action_renderer_map = {0: 1, 1: 2, 2: 0, 3: 3}
    mapped_lastaction = action_renderer_map.get(lastaction, lastaction)

    render_state_dict = {
        "row": state[0],
        "col": state[1],
        "pass_idx": state[2],
        "dest_idx": state[3],
    }

    return render_taxi_frame(
        state=render_state_dict, desc=np.asarray(MAP, dtype="c"), lastaction=mapped_lastaction,
        window_size=WINDOW_SIZE, locs=LOCS,
    )


# ---------------------------------------------------------------------------
# Animación de una secuencia de frames
# ---------------------------------------------------------------------------
def _build_animation(frames, interval=200):
    """Construye una animacion de Matplotlib a partir de una lista de frames.

    Esta funcion es utilizada tanto para mostrar la animacion en un notebook
    como para guardarla como GIF.
    """
    h, w = frames[0].shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    img_plot = ax.imshow(frames[0])

    def update(frame_idx):
        img_plot.set_data(frames[frame_idx])
        return [img_plot]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=True)
    return fig, ani


def animate_frames(frames, interval=200):
    """Muestra una secuencia de frames como animacion en un notebook.

    Args:
        frames: lista de arrays NumPy con formato (H, W, 3).
        interval: tiempo en milisegundos entre frames.
    """
    fig, ani = _build_animation(frames, interval=interval)
    plt.close(fig)
    display(HTML("<center>" + ani.to_jshtml(default_mode="loop") + "</center>"))


def save_animation(frames, output_path, fps=5):
    """Guarda una secuencia de frames como un GIF.

    Args:
        frames: lista de arrays NumPy con formato (H, W, 3).
        output_path: ruta donde se guardará el GIF.
        fps: cantidad de frames por segundo.
    """
    
    fig, ani = _build_animation(frames, interval=1000 / fps)
    ani.save(str(output_path), writer="pillow", fps=fps)
    plt.close(fig)
