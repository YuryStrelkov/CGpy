from typing import List
from rgba import RGBA


def _color_code(red: int, green: int, blue: int) -> str:
    return f"#{''.join('{:02X}'.format(max(min(a, 255), 0)) for a in (red, green, blue))}"


def hex_color_map_quad(map_amount: int = 3) -> List[str]:
    colors: List[str] = []
    dx = 1.0 / (map_amount - 1)
    for i in range(map_amount):
        xi = i * dx
        colors.append(_color_code(int(255 * max(1.0 - (2.0 * xi - 1.0) ** 2, 0)),
                                  int(255 * max(1.0 - (2.0 * xi - 2.0) ** 2, 0)),
                                  int(255 * max(1.0 - (2.0 * xi - 0.0) ** 2, 0))))
    return colors


def hex_color_map_lin(map_amount: int = 3) -> List[str]:
    colors: List[str] = []
    dx = 1.0 / (map_amount - 1)
    for i in range(map_amount):
        xi = i * dx
        colors.append(_color_code(int(255 * max(1.0 - 2.0 * xi, 0.0)),
                                  int(255 *    (1.0 - abs(2.0 * xi - 1.0))),
                                  int(255 * max(2.0 * xi - 1, 0.0))))
    return colors


def color_map_quad(map_amount: int = 32) -> List[RGBA]:
    colors: List[RGBA] = []
    dx = 1.0 / (map_amount - 1)
    for i in range(map_amount):
        xi = i * dx
        colors.append(RGBA(int(255 * max(1.0 - (2.0 * xi - 1.0) ** 2, 0)),
                           int(255 * max(1.0 - (2.0 * xi - 2.0) ** 2, 0)),
                           int(255 * max(1.0 - (2.0 * xi - 0.0) ** 2, 0))))
    return colors


def color_map_lin(map_amount: int = 32) -> List[RGBA]:
    colors: List[RGBA] = []
    dx = 1.0 / (map_amount - 1)
    for i in range(map_amount):
        xi = i * dx
        colors.append(RGBA(int(255 * max(1.0 - 2.0 * xi, 0.0)),
                           int(255 *    (1.0 - abs(2.0 * xi - 1.0))),
                           int(255 * max(2.0 * xi - 1, 0.0))))
    return colors
