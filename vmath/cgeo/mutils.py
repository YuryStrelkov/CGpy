from math import sqrt, pi, exp
from typing import Tuple
from numba import prange
import numpy as np
import numba

_2pij = 2j * pi

numerical_precision: float = 1e-9
Vector2 = Tuple[float, float]


@numba.njit(fastmath=True)
def gauss_test_surf(n: int) -> np.ndarray:
    """
    Создаёт двумерный массив значений функции z = exp(-(x^2 + y^2)).\n
    :param n: количество точек на сторону
    :return:
    """
    gauss = np.zeros((n, n,), dtype=float)
    dx = 3.0 / (n - 1)
    for i in prange(n * n):
        row = i // n
        col = i % n
        x_ = dx * col - 1.5
        y_ = dx * row - 1.5
        gauss[row, col] = exp(-(x_ * x_ + y_ * y_))
    return gauss


@numba.njit(fastmath=True)
def square_equation(a: float, b: float, c: float) -> Tuple[bool, float, float]:
    det: float = b * b - 4.0 * a * c
    if det < 0.0:
        return False, 0.0, 0.0
    det = sqrt(det)
    return True, (-b + det) / 2.0 / a, (-b - det) / 2.0 / a


@numba.njit(fastmath=True)
def _in_range(val: float, x_0: float, x_1: float) -> bool:
    """
    Проверяет вхождение числа в диапазон.\n
    :param val: число
    :param x_0: левая граница диапазона
    :param x_1: правая граница диапазона
    :return:
    """
    if val < x_0:
        return False
    if val > x_1:
        return False
    return True


@numba.njit(fastmath=True)
def clamp(val: float, min_: float, max_: float) -> float:
    """
    :param val: значение
    :param min_: минимальная граница
    :param max_: максимальная граница
    :return: возвращает указанное значение val в границах от min до max
    """
    if val < min_:
        return min_
    if val > max_:
        return max_
    return val


@numba.njit(fastmath=True)
def dec_to_rad_pt(x: float, y: float) -> Vector2:
    """
    Переводи пару координат из декартовой системы в полярную.\n
    :param x: x - координата
    :param y: y - координата
    :return:  кординаты rho и phi в полярной системе
    """
    return np.sqrt(x * x + y * y), np.arctan2(y, x)


@numba.njit(fastmath=True)
def rad_to_dec_pt(rho: float, phi: float) -> Vector2:
    """
    Переводи пару координат из полярной системы в декартову.\n
    :param rho: rho - радиус
    :param phi: phi - угол
    :return:  кординаты x и y в декартову системе
    """
    return rho * np.cos(phi), rho * np.sin(phi)


@numba.njit(parallel=True)
def dec_to_rad(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Переводи пару массивов координат из декартовой системы в полярную.\n
    :param x: x - массив координат
    :param y: y - массив координат
    :return:  массивы кординат rho и phi в полярной системе
    """
    if x.size != y.size:
        raise Exception("dec_to_rad :: x.size != y.size")
    if x.ndim != y.ndim:
        raise Exception("dec_to_rad :: x.ndim != y.ndim")

    if x.ndim == 2:
        if x.ndim == 2:
            for i in prange(x.shape[0]):
                for j in range(x.shape[1]):
                    x[i, j], y[i, j] = dec_to_rad_pt(x[i, j], y[i, j])
        return x, y

    if x.ndim == 1:
        for i in prange(x.shape[0]):
            x[i], y[i] = dec_to_rad_pt(x[i], y[i])
        return x, y

    raise Exception("dec_to_rad :: x and y has to be 1 or 2 dimensional")


@numba.njit(parallel=True)
def rad_to_dec(rho: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Переводи пару массивов координат из полярной системы в декартову.\n
    :param rho: rho - массив радиусов
    :param phi: phi - массив уголов
    :return:  массивы кординат x и y в декартову системе
    """
    if rho.size != phi.size:
        raise Exception("rad_to_dec :: rho.size != phi.size")

    if rho.ndim != phi.ndim:
        raise Exception("rad_to_dec :: rho.ndim != phi.ndim")

    if rho.ndim == 2:
        for i in prange(rho.shape[0]):
            for j in range(rho.shape[1]):
                rho[i, j], phi[i, j] = rad_to_dec_pt(rho[i, j], phi[i, j])
        return rho, phi

    if rho.ndim == 1:
        for i in prange(rho.shape[0]):
            rho[i], phi[i] = rad_to_dec_pt(rho[i], phi[i])
        return rho, phi

    raise Exception("rad_to_dec :: rho and phi has to be 1 or 2 dimensional")


@numba.njit(fastmath=True)
def _index_calc(index: int, indices_range: int) -> int:
    if index < 0:
        return indices_range - 1 + index % indices_range
    return index % indices_range


@numba.njit(fastmath=True)
def compute_derivatives_2_at_pt(points: np.ndarray, row: int, col: int) -> Tuple[float, float, float]:
    """
    Вычисляет произодные по х, по y и по xy. Используется центральный разностный аналог
    :param points: двумерный список узловых точек
    :param row: индекс строки точки из points для которой считаем произовдные
    :param col: индекс столбца точки из points для которой считаем произовдные
    :return: (df/dx, df/dy, df/dx/dy)
    """
    if points.ndim != 2:
        raise RuntimeError("compute_derivatives_2_at_pt :: points array has to be 2 dimensional")

    rows, colons = points.shape

    if not _in_range(row, 0, rows - 1):
        return 0.0, 0.0, 0.0

    if not _in_range(col, 0, colons - 1):
        return 0.0, 0.0, 0.0

    row_1 = _index_calc(row + 1, rows)  # min(rows - 1, row_ + 1)
    row_0 = _index_calc(row - 1, rows)  # max(0, row_ - 1)

    col_1 = _index_calc(col + 1, colons)  # col_1 = min(colons - 1, col_ + 1)
    col_0 = _index_calc(col - 1, colons)  # col_0 = max(0, col_ - 1)

    return (points[row,   col_1] - points[row,   col_0]) * 0.5, \
           (points[row_1, col]   - points[row_0, col]) * 0.5, \
           (points[row_1, col_1] - points[row_1, col_0]) * 0.25 - \
           (points[row_0, col_1] - points[row_0, col_0]) * 0.25


@numba.njit(fastmath=True)
def compute_derivatives_at_pt(points: np.ndarray, row: int, col: int) -> Vector2:
    """
    Вычисляет произодные по х, по y. Используется центральный разностный аналог
    :param points: двумерный список узловых точек
    :param row: индекс строки точки из points для которой считаем произовдные
    :param col: индекс столбца точки из points для которой считаем произовдные
    :return: (df/dx, df/dy, df/dx/dy)
    """
    if points.ndim != 2:
        raise RuntimeError("compute_derivatives_at_pt :: points array has to be 2 dimensional")

    rows, colons = points.shape

    if not _in_range(row, 0, rows - 1):
        return 0.0, 0.0

    if not _in_range(col, 0, colons - 1):
        return 0.0, 0.0

    row_1 = _index_calc(row + 1, rows)  # min(rows - 1, row_ + 1)
    row_0 = _index_calc(row - 1, rows)  # max(0, row_ - 1)

    col_1 = _index_calc(col + 1, colons)  # col_1 = min(colons - 1, col_ + 1)
    col_0 = _index_calc(col - 1, colons)  # col_0 = max(0, col_ - 1)

    return (points[row,   col_1] - points[row,   col_0]) * 0.5, \
           (points[row_1, col]   - points[row_0, col]) * 0.5


@numba.njit(fastmath=True, parallel=True)
def compute_derivatives_2(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляет произодные по х, по y и по xy. Используется центральный разностный аналог
    :param points: двумерный список узловых точек
    :return: (df/dx, df/dy, df/dx/dy), каждый элемент np.ndarray
    """
    if points.ndim != 2:
        raise RuntimeError("compute_derivatives_2 :: points array has to be 2 dimensional")

    rows, colons = points.shape

    points_dx  = np.zeros_like(points)
    points_dy  = np.zeros_like(points)
    points_dxy = np.zeros_like(points)

    for i in prange(points.size):
        row_ = int(i / colons)
        row_1 = min(rows - 1, row_ + 1)
        row_0 = max(0, row_ - 1)

        col_ = i % colons
        col_1 = min(colons - 1, col_ + 1)
        col_0 = max(0, col_ - 1)

        points_dx[row_, col_] = (points[row_, col_1] - points[row_, col_0]) * 0.5

        points_dy[row_, col_] = (points[row_1, col_] - points[row_0, col_]) * 0.5

        dx_1 = (points[row_1, col_1] -
                points[row_1, col_0]) * 0.25
        dx_2 = (points[row_0, col_1] -
                points[row_0, col_0]) * 0.25
        points_dxy[row_, col_] = (dx_2 - dx_1)

    return points_dx, points_dy, points_dxy


@numba.njit(fastmath=True, parallel=True)
def compute_derivatives(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет произодные по х, по y и по xy. Используется центральный разностный аналог
    :param points: двумерный список узловых точек
    :return: (df/dx, df/dy), каждый элемент np.ndarray
    """
    if points.ndim != 2:
        raise RuntimeError("compute_derivatives_2 :: points array has to be 2 dimensional")

    rows, colons = points.shape

    points_dx  = np.zeros_like(points)
    points_dy  = np.zeros_like(points)

    for i in prange(points.size):
        row_ = int(i / colons)
        row_1 = min(rows - 1, row_ + 1)
        row_0 = max(0, row_ - 1)

        col_ = i % colons
        col_1 = min(colons - 1, col_ + 1)
        col_0 = max(0, col_ - 1)

        points_dx[row_, col_] = (points[row_, col_1] - points[row_, col_0]) * 0.5

        points_dy[row_, col_] = (points[row_1, col_] - points[row_0, col_]) * 0.5

    return points_dx, points_dy


@numba.njit(fastmath=True, parallel=True)
def compute_normals(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2:
        raise RuntimeError("compute_derivatives_2 :: points array has to be 2 dimensional")

    rows, cols = points.shape

    points_n = np.zeros((rows, cols, 3,), dtype=float)

    for i in prange(points.size):
        row_ = int(i / cols)
        row_1 = min(rows - 1, row_ + 1)
        row_0 = max(0, row_ - 1)

        col_ = i % cols
        col_1 = min(cols - 1, col_ + 1)
        col_0 = max(0, col_ - 1)

        _dx = (points[row_, col_1] - points[row_, col_0]) * 0.5

        _dy = (points[row_1, col_] - points[row_0, col_]) * 0.5

        _rho = np.sqrt(1.0 + _dx * _dx + _dy * _dy)

        points_n[row_, col_, 0] = _dx / _rho

        points_n[row_, col_, 1] = _dy / _rho

        points_n[row_, col_, 2] = 1.0 / _rho

    return points_n


@numba.njit(fastmath=True)
def _fast_fourier_transform(signal: np.ndarray) -> None:
    _n = signal.size

    _k = _n

    theta_t = 3.14159265358979323846264338328 / _n

    phi_t = complex(np.cos(theta_t), -np.sin(theta_t))

    while _k > 1:
        _t_n = _k
        _k >>= 1
        phi_t = phi_t * phi_t
        _t = 1.0
        for _l in range(0, _k):
            for a in range(_l, _n, _t_n):
                b = a + _k
                t = signal[a] - signal[b]
                signal[a] += signal[b]
                signal[b] = t * _t
            _t *= phi_t

    _m = int(np.log2(_n))

    for a in range(_n):
        b = a
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1))
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2))
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4))
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8))
        b = ((b >> 16) | (b << 16)) >> (32 - _m)
        if b > a:
            t = signal[a]
            signal[a] = signal[b]
            signal[b] = t


@numba.njit(fastmath=True)
def fft(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    _x = x
    if do_copy:
        _x = x.copy()
    _fast_fourier_transform(_x)
    return _x


@numba.njit(fastmath=True)
def ifft(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    _x = x
    if do_copy:
        _x = x.copy()
    _x = _x.conjugate()
    _fast_fourier_transform(_x)
    _x = _x.conjugate()
    _x /= _x.size
    return _x


@numba.njit(parallel=True)
def fft2(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("fft2 :: x.ndim != 2")
    _x = img_to_pow_2_size(x)
    if do_copy:
        _x = img_to_pow_2_size(x.copy())

    for i in numba.prange(_x.shape[0]):
        _x[i, :] = fft(_x[i, :])

    for i in numba.prange(_x.shape[1]):
        _x[:, i] = fft(_x[:, i])

    return _x


@numba.njit(parallel=True)
def ifft2(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("fft2 :: x.ndim != 2")
    _x = x
    if do_copy:
        _x = x.copy()

    for i in numba.prange(_x.shape[0]):
        _x[i, :] = ifft(_x[i, :])

    for i in numba.prange(_x.shape[1]):
        _x[:, i] = ifft(_x[:, i])

    return _x


@numba.njit(fastmath=True)
def img_crop(img: np.ndarray, rows_bound: Vector2, cols_bound: Vector2) -> np.ndarray:
    if img.ndim == 1:
        raise RuntimeError("img_crop:: image has to be 2-dimensional, but 1-dimensional was given...")
    rows, cols, _ = img.shape
    x_min = max(cols_bound[0], 0)
    x_max = max(cols_bound[0], cols)
    y_min = max(rows_bound[1], 0)
    y_max = max(rows_bound[1], rows)
    return img[y_min: y_max, x_min: x_max, :]


@numba.njit(fastmath=True)
def img_to_pow_2_size(img: np.ndarray) -> np.ndarray:
    rows, cols, _ = img.shape
    rows2, cols2 = 2 ** int(np.log2(rows)), 2 ** int(np.log2(cols))
    if rows == rows2 and cols2 == cols:
        return img
    return img_crop(img, ((rows - rows2) // 2, (rows + rows2) // 2),
                         ((cols - cols2) // 2, (cols + cols2) // 2))

