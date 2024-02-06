from math import sqrt, pi, exp
from typing import Tuple, Any, List
import numpy as np
import operator

from cgeo.tris_mesh.fourier import img_to_pow_2_size

_2pij = 2j * pi

numerical_precision: float = 1e-9
Vector2 = Tuple[float, float]
Vector3 = Tuple[float, float, float]


def is_close(value_a: float, value_b: float, tolerance: float) -> bool:
    return abs(value_a - value_b) < tolerance


def smooth_step(value: float, bound_1: float, bound_2: float) -> float:
    if value <= bound_1:
        return bound_1
    if value >= bound_2:
        return bound_2
    x = (value - bound_1) / (bound_2 - bound_1)
    return x * x * (3 - 2 * x)



"""

def softmax(z):
    z -= np.max(z)
    return np.exp(z) / np.sum(np.exp(z))
    
def soft_max(value1: float, value1: float) -> float:
    if value <= bound_1:
        return bound_1
    if value >= bound_2:
        return bound_2
    x = (value - bound_1) / (bound_2 - bound_1)
    return x * x * (3 - 2 * x)
"""


def list_min(values: list) -> Tuple[int, Any]:
    return min(enumerate(values), key=operator.itemgetter(1))


def list_max(values: list) -> Tuple[int, Any]:
    return max(enumerate(values), key=operator.itemgetter(1))


# @numba.njit(fastmath=True)
def signum(value) -> float:
    """
    :param value:
    :return: возвращает знак числа
    """
    if value < 0:
        return -1.0
    return 1.0


# @numba.njit(fastmath=True)
def gauss_test_surf(n: int) -> np.ndarray:
    """
    Создаёт двумерный массив значений функции z = exp(-(x^2 + y^2)).\n
    :param n: количество точек на сторону
    :return:
    """
    gauss = np.zeros((n, n,), dtype=float)
    dx = 3.0 / (n - 1)
    half_pi = np.pi * 0.5
    inv_sqrt_2pi = 1.0 / np.sqrt(np.pi * 2.0)
    for i in range(n * n):
        row, col = divmod(i, n)
        x_ = dx * col - half_pi
        y_ = dx * row - half_pi
        gauss[row, col] = exp(-(x_ * x_ + y_ * y_) * 0.5) * inv_sqrt_2pi
    return gauss


# @numba.njit(fastmath=True)
def square_equation(a: float, b: float, c: float) -> Tuple[bool, float, float]:
    det: float = b * b - 4.0 * a * c
    if det < 0.0:
        return False, 0.0, 0.0
    det = sqrt(det)
    return True, (-b + det) / 2.0 / a, (-b - det) / 2.0 / a


# @numba.njit(fastmath=True)
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


# @numba.njit(fastmath=True)
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


# @numba.njit(fastmath=True)
def dec_to_rad_pt(x: float, y: float) -> Vector2:
    """
    Переводи пару координат из декартовой системы в полярную.\n
    :param x: x - координата
    :param y: y - координата
    :return:  кординаты rho и phi в полярной системе
    """
    return np.sqrt(x * x + y * y), np.arctan2(y, x)


# @numba.njit(fastmath=True)
def rad_to_dec_pt(rho: float, phi: float) -> Vector2:
    """
    Переводи пару координат из полярной системы в декартову.\n
    :param rho: rho - радиус
    :param phi: phi - угол
    :return:  кординаты x и y в декартову системе
    """
    return rho * np.cos(phi), rho * np.sin(phi)


# @numba.njit(parallel=True)
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
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x[i, j], y[i, j] = dec_to_rad_pt(x[i, j], y[i, j])
        return x, y

    if x.ndim == 1:
        for i in range(x.shape[0]):
            x[i], y[i] = dec_to_rad_pt(x[i], y[i])
        return x, y

    raise Exception("dec_to_rad :: x and y has to be 1 or 2 dimensional")


# @numba.njit(parallel=True)
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
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                rho[i, j], phi[i, j] = rad_to_dec_pt(rho[i, j], phi[i, j])
        return rho, phi

    if rho.ndim == 1:
        for i in range(rho.shape[0]):
            rho[i], phi[i] = rad_to_dec_pt(rho[i], phi[i])
        return rho, phi

    raise Exception("rad_to_dec :: rho and phi has to be 1 or 2 dimensional")


def _seg_quad_interp(dp1: float, p1: float, p2: float, n_points: int = 9) -> Tuple[List[float], float]:
    dt = 1.0 / (n_points - 1)
    x = [(p2 - p1 - dp1) * (dt * i)**2 + dp1 * dt * i + p1 for i in range(n_points)]
    return x, (x[-1] - x[-2]) / dt


def _seg_quad_interp2(dp1: Vector2, p1: Vector2, p2: Vector2, n_points: int = 9) -> Tuple[List[Vector2], Vector2]:
    dt = 1.0 / (n_points - 1)
    points: List[Vector2] = []
    for i in range(n_points):
        t_curr = i * dt
        points.append(((p2[0] - p1[0] - dp1[0]) * t_curr**2 + dp1[0] * t_curr + p1[0],
                       (p2[1] - p1[1] - dp1[1]) * t_curr**2 + dp1[1] * t_curr + p1[1]))
    return points, ((points[-1][0] - points[-2][0]) / dt, (points[-1][1] - points[-2][1]) / dt)


def _seg_quad_interp3(dp1: Vector3, p1: Vector3, p2: Vector3, n_points: int = 9) -> Tuple[List[Vector3], Vector3]:
    dt = 1.0 / (n_points - 1)
    points: List[Vector3] = []
    for i in range(n_points):
        t_curr = i * dt
        points.append(((p2[0] - p1[0] - dp1[0]) * t_curr**2 + dp1[0] * t_curr + p1[0],
                       (p2[1] - p1[1] - dp1[1]) * t_curr**2 + dp1[1] * t_curr + p1[1],
                       (p2[2] - p1[2] - dp1[2]) * t_curr**2 + dp1[2] * t_curr + p1[2]))

    return points, ((points[-1][0] - points[-2][0]) / dt,
                    (points[-1][1] - points[-2][1]) / dt,
                    (points[-1][2] - points[-2][2]) / dt )


def quad_interpolate_line(points: List[float], start_derivative: float = 1.0, segment_steps: int = 32) -> List[float]:
    n_points = len(points)
    xs = []
    dp = start_derivative
    for i in range(0, n_points-1):
        p1 = points[i]
        p2 = points[min(i + 1, n_points - 1)]
        x, dp = _seg_quad_interp(dp, p1, p2, segment_steps)
        xs.extend(x)
    return xs


def quad_interpolate_line2(points: List[Vector2], start_derivative: Vector2 = (1, 0), segment_steps: int = 32) ->\
        List[Vector2]:
    n_points = len(points)
    xs = []
    dp = start_derivative
    for i in range(0, n_points-1):
        p1 = points[i]
        p2 = points[min(i + 1, n_points - 1)]
        x, dp = _seg_quad_interp2(dp, p1, p2, segment_steps)
        xs.extend(x)
    return xs


def quad_interpolate_line3(points: List[Vector3], start_derivative: Vector3 = (1, 1, 1), segment_steps: int = 32) ->\
        List[Vector3]:
    n_points = len(points)
    xs = []
    dp = start_derivative
    for i in range(0, n_points-1):
        p1 = points[i]
        p2 = points[min(i + 1, n_points - 1)]
        x, dp = _seg_quad_interp3(dp, p1, p2, segment_steps)
        xs.extend(x)
    return xs


# @numba.njit(fastmath=True)
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

    row_1 =  min(rows - 1, row + 1)
    row_0 =  max(0, row - 1)

    col_1 = min(colons - 1, col + 1)
    col_0 = max(0, col - 1)

    return (points[row,   col_1] - points[row, col_0]) * 0.5, \
           (points[row_1, col]   - points[row_0, col]) * 0.5, \
           (points[row_1, col_1] - points[row_1, col_0]) * 0.25 - \
           (points[row_0, col_1] - points[row_0, col_0]) * 0.25


# @numba.njit(fastmath=True)
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

    row_1 =  min(rows - 1, row + 1)
    row_0 =  max(0, row - 1)

    col_1 = min(colons - 1, col + 1)
    col_0 = max(0, col - 1)

    return (points[row, col_1] - points[row, col_0]) * 0.5, \
           (points[row_1, col] - points[row_0, col]) * 0.5


# @numba.njit(fastmath=True, parallel=True)
def compute_derivatives_2(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляет произодные по х, по y и по xy. Используется центральный разностный аналог
    :param points: двумерный список узловых точек
    :return: (df/dx, df/dy, df/dx/dy), каждый элемент np.ndarray
    """
    if points.ndim != 2:
        raise RuntimeError("compute_derivatives_2 :: points array has to be 2 dimensional")

    rows, colons = points.shape

    points_dx = np.zeros_like(points)
    points_dy = np.zeros_like(points)
    points_dxy = np.zeros_like(points)

    for i in range(points.size):
        row_, col_ = divmod(i, colons)

        row_1 = min(rows - 1, row_ + 1)
        row_0 = max(0, row_ - 1)

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


# @numba.njit(fastmath=True, parallel=True)
def compute_derivatives(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет произодные по х, по y и по xy. Используется центральный разностный аналог
    :param points: двумерный список узловых точек
    :return: (df/dx, df/dy), каждый элемент np.ndarray
    """
    if points.ndim != 2:
        raise RuntimeError("compute_derivatives_2 :: points array has to be 2 dimensional")

    rows, colons = points.shape

    points_dx = np.zeros_like(points)
    points_dy = np.zeros_like(points)

    for i in range(points.size):
        row_, col_ = divmod(i, colons)

        row_1 = min(rows - 1, row_ + 1)
        row_0 = max(0, row_ - 1)

        col_1 = min(colons - 1, col_ + 1)
        col_0 = max(0, col_ - 1)

        points_dx[row_, col_] = (points[row_, col_1] - points[row_, col_0]) * 0.5

        points_dy[row_, col_] = (points[row_1, col_] - points[row_0, col_]) * 0.5

    return points_dx, points_dy


# @numba.njit(fastmath=True, parallel=True)
def compute_normals(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2:
        raise RuntimeError("compute_derivatives_2 :: points array has to be 2 dimensional")

    rows, cols = points.shape

    points_n = np.zeros((rows, cols, 3,), dtype=float)

    for i in range(points.size):
        row_, col_ = divmod(i, cols)

        row_1 = min(rows - 1, row_ + 1)
        row_0 = max(0, row_ - 1)

        col_1 = min(cols - 1, col_ + 1)
        col_0 = max(0, col_ - 1)

        _dx = (points[row_, col_1] - points[row_, col_0]) * 0.5

        _dy = (points[row_1, col_] - points[row_0, col_]) * 0.5

        _rho = np.sqrt(1.0 + _dx * _dx + _dy * _dy)

        points_n[row_, col_, 0] = _dx / _rho

        points_n[row_, col_, 1] = _dy / _rho

        points_n[row_, col_, 2] = 1.0 / _rho

    return points_n


# @numba.njit(fastmath=True)
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


# @numba.njit(fastmath=True)
def fft(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    _x = x.copy() if do_copy else x
    _fast_fourier_transform(_x)
    return _x


@numba.njit(fastmath=True)
def ifft(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    _x = x.copy() if do_copy else x
    _x = _x.conjugate()
    _fast_fourier_transform(_x)
    _x = _x.conjugate()
    _x /= _x.size
    return _x


@numba.njit(parallel=True)
def fft_2d(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("fft2 :: x.ndim != 2")
    _x = img_to_pow_2_size(x.copy()) if do_copy else img_to_pow_2_size(x)
    for i in range(_x.shape[0]):
        _x[i, :] = fft(_x[i, :])
    for i in range(_x.shape[1]):
        _x[:, i] = fft(_x[:, i])
    return _x


@numba.njit(parallel=True)
def ifft_2d(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("fft2 :: x.ndim != 2")
    _x = x.copy() if do_copy else x
    for i in prange(_x.shape[0]):
        _x[i, :] = ifft(_x[i, :])
    for i in prange(_x.shape[1]):
        _x[:, i] = ifft(_x[:, i])
    return _x


# @numba.njit(fastmath=True)
def img_crop(img: np.ndarray, rows_bound: Vector2, cols_bound: Vector2) -> np.ndarray:
    if img.ndim < 2:
        raise RuntimeError("img_crop:: image has to be 2-dimensional, but 1-dimensional was given...")
    rows, cols, _ = img.shape
    x_min = max(cols_bound[0], 0)
    x_max = max(cols_bound[0], cols)
    y_min = max(rows_bound[1], 0)
    y_max = max(rows_bound[1], rows)
    return img[y_min: y_max, x_min: x_max, :]





# @numba.njit(fastmath=True, parallel=True)
def convolve_2d(image: np.ndarray, core: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise RuntimeError(f"convolve2 :: image.ndim != 2")

    if core.ndim != 2:
        raise RuntimeError(f"convolve2 :: core.ndim != 2")

    if core.shape[0] != core.shape[1]:
        raise RuntimeError(f"convolve2 :: core.shape[0] != core.shape[1]")

    if core.shape[0] % 2 != 1:
        raise RuntimeError(f"convolve2 :: core.shape[0] % 2 != 1")

    _array = np.zeros_like(image)

    rows, cols = image.shape
    c_size_half = core.shape[0] // 2
    i_row: int
    j_col: int
    divider: float

    for _row in range(rows):
        for _col in range(cols):
            divider = 0.0
            value = 0.0
            for f_row in range(-c_size_half, c_size_half):
                for f_col in range(-c_size_half, c_size_half):

                    i_row = f_row + _row
                    i_col = f_col + _col

                    if i_row < 0:
                        continue
                    if i_col < 0:
                        continue
                    if i_row >= rows:
                        continue
                    if i_col >= cols:
                        continue

                    f_val = core[f_row + c_size_half, f_col + c_size_half]
                    value += image[i_row, i_col] * f_val
                    divider += f_val

            if divider != 0.0:
                value /= divider
            if _array[_row, _col] !=  0.0:
                _array[_row, _col] = min(_array[_row, _col], value)
            else:
                _array[_row, _col] = value
    return _array


def gauss_blur(image: np.ndarray, blur_size: int = 9) -> np.ndarray:
    if blur_size % 2 != 1:
        return convolve_2d(image, gauss_test_surf(blur_size + 1))
    return convolve_2d(image, gauss_test_surf(blur_size))


# @numba.njit(fastmath=True, parallel=True)
def median_filter_2d(array_data: np.ndarray, filter_size: int = 15):
    import bisect

    if filter_size % 2 != 1:
        raise Exception("Median filter length must be odd.")

    if array_data.ndim != 2:
        raise Exception("Input must be two-dimensional.")

    indexer = filter_size // 2

    data_final = np.zeros_like(array_data)

    height, width = array_data.shape

    for row in range(height):
        temp = []
        for col in range(width):
            for z in range(filter_size):
                if row + z - indexer < 0 or row + z - indexer > height - 1:
                    for c in range(filter_size):
                        bisect.insort(temp, 0)
                else:
                    if col + z - indexer < 0 or col + indexer > width - 1:
                        bisect.insort(temp, 0)
                        # temp.append(0)
                    else:
                        for k in range(filter_size):
                            bisect.insort(temp, array_data[col + z - indexer][col + k - indexer])
                            #  temp.append(array_data[i + z - indexer][j + k - indexer])
            # temp.sort()
            data_final[row][col] = temp[len(temp) // 2]
            temp.clear()
    del bisect
    return data_final
