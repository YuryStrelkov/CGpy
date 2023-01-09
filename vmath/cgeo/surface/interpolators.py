from cgeo.mutils import compute_derivatives_2_at_pt, compute_derivatives_2, clamp, numerical_precision
from typing import Tuple
from cmath import sqrt
import numpy as np
import numba


_bicubic_poly_coefficients = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              -3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                              -3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0,
                              9.0, -9.0, -9.0, 9.0, 6.0, 3.0, -6.0, -3.0, 6.0, -6.0, 3.0, -3.0, 4.0, 2.0, 2.0, 1.0,
                              -6.0, 6.0, 6.0, -6.0, -4.0, -2.0, 4.0, 2.0, -3.0, 3.0, -3.0, 3.0, -2.0, -1.0, -2.0, -1.0,
                              2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                              -6.0, 6.0, 6.0, -6.0, -3.0, -3.0, 3.0, 3.0, -4.0, 4.0, -2.0, 2.0, -2.0, -2.0, -1.0, -1.0,
                              4.0, -4.0, -4.0, 4.0, 2.0, 2.0, -2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 1.0, 1.0, 1.0, 1.0)
"""
:param _bicubic_poly_coefficients: 
"""


@numba.njit(fastmath=True)
def bi_linear_interp_pt(x: float, y: float, points: np.ndarray, width: float = 1.0, height: float = 1.0) -> float:
    """
    Билинейная иетерполяция точки (x,y)
    :param x: x - координата точки
    :param y: y - координата точки
    :param points: одномерный список узловых точек
    :param width: ширина области интеполяции
    :param height: высота области интеполяции
    :return:
    """
    if points.ndim != 2:
        raise RuntimeError("bi_linear_interp_pt :: points array has to be 2 dimensional")

    rows, cols = points.shape[0], points.shape[0]

    x = clamp(x, 0.0, width)

    y = clamp(y, 0.0, height)

    col_ = int((x / width) * (cols - 1))

    row_ = int((y / height) * (rows - 1))

    col_1 = min(col_ + 1, cols - 1)

    row_1 = min(row_ + 1, rows - 1)

    # q11 = nodes[row_, col_]

    # q00____q01
    # |       |
    # |       |
    # q10____q11

    dx_ = width / (cols - 1.0)
    dy_ = height / (rows - 1.0)

    tx = (x - dx_ * col_) / dx_
    ty = (y - dy_ * row_) / dy_

    q00: float = points[col_,  row_ ]
    q01: float = points[col_1, row_ ]
    q10: float = points[col_,  row_1]
    q11: float = points[col_1, row_1]

    return q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)


@numba.njit(fastmath=True, parallel=True)
def bi_linear_interp(x: np.ndarray, y: np.ndarray, points: np.ndarray,
                     width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """
    Билинейная иетерполяция диапазона точек x, y
    :param x: x - координаты точек
    :param y: y - координаты точек
    :param points: одномерный список узловых точек
    :param width: ширина области интеполяции
    :param height: высота области интеполяции
    :return:
    """
    if points.ndim != 2:
        raise RuntimeError("bi_linear_interp_pt :: points array has to be 2 dimensional")

    rows, cols = points.shape[0], points.shape[0]

    result = np.zeros((y.size, x.size,), dtype=float)

    dx_ = width / (cols - 1.0)

    dy_ = height / (rows - 1.0)

    for i in numba.prange(result.size):

        res_col_ = i % x.size

        res_row_ = i // x.size

        x_ = clamp(x[res_col_], 0.0, width)

        y_ = clamp(y[res_row_], 0.0, height)

        col_ = int((x_ / width) * (cols - 1))

        row_ = int((y_ / height) * (rows - 1))

        col_1 = min(col_ + 1, cols - 1)

        row_1 = min(row_ + 1, rows - 1)

        # q11 = nodes[row_, col_]
        # q00____q01
        # |       |
        # |       |
        # q10____q11

        tx = (x_ - dx_ * col_) / dx_
        ty = (y_ - dy_ * row_) / dy_

        q00: float = points[col_,  row_]
        q01: float = points[col_1, row_]
        q10: float = points[col_,  row_1]
        q11: float = points[col_1, row_1]

        result[res_row_, res_col_] = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)

    return result


@numba.njit(fastmath=True, parallel=True)
def bi_linear_cut(x_0: float, y_0: float, x_1: float, y_1: float, steps_n: int, points: np.ndarray,
                  width: float = 1.0, height: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Счечение интерполируемой би-линейным методом поверхности вдоль прямой через две точки
    :param x_0: x - координата первой точки секущей
    :param y_0: y - координата первой точки секущей
    :param x_1: x - координата второй точки секущей
    :param y_1: y - координата второй точки секущей
    :param steps_n: количество точке вдоль секущей
    :param points: одномерный список узловых точек
    :param width: ширина области интеполяции
    :param height: высота области интеполяции
    :return:
    """
    dx = x_1 - x_0
    dy = y_1 - y_0
    rho = dx * dx + dy * dy

    if rho > numerical_precision:
        rho = sqrt(rho)
        dx /= rho
        dy /= rho
    else:
        dx = 0.0
        dy = 0.0

    dt = 1.0 / (steps_n - 1)

    points_x = np.zeros((steps_n,), dtype=float)

    points_y = np.zeros((steps_n,), dtype=float)

    points_fxy = np.zeros((steps_n,), dtype=float)

    for i in numba.prange(steps_n):
        points_x[i] = dt * dx * i + x_0
        points_y[i] = dt * dy * i + y_0
        points_fxy[i] = bi_linear_interp_pt(points_x[i], points_y[i], points, width, height)

    return points_x, points_y, points_fxy


@numba.njit(fastmath=True, parallel=True)
def bi_linear_cut_along_curve(x_pts: np.ndarray, y_pts: np.ndarray, points: np.ndarray,
                              width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """
    Счечение интерполируемой би-линейным методом поверхности вдоль кривой, заданной в виде массива точек
    :param x_pts: координаты кривой по х
    :param y_pts: координаты кривой по y
    :param points: одномерный список узловых точек
    :param width: ширина области интеполяции
    :param height: высота области интеполяции
    :return:
    """
    cut_values = np.zeros((min(x_pts.size, y_pts.size),), dtype=float)
    for i in numba.prange(cut_values.size):
        cut_values[i] = bi_linear_interp_pt(x_pts[i], y_pts[i], points, width, height)
    return cut_values


@numba.njit(fastmath=True)
def _cubic_poly(x: float, y: float, m: np.ndarray) -> float:
    """
    Вспомогательная  функция для вычисления кубического полинома би кубической интерполяции0
    :param x: x - координата
    :param y: y - координата
    :param m: матрица коэффициентов
    :return:
    """
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    return (m[0]  + m[1] *  y + m[2]  * y2 + m[3]  * y3) + \
           (m[4]  + m[5] *  y + m[6]  * y2 + m[7]  * y3) * x + \
           (m[8]  + m[9] *  y + m[10] * y2 + m[11] * y3) * x2 + \
           (m[12] + m[13] * y + m[14] * y2 + m[15] * y3) * x3


@numba.njit(fastmath=True)
def __bi_qubic_interp_pt(x: float, y: float, points: np.ndarray, points_dx: np.ndarray,
                         points_dy: np.ndarray, points_dxy: np.ndarray,
                         width: float = 1.0, height: float = 1.0) -> float:
    """
    :param x: координата точки по х
    :param y: координата точки по y
    :param points: одномерный список узловых точек
    :param points_dx: производная по х в узловых точках
    :param points_dy: производная по y в узловых точках
    :param points_dxy: производная по хy в узловых точках
    :param width: ширина области интеполяции
    :param height: высота области интеполяции
    :return:
    """
    rows, cols = points.shape[0], points.shape[0]

    x = clamp(x, 0.0, width)

    y = clamp(y, 0.0, height)

    col_ = int((x / width) * (cols - 1))

    row_ = int((y / height) * (rows - 1))

    col_1 = min(col_ + 1, cols - 1)

    row_1 = min(row_ + 1, rows - 1)

    # q11 = nodes[row_, col_]

    # p00____p01
    # |       |
    # |       |
    # p10____p11

    dx_ = width / (cols - 1.0)
    dy_ = height / (rows - 1.0)

    tx = (x - dx_ * col_) / dx_
    ty = (y - dy_ * row_) / dy_
    pids = ((col_,  row_),   # p00
            (col_1, row_),   # p01
            (col_,  row_1),  # p10
            (col_1, row_1))  # p11

    b = np.zeros((16,), dtype=float)  # TODO CHECK IF np.zeros(...) MAY BE REPLACED BY SOMETHING

    c = np.zeros((16,), dtype=float)  # TODO CHECK IF np.zeros(...) MAY BE REPLACED BY SOMETHING

    for i in range(4):
        k, w = pids[i]
        b[i]      = points    [k, w]
        b[4 + i]  = points_dx [k, w]  # * dx_
        b[8 + i]  = points_dy [k, w]  # * dy_
        b[12 + i] = points_dxy[k, w]  # * dx_ * dy_

    for i in range(c.size):
        for j in range(b.size):
            c[i] += _bicubic_poly_coefficients[i * 16 + j] * b[j]

    return _cubic_poly(tx, ty, c)


@numba.njit(fastmath=True)
def bi_qubic_interp_pt(x: float, y: float, points: np.ndarray, width: float = 1.0, height: float = 1.0) -> float:
    """
    Бикубическая иетерполяция точки (x,y)
    :param x: x - координата точки
    :param y: y - координата точки
    :param points: одномерный список узловых точек
    :param width: ширина области интеполяции
    :param height: высота области интеполяции
    :return:
    """
    if points.ndim != 2:
        raise RuntimeError("bi_linear_interp_pt :: points array has to be 2 dimensional")

    rows, cols = points.shape[0], points.shape[0]

    x = clamp(x, 0.0, width)

    y = clamp(y, 0.0, height)

    col_ = int((x / width) * (cols - 1))

    row_ = int((y / height) * (rows - 1))

    col_1 = min(col_ + 1, cols - 1)

    row_1 = min(row_ + 1, rows - 1)

    # q11 = nodes[row_, col_]

    # p00____p01
    # |       |
    # |       |
    # p10____p11

    dx_ = width / (cols - 1.0)
    dy_ = height / (rows - 1.0)

    tx = (x - dx_ * col_) / dx_
    ty = (y - dy_ * row_) / dy_
    pids = ((col_,  row_),  # p00
            (col_1, row_),  # p01
            (col_,  row_1), # p10
            (col_1, row_1)) # p11

    b = np.zeros((16,), dtype=float)

    c = np.zeros((16,), dtype=float)

    for i in range(4):
        w, k = pids[i]
        b[i] = points[w, k]
        dx, dy, dxy = compute_derivatives_2_at_pt(points, w, k)
        b[4 + i] = dx
        b[8 + i] = dy
        b[12 + i] = dxy

    for i in range(c.size):
        for j in range(b.size):
            c[i] += _bicubic_poly_coefficients[i * 16 + j] * b[j]

    return _cubic_poly(tx, ty, c)


@numba.njit(fastmath=True, parallel=True)
def bi_qubic_interp(x: np.ndarray, y: np.ndarray,
                    points: np.ndarray, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """
    Бикубическая иетерполяция диапазона точек x, y
    :param x: x - координаты точек
    :param y: y - координаты точек
    :param points: одномерный список узловых точек
    :param width: ширина области интеполяции
    :param height: высота области интеполяции
    :return:
    """
    result = np.zeros((y.size, x.size), dtype=float)

    points_dx, points_dy, points_dxy = compute_derivatives_2(points)

    for i in numba.prange(result.size):
        res_col_ = i % x.size

        res_row_ = i // x.size
        """
        result[res_row_, res_col_] = bi_qubic_interp_pt(x[res_col_], y[res_row_], points, width, height)
        """
        result[res_row_, res_col_] = __bi_qubic_interp_pt(x[res_col_], y[res_row_], points, points_dx,
                                                          points_dy, points_dxy, width, height)
    return result


@numba.njit(fastmath=True, parallel=True)
def bi_qubic_cut(x_0: float, y_0: float, x_1: float, y_1: float, steps_n: int, points: np.ndarray,
                 width: float = 1.0, height: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Счечение интерполируемой би-кубическим методом поверхности вдоль прямой через две точки
    :param x_0: x - координата первой точки секущей
    :param y_0: y - координата первой точки секущей
    :param x_1: x - координата второй точки секущей
    :param y_1: y - координата второй точки секущей
    :param steps_n: количество точке вдоль секущей
    :param points: одномерный список узловых точек
    :param width: ширина области интеполяции
    :param height: высота области интеполяции
    :return:
    """
    dx = x_1 - x_0
    dy = y_1 - y_0
    rho = dx * dx + dy * dy
    if rho > 1e-12:
        rho = sqrt(rho)
        dx /= rho
        dy /= rho
    else:
        dx = 0.0
        dy = 0.0

    dt = 1.0 / (steps_n - 1)

    points_x = np.zeros((steps_n,), dtype=float)

    points_y = np.zeros((steps_n,), dtype=float)

    points_fxy = np.zeros((steps_n,), dtype=float)

    for i in numba.prange(steps_n):
        points_x[i] = dt * dx * i + x_0
        points_y[i] = dt * dy * i + y_0
        points_fxy[i] = bi_qubic_interp_pt(points_x[i], points_y[i], points, width, height)

    return points_x, points_y, points_fxy


@numba.njit(fastmath=True, parallel=True)
def bi_qubic_cut_along_curve(x_pts: np.ndarray, y_pts: np.ndarray, points: np.ndarray,
                             width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """
    Счечение интерполируемой би-кубическим методом поверхности вдоль кривой, заданной в виде массива точек
    :param x_pts: координаты кривой по х
    :param y_pts: координаты кривой по y
    :param points: одномерный список узловых точек
    :param width: ширина области интеполяции
    :param height: высота области интеполяции
    :return:
    """
    cut_values = np.zeros((min(x_pts.size, y_pts.size),), dtype=float)
    for i in numba.prange(cut_values.size):
        cut_values[i] = bi_qubic_interp_pt(x_pts[i], y_pts[i], points, width, height)
    return cut_values
