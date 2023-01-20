import matplotlib.pyplot as plt

from cgeo.trigonometry.trig_tables import SQUARE_INTERPOLATOR, LINEAR_INTERPOLATOR, A_TAN_RATIO, pi05, \
    INTERP_MODE_TABLES, SIN, pi2, COS, TAN, A_SIN, A_COS, A_TAN, pi
# from matplotlib import pyplot as plt
from typing import Union
import numpy as np
# import numba
import time


_accuracy = 1.0 - 1e-9


#######################################
#####  sin/cos lin interpolate   ######
#######################################
# @numba.njit(fastmath=True)
def _tab_func_lin(arg: float, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> float:
    t: float
    index: int
    t = (arg - x_0) * _accuracy / (x_1 - x_0)
    t -= int(t)
    if t < 0:
        t += 1.0
    t *= (tab.size - 1)
    index = int(t)
    t -= index
    return tab[index] + (tab[index + 1] - tab[index]) * t


# @numba.njit(fastmath=True)
def _tab_func_lin_np(arg: np.ndarray, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros_like(arg)
    t: float
    index: int
    for i in range(arg.size):
        t = (arg[i] - x_0) * _accuracy / (x_1 - x_0)
        t -= int(t)
        if t < 0:
            t += 1.0
        t *= (tab.size - 1)
        index = int(t)
        t -= index
        res[i] = tab[index] + (tab[index + 1] - tab[index]) * t
    return res


#########################################
#####  sin/cos qubic interpolate   ######
#########################################
# @numba.njit(fastmath=True)
def _tab_func_quad(arg: float, table: np.ndarray, x_0: float = 0.0, x_1: float = 1.0) -> float:
    t: float = (arg - x_0) * _accuracy / (x_1 - x_0)
    t -= int(t)
    if t < 0:
        t -= 1.0
    t *= table.shape[0]
    index: int = int(t)
    t -= index
    return table[index][0] * t * t + table[index][1] * t + table[index][2]


# @numba.njit(fastmath=True)
def _tab_func_quad_np(arg: np.ndarray, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros((arg.size,), dtype=float)
    dx = _accuracy / (x_1 - x_0)
    t: float
    index: int
    for i in range(arg.size):
        t = (arg[i] - x_0) * dx
        t -= int(t)
        if t < 0:
            t += 1.0
        t *= table.shape[0]
        index = int(t)
        t -= index
        res[i] = table[index][0] * t * t + table[index][1] * t + table[index][2]
    return res


def _tab_function(arg: Union[float, np.ndarray], tab: np.array, x_0: float = 0.0, x_1: float = 1.0,
                  mode: int = SQUARE_INTERPOLATOR) -> Union[float, np.ndarray]:
    if mode == LINEAR_INTERPOLATOR:
        if isinstance(arg, float):
            return _tab_func_lin(arg, tab, x_0, x_1)

        if isinstance(arg, np.ndarray):
            return _tab_func_lin_np(arg, tab, x_0, x_1)

    if mode == SQUARE_INTERPOLATOR:
        if isinstance(arg, float):
            return _tab_func_quad(arg, tab, x_0, x_1)

        if isinstance(arg, np.ndarray):
            return _tab_func_quad_np(arg, tab, x_0, x_1)

    raise ValueError("unsupported data type or interp method")


################################################
#####  arcsin/arccos linear interpolate   ######
################################################
# @numba.njit(fastmath=True)
def _tab_arc_sin_arc_cos_lin(arg: float, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> float:
    t: float
    index: int
    if x_0 > arg:
        return 0.0
    if x_1 < arg:
        return 0.0
    t = (arg - x_0) * _accuracy / (x_1 - x_0)
    t *= (tab.size - 1)
    index = int(t)
    t -= index
    return tab[index] + (tab[index + 1] - tab[index]) * t


# @numba.njit(fastmath=True)
def _tab_arc_sin_arc_cos_lin_np(arg: np.ndarray, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros_like(arg)
    t: float
    index: int
    for i in range(arg.size):
        if x_0 > arg[i]:
            res[i] = 0.0
            continue
        if x_1 < arg[i]:
            res[i] = 0.0
            continue
        t = (arg[i] - x_0) * _accuracy / (x_1 - x_0)
        t *= (tab.size - 1)
        index = int(t)
        t -= index
        res[i] = tab[index] + (tab[index + 1] - tab[index]) * t

    return res


###############################################
#####  arcsin/arccos qubic interpolate   ######
###############################################
# @numba.njit(fastmath=True)
def _tab_arc_sin_arc_cos_quad(arg: float, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> float:
    t: float
    index: int
    if x_0 > arg:
        return 0.0
    if x_1 < arg:
        return 0.0
    t = (arg - x_0) * _accuracy / (x_1 - x_0)
    t *= (table.shape[0] - 1)
    index = int(t)
    t -= index
    return table[index][0] * t * t + table[index][1] * t + table[index][2]


# @numba.njit(fastmath=True)
def _tab_arc_sin_arc_cos_quad_np(arg: np.ndarray, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros_like(arg)
    t: float
    index: int
    for i in range(arg.size):
        if x_0 > arg[i]:
            res[i] = 0.0
            continue
        if x_1 < arg[i]:
            res[i] = 0.0
            continue
        t = (arg[i] - x_0) * _accuracy / (x_1 - x_0)
        t *= (table.shape[0] - 1)
        index = int(t)
        t -= index
        res[i] = table[index][0] * t * t + table[index][1] * t + table[index][2]
    return res


def _tab_arc_sin_arc_cos(arg: Union[float, np.ndarray], tab: np.array, x_0: float = 0.0, x_1: float = 1.0,
                         mode: int = SQUARE_INTERPOLATOR) -> Union[float, np.ndarray]:
    if mode == LINEAR_INTERPOLATOR:
        if isinstance(arg, float):
            return _tab_arc_sin_arc_cos_lin(arg, tab, x_0, x_1)

        if isinstance(arg, np.ndarray):
            return _tab_arc_sin_arc_cos_lin_np(arg, tab, x_0, x_1)

    if mode == SQUARE_INTERPOLATOR:
        if isinstance(arg, float):
            return _tab_arc_sin_arc_cos_quad(arg, tab, x_0, x_1)

        if isinstance(arg, np.ndarray):
            return _tab_arc_sin_arc_cos_quad_np(arg, tab, x_0, x_1)

    raise ValueError("unsupported data type or interp method")


###################################
#####  tan lin interpolate   ######
###################################
# @numba.njit(fastmath=True)
def _tab_tan_lin(arg: float, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> float:
    if arg <= -1.5605:
        return -1.0 / (arg + pi05)
    if arg >= 1.5605:
        return -1.0 / (arg - pi05)
    t: float
    index: int
    t = (arg - x_0) * _accuracy / (x_1 - x_0)
    t -= int(t)
    if t < 0:
        t += 1.0
    t *= (tab.size - 1)
    index = int(t)
    t -= index
    return tab[index] + (tab[index + 1] - tab[index]) * t


# @numba.njit(fastmath=True, parallel=True)
def _tab_tan_lin_np(arg: np.ndarray, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros((arg.size,), dtype=float)
    dx = _accuracy / (x_1 - x_0)
    t: float
    index: int
    # for i in numba.prange(arg.size):
    for i in range(arg.size):
        if arg[i] <= -1.5605:
            res[i] = -1.0 / (arg[i] + pi05)
            continue
        if arg[i] >= 1.5605:
            res[i] = -1.0 / (arg[i] - pi05)
            continue
        t = (arg[i] - x_0) * dx
        t -= int(t)
        if t < 0:
            t += 1.0
        t *= (tab.size - 1)
        index = int(t)
        t -= index
        res[i] = tab[index] + (tab[index + 1] - tab[index]) * t
    return res


####################################
#####  tan quad interpolate   ######
####################################
# @numba.njit(fastmath=True)
def _tab_tan_quad(arg: float, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> float:
    if arg <= -1.5605:
        return -1.0 / (arg + pi05)
    if arg >= 1.5605:
        return -1.0 / (arg - pi05)
    t: float
    index: int
    t = (arg - x_0) * _accuracy / (x_1 - x_0)
    t -= int(t)
    if t < 0:
        t += 1.0
    t *= (table.shape[0] - 1)
    index = int(t)
    t -= index
    return table[index][0] * t * t + table[index][1] * t + table[index][2]


# @numba.njit(fastmath=True, parallel=True)
def _tab_tan_quad_np(arg: np.ndarray, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros((arg.size,), dtype=float)
    dx = _accuracy / (x_1 - x_0)
    t: float
    index: int
    # for i in numba.prange(arg.size):
    for i in range(arg.size):
        if arg[i] <= -1.5605:
            res[i] = -1.0 / (arg[i] + pi05)
            continue
        if arg[i] >= 1.5605:
            res[i] = -1.0 / (arg[i] - pi05)
            continue
        t = (arg[i] - x_0) * dx
        t -= int(t)
        if t < 0:
            t += 1.0
        t *= (table.shape[0] - 1)
        index = int(t)
        t -= index
        res[i] = table[index][0] * t * t + table[index][1] * t + table[index][2]
    return res


def _tab_function_tan(arg: Union[float, np.ndarray], tab: np.array, x_0: float = 0.0, x_1: float = 1.0,
                      mode: int = SQUARE_INTERPOLATOR) -> Union[float, np.ndarray]:
    if mode == LINEAR_INTERPOLATOR:
        if isinstance(arg, float):
            return _tab_tan_lin(arg, tab, x_0, x_1)

        if isinstance(arg, np.ndarray):
            return _tab_tan_lin_np(arg, tab, x_0, x_1)

    if mode == SQUARE_INTERPOLATOR:
        if isinstance(arg, float):
            return _tab_tan_quad(arg, tab, x_0, x_1)

        if isinstance(arg, np.ndarray):
            return _tab_tan_quad_np(arg, tab, x_0, x_1)

    raise ValueError("unsupported data type or interp method")


#######################################
#####  arc tan lin interpolate   ######
#######################################
# @numba.njit(fastmath=True)
def _tab_arc_tan_lin(arg: float, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> float:
    if arg <= -A_TAN_RATIO:
        return -1.0 / arg - pi05
    if arg >= A_TAN_RATIO:
        return -1.0 / arg + pi05
    t: float
    index: int
    t = (arg - x_0) * _accuracy / (x_1 - x_0)
    t -= int(t)
    if t < 0:
        t += 1.0
    t *= (tab.size - 1)
    index = int(t)
    t -= index
    return tab[index] + (tab[index + 1] - tab[index]) * t


# @numba.njit(fastmath=True, parallel=True)
def _tab_arc_tan_lin_np(arg: np.ndarray, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros((arg.size,), dtype=float)
    dx = _accuracy / (x_1 - x_0)
    t: float
    index: int
    # for i in numba.prange(arg.size):
    for i in range(arg.size):
        if arg[i] <= -A_TAN_RATIO:
            res[i] = -1.0 / arg[i] - pi05
            continue
        if arg[i] >= A_TAN_RATIO:
            res[i] = -1.0 / arg[i] + pi05
            continue
        t = (arg[i] - x_0) * dx
        t -= int(t)
        if t < 0:
            t += 1.0
        t *= (tab.size - 1)
        index = int(t)
        t -= index
        res[i] = tab[index] + (tab[index + 1] - tab[index]) * t
    return res


########################################
#####  arc tan quad interpolate   ######
########################################
# @numba.njit(fastmath=True)
def _tab_arc_tan_quad(arg: float, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> float:
    if arg <= -A_TAN_RATIO:
        return -1.0 / arg - pi05
    if arg >= A_TAN_RATIO:
        return -1.0 / arg + pi05
    t: float
    index: int
    t = (arg - x_0) * _accuracy / (x_1 - x_0)
    t -= int(t)
    if t < 0:
        t += 1.0
    t *= (table.shape[0] - 1)
    index = int(t)
    t -= index
    return table[index][0] * t * t + table[index][1] * t + table[index][2]


# @numba.njit(fastmath=True, parallel=True)
def _tab_arc_tan_quad_np(arg: np.ndarray, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros((arg.size,), dtype=float)
    dx = _accuracy / (x_1 - x_0)
    t: float
    index: int
    # for i in numba.prange(arg.size):
    for i in range(arg.size):
        if arg[i] <= -A_TAN_RATIO:
            res[i] = -1.0 / arg[i] - pi05
            continue
        if arg[i] >= A_TAN_RATIO:
            res[i] = -1.0 / arg[i] + pi05
            continue
        t = (arg[i] - x_0) * dx
        t -= int(t)
        if t < 0:
            t += 1.0
        t *= (table.shape[0] - 1)
        index = int(t)
        t -= index
        res[i] = table[index][0] * t * t + table[index][1] * t + table[index][2]
    return res


def _tab_function_arc_tan(arg: Union[float, np.ndarray], tab: np.array, x_0: float = 0.0, x_1: float = 1.0,
                          mode: int = SQUARE_INTERPOLATOR) -> Union[float, np.ndarray]:
    if mode == LINEAR_INTERPOLATOR:
        if isinstance(arg, float):
            return _tab_arc_tan_lin(arg, tab, x_0, x_1)

        if isinstance(arg, np.ndarray):
            return _tab_arc_tan_lin_np(arg, tab, x_0, x_1)

    if mode == SQUARE_INTERPOLATOR:
        if isinstance(arg, float):
            return _tab_arc_tan_quad(arg, tab, x_0, x_1)

        if isinstance(arg, np.ndarray):
            return _tab_arc_tan_quad_np(arg, tab, x_0, x_1)

    raise ValueError("unsupported data type or interp method")


def sin(arg: Union[float, np.ndarray], mode: int = 0) -> Union[float, np.ndarray]:
    if mode not in INTERP_MODE_TABLES:
        raise RuntimeError("Sin :: Incorrect interpolation mode")
    return _tab_function(arg, INTERP_MODE_TABLES[mode][SIN], 0, pi2, mode)


def cos(arg: Union[float, np.ndarray], mode: int = 0) -> Union[float, np.ndarray]:
    if mode not in INTERP_MODE_TABLES:
        raise RuntimeError("Cos :: Incorrect interpolation mode")
    return _tab_function(arg, INTERP_MODE_TABLES[mode][COS], 0, pi2, mode)


def tan(arg: Union[float, np.ndarray], mode: int = 0) -> Union[float, np.ndarray]:
    if mode not in INTERP_MODE_TABLES:
        raise RuntimeError("Tan :: Incorrect interpolation mode")
    return _tab_function_tan(arg, INTERP_MODE_TABLES[mode][TAN], -pi05, pi05, mode)


def a_sin(arg: Union[float, np.ndarray], mode: int = 0) -> Union[float, np.ndarray]:
    if mode not in INTERP_MODE_TABLES:
        raise RuntimeError("A_Sin :: Incorrect interpolation mode")
    return _tab_arc_sin_arc_cos(arg, INTERP_MODE_TABLES[mode][A_SIN], -1.0, 1.0, mode)


def a_cos(arg: Union[float, np.ndarray], mode: int = 0) -> Union[float, np.ndarray]:
    if mode not in INTERP_MODE_TABLES:
        raise RuntimeError("A_Cos :: Incorrect interpolation mode")
    return _tab_arc_sin_arc_cos(arg, INTERP_MODE_TABLES[mode][A_COS], -1.0, 1.0, mode)


def a_tan(arg: Union[float, np.ndarray], mode: int = 0) -> Union[float, np.ndarray]:
    if mode not in INTERP_MODE_TABLES:
        raise RuntimeError("A_Tan :: Incorrect interpolation mode")
    return _tab_function_arc_tan(arg, INTERP_MODE_TABLES[mode][A_TAN], -A_TAN_RATIO, A_TAN_RATIO, mode)


if __name__ == "__main__":
    x = np.linspace(-200.0, 200.0, 100000)
    asin = sin(x, 1)
    acos = cos(x, 1)

    asin_np = np.sin(x)
    acos_np = np.cos(x)

    plt.plot(x,asin_np - asin, 'r')
    plt.plot(x,acos_np - acos, 'g')

    # plt.plot(x,asin_np, ':r')
    # plt.plot(x,acos_np, ':g')

    plt.show()