from cgeo.trigonometry.trig_tables import SQUARE_INTERPOLATOR, LINEAR_INTERPOLATOR, A_TAN_RATIO, pi05, \
    INTERP_MODE_TABLES, SIN, pi2, COS, TAN, A_SIN, A_COS, A_TAN, pi
from matplotlib import pyplot as plt
from typing import Union
import numpy as np
import numba
import time


_accuracy = 1.0 - 1e-12


#######################################
#####  sin/cos lin interpolate   ######
#######################################
@numba.njit(fastmath=True)
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


@numba.njit(fastmath=True)
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
@numba.njit(fastmath=True)
def _tab_func_quad(arg: float, table: np.ndarray, x_0: float = 0.0, x_1: float = 1.0) -> float:
    t: float = (arg - x_0) * _accuracy / (x_1 - x_0)
    t -= int(t)
    if t < 0:
        t -= 1.0
    t *= table.shape[0]
    index: int = int(t)
    t -= index
    return table[index][0] * t * t + table[index][1] * t + table[index][2]


@numba.njit(fastmath=True)
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
@numba.njit(fastmath=True)
def _tab_arc_sin_arc_cos_lin(arg: float, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> float:
    t: float
    index: int
    if x_0 > arg:
        # print("Value {:.4f} out of range[{:.4f}, {:.4f}]".format(arg, x_0, x_1))
        return 0.0
    if x_1 < arg:
        # print("Value {:.4f} out of range[{:.4f}, {:.4f}]".format(arg, x_0, x_1))
        return 0.0
    t = (arg - x_0) * _accuracy / (x_1 - x_0)
    t -= int(t)
    if t < 0:
        t += 1.0
    t *= (tab.size - 1)
    index = int(t)
    t -= index
    return tab[index] + (tab[index + 1] - tab[index]) * t


@numba.njit(fastmath=True)
def _tab_arc_sin_arc_cos_lin_np(arg: np.ndarray, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros_like(arg)
    t: float
    index: int
    for i in range(arg.size):
        if x_0 > arg[i]:
            # print("Value {:.4f} out of range[{:.4f}, {:.4f}]".format(arg[i], x_0, x_1))
            res[i] = 0.0
            continue
        if x_1 < arg[i]:
            # print("Value {:.4f} out of range[{:.4f}, {:.4f}]".format(arg[i], x_0, x_1))
            res[i] = 0.0
            continue
        t = (arg[i] - x_0) * _accuracy / (x_1 - x_0)
        t -= int(t)
        if t < 0:
            t += 1.0
        t *= (tab.size - 1)
        index = int(t)
        t -= index
        res[i] = tab[index] + (tab[index + 1] - tab[index]) * t

    return res


###############################################
#####  arcsin/arccos qubic interpolate   ######
###############################################
@numba.njit(fastmath=True)
def _tab_arc_sin_arc_cos_quad(arg: float, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> float:
    t: float
    index: int
    if x_0 > arg:
        # print("Value {:.4f} out of range[{:.4f}, {:.4f}]".format(arg, x_0, x_1))
        return 0.0
    if x_1 < arg:
        # print("Value {:.4f} out of range[{:.4f}, {:.4f}]".format(arg, x_0, x_1))
        return 0.0
    t = (arg - x_0) * _accuracy / (x_1 - x_0)
    t -= int(t)
    if t < 0:
        t += 1.0
    t *= (table.shape[0] - 1)
    index = int(t)
    t -= index
    return table[index][0] * t * t + table[index][1] * t + table[index][2]


@numba.njit(fastmath=True)
def _tab_arc_sin_arc_cos_quad_np(arg: np.ndarray, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros_like(arg)
    t: float
    index: int
    for i in range(arg.size):
        if x_0 > arg[i]:
            # print("Value {:.4f} out of range[{:.4f}, {:.4f}]".format(arg[i], x_0, x_1))
            res[i] = 0.0
            continue
        if x_1 < arg[i]:
            #  print("Value {:.4f} out of range[{:.4f}, {:.4f}]".format(arg[i], x_0, x_1))
            res[i] = 0.0
            continue
        t = (arg[i] - x_0) * _accuracy / (x_1 - x_0)
        t -= int(t)
        if t < 0:
            t += 1.0
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
@numba.njit(fastmath=True)
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


@numba.njit(fastmath=True, parallel=True)
def _tab_tan_lin_np(arg: np.ndarray, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros((arg.size,), dtype=float)
    dx = _accuracy / (x_1 - x_0)
    t: float
    index: int
    for i in numba.prange(arg.size):
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
@numba.njit(fastmath=True)
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


@numba.njit(fastmath=True, parallel=True)
def _tab_tan_quad_np(arg: np.ndarray, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros((arg.size,), dtype=float)
    dx = _accuracy / (x_1 - x_0)
    t: float
    index: int
    for i in numba.prange(arg.size):
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
@numba.njit(fastmath=True)
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


@numba.njit(fastmath=True, parallel=True)
def _tab_arc_tan_lin_np(arg: np.ndarray, tab: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros((arg.size,), dtype=float)
    dx = _accuracy / (x_1 - x_0)
    t: float
    index: int
    for i in numba.prange(arg.size):
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
@numba.njit(fastmath=True)
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


@numba.njit(fastmath=True, parallel=True)
def _tab_arc_tan_quad_np(arg: np.ndarray, table: np.array, x_0: float = 0.0, x_1: float = 1.0) -> np.ndarray:
    res = np.zeros((arg.size,), dtype=float)
    dx = _accuracy / (x_1 - x_0)
    t: float
    index: int
    for i in numba.prange(arg.size):
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


def sin_cos_test():
    x = np.linspace(-1.5 * pi, pi2 * 3.0, 2048)
    y1 = sin(x, 0)
    y2 = sin(x * 2)
    y3 = np.sin(x * 4)

    y_1 = cos(x, 0)
    y_2 = cos(x * 2)
    y_3 = np.cos(x * 4)
    """
    plt.plot(x, y1 - y3, 'r')
    plt.plot(x, y2 - y3, 'g')

    plt.plot(x, y_1 - y_3, ':r')
    plt.plot(x, y_2 - y_3, ':g')
    """
    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'g')

    # plt.plot(x, y_1, ':r')
    # plt.plot(x, y_2, ':g')
    plt.grid(True)
    plt.show()


def a_sin_cos_test():
    x = np.linspace(-1.2, 1.2, 1024)
    y1 = a_sin(x, 0)
    y2 = a_sin(x)
    y3 = np.arcsin(x)

    y1_ = a_cos(x, 0)
    y2_ = a_cos(x)
    y3_ = np.arccos(x)

    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'g')
    plt.plot(x, y3, 'b')

    plt.plot(x, y1_, ':r')
    plt.plot(x, y2_, ':g')
    plt.plot(x, y3_, ':b')

    plt.grid(True)
    plt.show()


def a_tan_test():
    x_ = np.linspace(-300.5, 300.5, 10000)
    x = np.linspace(-pi05 * 0.995, pi05 * 0.995, 10000)
    y1 = tan(x)
    y2 = tan(x, 1)
    y3 = np.tan(x)
    t_linear = 0.0
    t_qubic = 0.0
    t_np = 0.0
    t = 0.0
    for i in range(1000):
        t = time.perf_counter()
        y = sin(x)
        t = time.perf_counter() - t
        t_linear += t
        t = time.perf_counter()
        y = sin(x, 1)
        t = time.perf_counter() - t
        t_qubic += t
        t = time.perf_counter()
        y = np.sin(x)
        t = time.perf_counter() - t
        t_np += t
    print(f"t_linear: {t_linear / 1000}\n"
          f"t_qubic : {t_qubic / 1000}\n"
          f"t_np    : {t_np / 1000}")
    y1_ = a_tan(x_)
    y2_ = a_tan(x_, 1)
    y3_ = np.arctan(x_)

    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'g')
    plt.plot(x, y3, 'b')

    # plt.plot(x_, y1_, ':r')
    # plt.plot(x_, y2_, ':g')
    # plt.plot(x_, y3_, ':b')

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # sin_cos_test()
    a_sin_cos_test()
    # a_tan_test()
