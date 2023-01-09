from typing import Callable, Tuple
import numpy as np
import numba

LINEAR_INTERPOLATOR = 0
SQUARE_INTERPOLATOR = 1
############################
#####  RATIO VALUES   ######
############################

A_TAN_RATIO = 250.0
A_TAN_RATIO_VAL = np.arctan(A_TAN_RATIO)
pi: float = np.pi
pi2: float = np.pi * 2.0
pi05: float = np.pi * 0.5

######################################
#####  lin interpolate tables   ######
######################################

tab_values_lin: int = 1024
sin_values_lin: np.ndarray = \
    np.array([np.sin(np.pi * 2.0 / (tab_values_lin - 1) * i) for i in range(tab_values_lin)])
cos_values_lin: np.ndarray = \
    np.array([np.cos(np.pi * 2.0 / (tab_values_lin - 1) * i) for i in range(tab_values_lin)])
tan_values_lin: np.ndarray = np.array(
    [np.tan(-np.pi * 0.4999999 + 0.4999999 * pi2 / (tab_values_lin - 1) * i) for i in range(tab_values_lin)])
a_sin_values_lin: np.ndarray = np.array([np.arcsin(val) for val in np.linspace(-1.0, 1.0, tab_values_lin)])
a_cos_values_lin: np.ndarray = np.array([np.arccos(val) for val in np.linspace(-1.0, 1.0, tab_values_lin)])
a_tan_values_lin: np.ndarray = \
   np.array([np.arctan(val) for val in np.linspace(-A_TAN_RATIO, A_TAN_RATIO, tab_values_lin)])


#######################################
#####  quad interpolate tables   ######
#######################################
@numba.njit(fastmath=True)
def _sect_quad_coefficients(y1: float, y2: float, y3: float) -> Tuple[float, float, float]:
    """
    [[ 2. -4.  2.]
     [-3.  4. -1.]
     [ 1.  0.  0.]]
    :param y1:
    :param y2:
    :param y3:
    :return:
    """
    return 2.0 * y1 - 4.0 * y2 + 2.0 * y3, -3.0 * y1 + 4.0 * y2 - y3, y1


#######################################
#####  quad interpolate tables   ######
#######################################
def _func_quad_coefficients(func: Callable[[float], float], x_0: float = 0.0, x_1: float = 1.0, n_points: int = 256) ->\
        np.ndarray:
    if x_1 < x_0:
        x_1, x_0 = x_0, x_1

    dx = (x_1 - x_0) / (n_points - 1)
    return np.array([_sect_quad_coefficients(func(x_0 + i * dx),
                                             func(x_0 + (i + 0.5) * dx),
                                             func(x_0 + (i + 1.0) * dx)) for i in range(n_points - 1)])


tab_values_sq: int = 512
sin_values_sq: np.ndarray = _func_quad_coefficients(lambda xi: np.sin(xi), 0.0, pi2, tab_values_sq)
cos_values_sq: np.ndarray = _func_quad_coefficients(lambda xi: np.cos(xi), 0.0, pi2, tab_values_sq)
tan_values_sq: np.ndarray = _func_quad_coefficients(lambda xi: np.tan(xi), -pi05, pi05, tab_values_sq)

a_sin_values_sq: np.ndarray = _func_quad_coefficients(lambda xi: np.arcsin(xi), -1.0, 1.0, tab_values_sq)
a_cos_values_sq: np.ndarray = _func_quad_coefficients(lambda xi: np.arccos(xi), -1.0, 1.0, tab_values_sq)
a_tan_values_sq: np.ndarray = \
    _func_quad_coefficients(lambda xi: np.arctan(xi), -A_TAN_RATIO, A_TAN_RATIO, tab_values_sq)


SIN = 0
COS = 1
TAN = 2


A_SIN = 3
A_COS = 4
A_TAN = 5


INTERP_TABLES_LIN = \
    {SIN: sin_values_lin,
     COS: cos_values_lin,
     TAN: tan_values_lin,
     A_SIN: a_sin_values_lin,
     A_COS: a_cos_values_lin,
     A_TAN: a_tan_values_lin}


INTERP_TABLES_SQ = \
    {SIN: sin_values_sq,
     COS: cos_values_sq,
     TAN: tan_values_sq,
     A_SIN: a_sin_values_sq,
     A_COS: a_cos_values_sq,
     A_TAN: a_tan_values_sq}


INTERP_MODE_TABLES = \
    {0: INTERP_TABLES_LIN,
     1: INTERP_TABLES_SQ}