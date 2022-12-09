from typing import Union
import numpy as np


_pi = np.pi
_2pi = np.pi * 2.0
_05pi = np.pi * 0.5

tab_values_n = 1024
sin_values = np.array([np.sin(np.pi * 2.0 / (tab_values_n - 1) * i) for i in range(tab_values_n)])
cos_values = np.array([np.cos(np.pi * 2.0 / (tab_values_n - 1) * i) for i in range(tab_values_n)])
tan_values = np.array([np.tan(np.pi / (tab_values_n - 1) * i) for i in range(tab_values_n)])
a_sin_values = np.array([np.asin(sin) for sin in sin_values])
a_cos_values = np.array([np.acos(cos) for cos in cos_values])
a_tan_values = np.array([np.atan(tan) for tan in tan_values])


def frac(x):
	return x - int(x)


def tab_function(arg: Union[float, np.ndarray], x_0: float, x_1: float, tab: np.array):
	if isinstance(arg, float):
		arg = frac((arg - x_0)/(x_1 - x_0))
		x_ = (sin_values.size - 1) * arg
		t = frac(x_)
		index = int(x_)
		return tab[index] + (tab[index + 1] - tab[index]) * t

	if isinstance(arg, np.ndarray):
		res = np.zeros_like(arg)
		for i in range(arg.size):
			arg = frac((arg[i] - x_0) / (x_1 - x_0))
			x_ = (sin_values.size - 1) * arg[i]
			t = frac(x_)
			index = int(x_)
			res[i] = tab[index] + (tab[index + 1] - tab[index]) * t
		return res
	raise ValueError("unsupported data type")


def t_sin(arg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return tab_function(arg, 0, _2pi, sin_values)


def t_cos(arg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return tab_function(arg, 0, _2pi, cos_values)


def t_a_sin(arg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return tab_function(arg, -1.0, 1.0, a_sin_values)


def t_a_cos(arg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return tab_function(arg, -1.0, 1.0, a_cos_values)


def t_tan(arg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return tab_function(arg, -_05pi, _05pi, tan_values)


if __name__ == "__main__":
	pass