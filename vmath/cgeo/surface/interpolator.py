# from matplotlib import pyplot as plt

from cgeo.surface.interpolators import bi_linear_interp, bi_linear_interp_pt, bi_qubic_interp_pt, bi_qubic_interp, \
    bi_linear_cut, bi_qubic_cut, bi_linear_cut_along_curve, bi_qubic_cut_along_curve, bi_linear_interp_derivatives, \
    bi_linear_interp_derivatives2, bi_linear_interp_derivatives2_pt, bi_linear_interp_derivatives_pt, \
    bi_cubic_interp_derivatives, bi_cubic_interp_derivatives2, bi_cubic_interp_derivatives2_pt, \
    bi_cubic_interp_derivatives_pt
from cgeo.vectors import Vec2, Vec3
from typing import Tuple
from cgeo import gutils, LoopTimer
import numpy as np
import copy
import json


class Interpolator:
    @staticmethod
    def __arithmetic_operation_inner(interpolator_a, interpolator_b, operation) -> None:
        if isinstance(interpolator_b, float) or isinstance(interpolator_b, int):
            for i in range(len(interpolator_a.control_points)):
                interpolator_a.__control_points[i] = \
                    operation(interpolator_a.__z_0 + interpolator_a.__control_points[i], interpolator_b)
            interpolator_a.__z_0 = 0.0
            return

        if not isinstance(interpolator_b, Interpolator):
            raise Exception("Interpolator::arithmetic operation error::wrong operand type")

        a_min = Vec2(interpolator_a.x_0, interpolator_a.y_0)
        a_max = Vec2(interpolator_a.x_0 + interpolator_a.width, interpolator_a.y_0 + interpolator_a.height)
        b_min = Vec2(interpolator_b.x_0, interpolator_b.y_0)
        b_max = Vec2(interpolator_b.x_0 + interpolator_b.width, interpolator_b.y_0 + interpolator_b.height)
        flag, pt_1, pt_2 = gutils.rect_intersection(a_min, a_max, b_min, b_max)
        if not flag:
            return
        _rows, _cols = max(interpolator_a.rows, interpolator_b.rows), max(interpolator_a.colons, interpolator_b.colons)

        _dx = (pt_2[0] - pt_1[0]) / (_cols - 1)

        _dy = (pt_2[1] - pt_1[1]) / (_cols - 1)

        control_points = []

        for i in range(_rows * _cols):
            row, col = divmod(i, _cols)
            control_points.append(operation(interpolator_a.interpolate_point(col * _dx, row * _dy),
                                            interpolator_b.interpolate_point(col * _dx, row * _dy)))

        interpolator_a.__control_points = np.reshape(np.array(control_points), (_rows, _cols))
        interpolator_a.__points_file = "none"
        interpolator_a.__origin = Vec3(pt_1[0], pt_1[1], 0.0)
        interpolator_a.__width = pt_2[0] - pt_1[0]
        interpolator_a.__height = pt_2[1] - pt_1[1]

    def __interpolate_point_bi_linear(self, x: float, y: float) -> float:
        return bi_linear_interp_pt(x - self.x_0, y - self.y_0, self.control_points,
                                   self.width, self.height) + self.z_0

    def __interpolate_point_bi_linear_derivative(self, x: float, y: float,
                                                 dx: float = 0.001, dy: float = 0.001) -> Tuple[float, float]:
        x, y = bi_linear_interp_derivatives_pt(x - self.x_0, y - self.y_0, self.control_points,
                                               self.width, self.height, dx, dy)
        return x + self.z_0, y + self.z_0

    def __interpolate_point_bi_linear_derivative2(self, x: float, y: float,
                                                  dx: float = 0.001, dy: float = 0.001) -> Tuple[float, float, float]:
        x, y, z = bi_linear_interp_derivatives2_pt(x - self.x_0, y - self.y_0, self.control_points,
                                                   self.width, self.height, dx, dy)
        return x + self.z_0, y + self.z_0,  z + self.z_0

    def __interpolate_bi_linear(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return bi_linear_interp(x - self.x_0, y - self.y_0, self.control_points,
                                self.width, self.height) + self.z_0

    def __interpolate_bi_linear_derivative(self, x: np.ndarray, y: np.ndarray,
                                           dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
        return bi_linear_interp_derivatives(x - self.x_0, y - self.y_0, self.control_points,
                                            self.width, self.height, dx, dy) + self.z_0

    def __interpolate_bi_linear_derivative2(self, x: np.ndarray, y: np.ndarray,
                                            dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
        return bi_linear_interp_derivatives2(x - self.x_0, y - self.y_0, self.control_points,
                                             self.width, self.height, dx, dy) + self.z_0

    def __interpolate_point_bi_cubic(self, x: float, y: float) -> float:
        return bi_qubic_interp_pt(x - self.x_0, y - self.y_0, self.control_points,
                                  self.width, self.height) + self.z_0

    def __interpolate_point_bi_cubic_derivative(self, x: float, y: float,
                                                dx: float = 0.001, dy: float = 0.001) -> Tuple[float, float]:
        x, y = bi_cubic_interp_derivatives_pt(x - self.x_0, y - self.y_0, self.control_points,
                                              self.width, self.height, dx, dy)
        return x + self.z_0, y + self.z_0

    def __interpolate_point_bi_cubic_derivative2(self, x: float, y: float,
                                                 dx: float = 0.001, dy: float = 0.001) -> Tuple[float, float, float]:
        x, y, z = bi_cubic_interp_derivatives2_pt(x - self.x_0, y - self.y_0, self.control_points,
                                                  self.width, self.height, dx, dy)
        return x + self.z_0, y + self.z_0,  z + self.z_0

    def __interpolate_bi_cubic(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return bi_qubic_interp(x - self.x_0, y - self.y_0, self.control_points,
                               self.width, self.height) + self.z_0

    def __interpolate_bi_cubic_derivative(self, x: np.ndarray, y: np.ndarray,
                                          dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
        return bi_cubic_interp_derivatives(x - self.x_0, y - self.y_0, self.control_points,
                                           self.width, self.height, dx, dy) + self.z_0

    def __interpolate_bi_cubic_derivative2(self, x: np.ndarray, y: np.ndarray,
                                           dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
        return bi_cubic_interp_derivatives2(x - self.x_0, y - self.y_0, self.control_points,
                                            self.width, self.height, dx, dy) + self.z_0

    def __bi_linear_cut(self, x_0: float, y_0: float, x_1: float, y_1: float, n_steps: int = 128) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = bi_linear_cut(x_0 - self.x_0, y_0 - self.y_0,
                                x_1 - self.x_0, y_1 - self.y_0,
                                n_steps, self.__control_points,
                                self.width, self.height)
        z += self.z_0
        return x, y, z

    def __bi_cubic_cut(self, x_0: float, y_0: float, x_1: float, y_1: float, n_steps: int = 128) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = bi_qubic_cut(x_0 - self.x_0, y_0 - self.y_0,
                               x_1 - self.x_0, y_1 - self.y_0,
                               n_steps, self.control_points,
                               self.width, self.height)
        z += self.z_0
        return x, y, z

    def __bi_linear_cut_along_curve(self, x_0: np.ndarray, y_0: np.ndarray) -> np.ndarray:
        return bi_linear_cut_along_curve(x_0 - self.x_0, y_0 - self.y_0, self.control_points,
                                         self.width, self.height) + self.z_0

    def __bi_cubic_cut_along_curve(self, x_0: np.ndarray, y_0: np.ndarray) -> np.ndarray:
        return bi_qubic_cut_along_curve(x_0 - self.x_0, y_0 - self.y_0, self.control_points,
                                        self.width, self.height) + self.z_0

    __slots__ = "__control_points", "__points_file", "__width", "__height", "__origin", "__mode"

    def __init__(self, points: np.ndarray = None):
        self.__control_points: np.ndarray
        if points is not None:
            if points.ndim != 2:
                raise RuntimeError("Interpolator :: interpolator points has to be 2-dimensional array")
            self.__control_points: np.ndarray = points
        else:
            self.__control_points: np.ndarray = np.array([[1, 3, 1, 3],
                                                          [3, 1, 3, 1],
                                                          [1, 3, 1, 3]], dtype=float)

        self.__points_file: str = "no file"
        self.__width: float = 1.0
        self.__height: float = 1.0
        self.__origin: Vec3 = Vec3()
        self.__mode = 0
        # mode == 0 - bi-linear
        # mode == 1 - bi-quadratic bezier
        # mode == 2 - bi-cubic bezier

    def __copy__(self):
        _copy = Interpolator()
        _copy.__control_points = self.__control_points.copy()
        _copy.__points_file = self.__points_file
        _copy.__width = self.__width
        _copy.__height = self.__height
        _copy.__origin = Vec3(self.__origin)
        _copy.__mode = self.__mode
        return _copy

    def __iadd__(self, other):
        Interpolator.__arithmetic_operation_inner(self, other, lambda a, b: a + b)
        return self

    def __isub__(self, other):
        Interpolator.__arithmetic_operation_inner(self, other, lambda a, b: a - b)
        return self

    def __imul__(self, other):
        Interpolator.__arithmetic_operation_inner(self, other, lambda a, b: a * b)
        return self

    def __itruediv__(self, other):
        Interpolator.__arithmetic_operation_inner(self, other, lambda a, b: a / b)
        return self

    def __add__(self, other):
        self_copy = copy.copy(self)
        self_copy += other
        return self_copy

    __radd__ = __add__

    def __sub__(self, other):
        self_copy = copy.copy(self)
        self_copy -= other
        return self_copy

    def __rsub__(self, other):
        other_copy = copy.copy(other)
        other_copy -= self
        return other_copy

    def __mul__(self, other):
        self_copy = copy.copy(self)
        self_copy *= other
        return self_copy

    __rmul__ = __mul__

    def __truediv__(self, other):
        self_copy = copy.copy(self)
        self_copy /= other
        return self_copy

    def __rtruediv__(self, other):
        other_copy = copy.copy(other)
        other_copy /= self
        return other_copy

    def __str__(self):
        nl = ", "

        def points_to_srt():
            return ',\n'.join('\t\t' + ','.join(f'{self.control_points[j, i]:.10}'
                                                for i in range(self.colons)) for j in range(self.rows))
        return f'{{\n' \
               f'\t\"colons\"         : {self.colons},\n' \
               f'\t\"rows\"           : {self.rows},\n' \
               f'\t\"width\"          : {self.width},\n' \
               f'\t\"height\"         : {self.height},\n' \
               f'\t\"x_0\"            : {self.x_0},\n' \
               f'\t\"y_0\"            : {self.y_0},\n' \
               f'\t\"z_0\"            : {self.z_0},\n' \
               f'\t\"points_source\"  : \"{self.__points_file}\",\n' \
               f'\t\"control_points\" : \n\t[\n{points_to_srt()}\n\t]\n' \
               f'}}'

    def load(self, file_path: str) -> bool:
        flag = False
        with open(file_path, 'rt') as input_file:
            raw_json = json.loads(input_file.read())
            rows: int
            cols: int
            points: np.ndarray
            orig: Vec3 = Vec3()
            for key, value in raw_json.items():
                if key == "colons":
                    cols = value
                    continue
                if key == "rows":
                    rows = value
                    continue
                if key == "width":
                    self.__width = value
                    continue
                if key == "height":
                    self.__height = value
                    continue
                if key == "x_0":
                    orig.x = value
                    continue
                if key == "y_0":
                    orig.y = value
                    continue
                if key == "z_0":
                    orig.z = value
                    continue
                if key == "points_source":
                    self.__points_file = value
                    continue
                if key == "control_points":
                    points = np.array(value)
                    continue
                if points.size != rows * cols:
                    rows = points.size // cols
                    points = points[0:rows * cols]
            self.__control_points = np.reshape(points, (rows, cols))
            self.__origin = orig
            flag = True

        if flag:
            self.__points_file = file_path

        return flag

    def save(self, file_path: str) -> None:
        with open(file_path, 'wt') as output_file:
            print(self, file=output_file)

    def interpolate_point(self, x: float, y: float) -> float:
        """
        Значение интерполяции точки с координатами х, у.\n
        :param x: х - координата
        :param y: y - координата
        :return:
        """
        if self.control_points is None:
            print("Interpolator ::  control_points is None")
            return 0.0

        if self.bi_linear:
            return self.__interpolate_point_bi_linear(x, y)

        if self.bi_cubic:
            return self.__interpolate_point_bi_cubic(x, y)

        return 0.0

    def interpolate_point_derivative(self, x: float, y: float,
                                     dx: float = 0.001, dy: float = 0.001) -> Tuple[float, float]:
        """
        Значение интерполяции точки с координатами х, у.\n
        :param x: х - координата
        :param y: y - координата
        :param dx: dх
        :param dy: dy
        :return:
        """
        if self.control_points is None:
            print("Interpolator ::  control_points is None")
            return 0.0, 0.0

        if self.bi_linear:
            return self.__interpolate_point_bi_linear_derivative(x, y, dx, dy)

        if self.bi_cubic:
            return self.__interpolate_point_bi_cubic_derivative(x, y, dx, dy)

        return 0.0, 0.0

    def interpolate_point_derivative2(self, x: float, y: float,
                                      dx: float = 0.001, dy: float = 0.001) -> Tuple[float, float, float]:
        """
        Значение интерполяции точки с координатами х, у.\n
        :param x: х - координата
        :param y: y - координата
        :param dx: dх
        :param dy: dy
        :return:
        """
        if self.control_points is None:
            print("Interpolator ::  control_points is None")
            return 0.0, 0.0, 0.0

        if self.bi_linear:
            return self.__interpolate_point_bi_linear_derivative2(x, y, dx, dy)

        if self.bi_cubic:
            return self.__interpolate_point_bi_cubic_derivative2(x, y, dx, dy)

        return 0.0, 0.0, 0.0

    def interpolate(self, x_: np.ndarray, y_: np.ndarray) -> np.ndarray:
        """
        Значение интерполяции массивов точкек с координатами х, у.\n
        :param x_: х - массив координат
        :param y_: y - массив координат
        :return:
        """
        if self.control_points is None:
            print("Interpolator ::  control_points is None")
            return np.zeros((x_.size, y_.size), dtype=float)

        if self.bi_linear:
            return self.__interpolate_bi_linear(x_, y_)

        if self.bi_cubic:
            return self.__interpolate_bi_cubic(x_, y_)

        return np.zeros((x_.size, y_.size), dtype=float)

    def interpolate_derivative(self, x_: np.ndarray, y_: np.ndarray,
                               dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
        """
        Значение интерполяции массивов точкек с координатами х, у.\n
        :param x_: х - массив координат
        :param y_: y - массив координат
        :param dx: dх
        :param dy: dy
        :return:
        """
        if self.control_points is None:
            print("Interpolator ::  control_points is None")
            return np.zeros((x_.size, y_.size, 2), dtype=float)

        if self.bi_linear:
            return self.__interpolate_bi_linear_derivative(x_, y_, dx, dy)

        if self.bi_cubic:
            return self.__interpolate_bi_cubic_derivative(x_, y_, dx, dy)

        return np.zeros((x_.size, y_.size, 2), dtype=float)

    def interpolate_derivative2(self, x_: np.ndarray, y_: np.ndarray,
                                dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
        """
        Значение интерполяции массивов точкек с координатами х, у.\n
        :param x_: х - массив координат
        :param y_: y - массив координат
        :param dx: dх
        :param dy: dy
        :return:
        """
        if self.control_points is None:
            print("Interpolator ::  control_points is None")
            return np.zeros((x_.size, y_.size, 3), dtype=float)

        if self.bi_linear:
            return self.__interpolate_bi_linear_derivative2(x_, y_, dx, dy)

        if self.bi_cubic:
            return self.__interpolate_bi_cubic_derivative2(x_, y_, dx, dy)

        return np.zeros((x_.size, y_.size, 3), dtype=float)

    def cut(self, x_0: float, y_0: float, x_1: float, y_1: float, n_steps: int = 128) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Определяет координаты счения интерполируемой поверхности плоскостью проходящей через точки\n
        (x_0, y_0, 0.0), (x_1, y_1, 0.0) и вектором (0.0, 0.0, 1.0).\n
        :param x_0: кордината по х первой точки сечения
        :param y_0: кордината по y первой точки сечения
        :param x_1: кордината по х второй точки сечения
        :param y_1: кордината по y второй точки сечения
        :param n_steps: количество точек вдоль сечения
        :return: (точки по х, точки по y, точки по z)
        """
        if self.control_points is None:
            print("Interpolator ::  control_points is None")
            return np.zeros((n_steps,), dtype=float), np.zeros((n_steps,), dtype=float), np.zeros((n_steps,),
                                                                                                  dtype=float)
        if self.bi_linear:
            return self.__bi_linear_cut(x_0, y_0, x_1, y_1, n_steps)

        if self.bi_cubic:
            return self.__bi_cubic_cut(x_0, y_0, x_1, y_1, n_steps)

        return np.zeros((n_steps,), dtype=float), np.zeros((n_steps,), dtype=float), np.zeros((n_steps,), dtype=float)

    def curve_cut(self, pts_x: np.ndarray, pts_y: np.ndarray) -> np.ndarray:
        """
        Определяет координаты счения интерполируемой поверхности цилиндрической поверхностью, имеющей счение по (x, y)\n
        pts_x и pts_y.\n
        :param pts_x: координаты кривой сечения по х
        :param pts_y: координаты кривой сечения по y
        :return:
        """
        if pts_x.size != pts_y.size:
            raise ValueError("Interpolator :: cut_along_curve :: pts_x.size != pts_y.size")

        if self.control_points is None:
            print("Interpolator :: cut_along_curve ::  control_points is None")
            return np.zeros((pts_x.size,), dtype=float)

        if self.bi_linear:
            return self.__bi_linear_cut_along_curve(pts_x, pts_y)

        if self.bi_cubic:
            return self.__bi_cubic_cut_along_curve(pts_x, pts_y)

        return np.zeros((pts_x.size,), dtype=float)
    """
    def show_interpolation(self, n_: int = 32) -> None:
        x_ = np.linspace(self.x_0, self.x_0 + self.width, self.colons * n_ - 1, dtype=float)
        y_ = np.linspace(self.y_0, self.y_0 + self.height, self.rows * n_ - 1, dtype=float)
        # print(f"x-size: {x_.size}, y-size: {y_.size}")
        z_ = self.interpolate(x_, y_)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x_, y_ = np.meshgrid(x_, y_)
        # x_, y_ = rad_to_dec(y_, x_)
        jet = plt.get_cmap('jet')
        surf = ax.plot_surface(x_, y_, z_, cmap=jet, linewidth=0, antialiased=False, edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=8)
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.show()
    """

    @property
    def points_source(self) -> str:
        """
        Адресс файла со значениями узловых точек.\n
        :return:
        """
        return self.__points_file

    @points_source.setter
    def points_source(self, file_path) -> None:
        """
        Адресс файла со значениями узловых точек.\n
        :return:
        """
        self.load(file_path)

    @property
    def bi_linear(self) -> bool:
        """
        Включена би-линейная интерпляция или нет.\n
        :return:
        """
        return self.__mode == 0

    @property
    def bi_cubic(self) -> bool:
        """
        Включена би-кубическая интерпляция или нет.\n
        :return:
        """
        return self.__mode == 2

    @bi_linear.setter
    def bi_linear(self, val: bool) -> None:
        if val:
            self.__mode = 0

    @bi_cubic.setter
    def bi_cubic(self, val: bool) -> None:
        if val:
            self.__mode = 2

    @property
    def control_points(self) -> np.ndarray:
        """
        Узловые точки
        :return:
        """
        return self.__control_points

    @control_points.setter
    def control_points(self, points: np.ndarray) -> None:
        """
        Устанавливает новый набор узловых точек.\n
        :param points: новый набор узловых точек
        :return:
        """
        if points.ndim != 2:
            raise RuntimeError("Interpolator :: interpolator points has to be 2-dimensional array")

        self.__control_points = points
        self.__points_file = "no file"

    @property
    def n_control_points(self) -> int:
        """
        Узловые точки
        :return:
        """
        return self.__control_points.size

    @property
    def colons(self) -> int:
        """
        количество столбцов в узловых точек\n
        :return:
        """
        return self.control_points.shape[1]

    @property
    def rows(self) -> int:
        """
        количество столбцов в узловых точек\n
        :return:
        """
        return self.control_points.shape[0]

    @property
    def width(self) -> float:
        """
        Ширина области интерполяции\n
        :return:
        """
        return self.__width

    @width.setter
    def width(self, val: float) -> None:
        self.__width = max(0.0, val)

    @property
    def height(self) -> float:
        """
        Высота области интерполяции\n
        :return:
        """
        return self.__height

    @height.setter
    def height(self, val: float) -> None:
        self.__height = max(0.0, val)

    @property
    def x_0(self) -> float:
        """
        Смещение по х области интерполяции\n
        :return:
        """
        return self.__origin.x

    @property
    def y_0(self) -> float:
        """
        Смещение по y области интерполяции\n
        :return:
        """
        return self.__origin.y

    @property
    def z_0(self) -> float:
        """
        Смещение по z области интерполяции\n
        :return:
        """
        return self.__origin.z

    @x_0.setter
    def x_0(self, val: float) -> None:
        self.__origin.x = val

    @y_0.setter
    def y_0(self, val: float) -> None:
        self.__origin.y = val

    @z_0.setter
    def z_0(self, val: float) -> None:
        self.__origin.z = val

"""

"""
if __name__  == "__main__":
    res_i = Interpolator()
    res_i.bi_cubic = True
    #res_i.load("interpolator_a.json")
    #print(res_i)
    x = np.linspace(0, 1, 1024)
    # dxy  = res_i.interpolate_derivative2(x, x)
    lt = LoopTimer()
    with lt:
        d2xy = res_i.interpolate(x, x)
    print(f"loop time : {lt.last_loop_time}")
    #res_i.control_points = np.zeros((32, 32,), dtype=int)
    #print(res_i)
    # plt.imshow(d2xy[:, :])
    # plt.show()
