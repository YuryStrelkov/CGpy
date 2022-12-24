from core import geometry_utils
from typing import List, Tuple
from core.matrices import Mat4
import numpy as np
import ctypes
import math


class Quaternion:
    @staticmethod
    def __unpack_args(*args) -> tuple:
        args = args[0]
        number_of_args = len(args)
        if number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is Quaternion:
                return args[0].ex, args[0].ey, args[0].ey, args[0].ew

            if arg_type is float or arg_type is int:  # single int or float argument
                return 1, 0, 0, args[0]

        if number_of_args == 4:
            return args

        if number_of_args == 0:
            return 1.0, 0.0, 0.0, 0.0  # no arguments

        raise TypeError(f'Invalid Input: {args}')

    def __build_from_matrix(self, m: Mat4) -> None:
        tr = m.m00 + m.m11 + m.m22
        if tr > 0.0:
            s = math.sqrt(tr + 1.0)
            self.ew = s * 0.5
            s = 0.5 / s
            self.ex = (m.m12 - m.m21) * s
            self.ey = (m.m20 - m.m02) * s
            self.ez = (m.m01 - m.m10) * s
            return
        i: int = 0
        j: int = 0
        k: int = 0
        if m.m11 > m.m00:
            if m.m22 > m.m11:
                i = 2
                j = 0
                k = 1
            else:
                i = 1
                j = 2
                k = 0
        elif m.m22 > m.m00:
            i = 2
            j = 0
            k = 1
        else:
            i = 0
            j = 1
            k = 2
        s = math.sqrt((m[i * 4 + i] - (m[j * 4 + j] + m[k * 4 + k])) + 1.0)
        self.__quaternion[i] = s * 0.5
        if s != 0.0:
            s = 0.5 / s
        self.__quaternion[j] = (m[i * 4 + j] + m[j * 4 + i]) * s
        self.__quaternion[k] = (m[i * 4 + k] + m[k * 4 + i]) * s
        self.__quaternion[3] = (m[j * 4 + k] - m[k * 4 + j]) * s

    def __build_from_angles(self, ax: float, ay: float, az: float) -> None:
        self.__build_from_matrix(geometry_utils.rotate(ax, ay, az))

    def __build_from_quaternion(self, quaternion) -> None:
        self.ex = quaternion.ex
        self.ey = quaternion.ey
        self.ez = quaternion.ez
        self.ew = quaternion.ew

    __slots__ = "__quaternion"

    def __init__(self, *args):
        self.__quaternion: List[float] = list(Quaternion.__unpack_args(args))

    def __sizeof__(self):
        return ctypes.c_float * 4

    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return False
        if not (self.ex == other.ex):
            return False
        if not (self.ey == other.ey):
            return False
        if not (self.ez == other.ez):
            return False
        if not (self.ew == other.ew):
            return False
        return True

    def __hash__(self):
        return hash((self.ex, self.ey, self.ez, self.ew))

    def __neg__(self):
        return Quaternion(-self.ex, -self.ey, -self.ez, -self.ew)

    def __copy__(self):
        return Quaternion(self.ex, self.ey, self.ez, self.ew)

    copy = __copy__

    def __str__(self):
        return f"{{\"ex\": {self.ex:20}, \"ey\": {self.ey:20}, \"ez\": {self.ez:20}, \"angle\": {self.ew:20}}}"

    def __getitem__(self, index):
        if index < 0 or index >= 4:
            raise IndexError(f"Quaternion :: trying to access index: {index}")
        return self.__quaternion[index]

    def __setitem__(self, index: int, value: float):
        if index < 0 or index >= 4:
            raise IndexError(f"Quaternion :: trying to access index: {index}")
        self.__quaternion[index] = value

    ##########################
    #####  + operetor   ######
    ##########################

    def __add__(self, *args):
        other = self.__unpack_args(args)
        return Quaternion(self.ex + other[0],
                          self.ey + other[1],
                          self.ez + other[2],
                          self.ew + other[3])

    def __iadd__(self, *args):
        other = self.__unpack_args(args)
        self.ex    += other[0]
        self.ey    += other[1]
        self.ez    += other[2]
        self.ew += other[43]
        return self

    __radd__ = __add__

    ##########################
    #####  - operetor   ######
    ##########################

    def __sub__(self, *args):
        other = self.__unpack_args(args)
        return Quaternion(self.ex - other[0],
                          self.ey - other[1],
                          self.ez - other[2],
                          self.ew - other[3])

    def __isub__(self, *args):
        other = self.__unpack_args(args)
        self.ex -= other[0]
        self.ey -= other[1]
        self.ez -= other[2]
        self.ew -= other[3]
        return self

    def __rsub__(self, *args):
        other = self.__unpack_args(args)
        return Quaternion(other[0] - self.ex,
                          other[1] - self.ey,
                          other[2] - self.ez,
                          other[3] - self.ew)

    ##########################
    #####  * operetor   ######
    ##########################

    # def __mul__(self, *args):
    #     ex, ey, ez, ew = self.__unpack_args(args)
    #     return Vec2(self.x * other[0], self.y * other[1])

    def __imul__(self, *args):
        """
        https://github.com/BennyQBD/3DGameEngine/blob/master/src/com/base/engine/core/Quaternion.java
        """
        ex, ey, ez, ew = self.__unpack_args(args)
        w_ = self.ew * ew - self.ex * ex - self.ey * ey - self.ez * ez
        x_ = self.ex * ew + self.ew * ex + self.ey * ez - self.ez * ey
        y_ = self.ey * ew + self.ew * ey + self.ez * ex - self.ex * ez
        z_ = self.ez * ew + self.ew * ez + self.ex * ey - self.ey * ex
        self.ex = x_
        self.ey = y_
        self.ez = z_
        self.ew = w_
        return self

    # __rmul__ = __mul__

    def unique_id(self) -> int:
        return id(self)

    def from_np_array(self, data: np.ndarray) -> None:
        for element_id, element in enumerate(data.flat):
            self.__quaternion[element_id] = element
            if element_id == 2:
                break

    def normalize(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        nrm = 1.0 / nrm
        self.ex *= nrm
        self.ey *= nrm
        self.ez *= nrm
        self.ew *= nrm
        return self

    def conj(self):
        """
        conj self
        :return:
        """
        self.ex *= -1.0
        self.ey *= -1.0
        self.ez *= -1.0
        return self

    def conjugate(self):
        """
        returns new conjugated Quaternion
        :return:
        """
        return Quaternion(self.ex * -1.0,
                          self.ey * -1.0,
                          self.ez * -1.0,
                          self.ew)

    def inverse(self):
        self.conjugate()
        self.normalize()
        return self

    def normalized(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        nrm = 1.0 / nrm
        return Quaternion(self.ex * nrm, self.ey * nrm, self.ez * nrm, self.ew * nrm)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.ex * self.ex +
                         self.ey * self.ey +
                         self.ez * self.ez +
                         self.ew * self.ew)

    @property
    def np_array(self) -> np.ndarray:
        return np.array(self.__quaternion, dtype=np.float32)

    @property
    def as_list(self) -> List[float]:
        return self.__quaternion

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.__quaternion[0], self.__quaternion[1], self.__quaternion[2], self.__quaternion[3]

    @property
    def ex(self) -> float: return self.__quaternion[0]

    @property
    def ey(self) -> float: return self.__quaternion[1]

    @property
    def ez(self) -> float: return self.__quaternion[2]

    @property
    def ew(self) -> float: return self.__quaternion[3]

    @ex.setter
    def ex(self, x: float) -> None: self.__quaternion[0] = x

    @ey.setter
    def ey(self, y: float) -> None: self.__quaternion[1] = y

    @ez.setter
    def ez(self, z: float) -> None: self.__quaternion[2] = z

    @ew.setter
    def ew(self, angle: float)  -> None: self.__quaternion[3] = angle



