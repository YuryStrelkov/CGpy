from typing import List, Tuple
import numpy as np
import ctypes
import math


def _smooth_step(value: float, bound_1: float, bound_2: float) -> float:
    if value <= bound_1:
        return bound_1
    if value >= bound_2:
        return bound_2
    x = (value - bound_1) / (bound_2 - bound_1)
    return x * x * (3 - 2 * x)


class Vec2:

    @staticmethod
    def __unpack_args(*args) -> Tuple[float, float]:
        args = args[0]
        n_args = len(args)
        if n_args == 1:  # one argument
            args = args[0]
            if isinstance(args, Vec2):
                return args.x, args.y
            if isinstance(args, float) or isinstance(args, int):
                return args, args
        if n_args == 2:
            return args[0], args[1]  # x, y and z passed in
        if n_args == 0:
            return 0.0, 0.0
        raise TypeError(f'Invalid Input: {args}')

    __slots__ = "__xy"

    def __sizeof__(self):
        return ctypes.c_float * 2

    def __init__(self, *args):
        self.__xy: List[float] = list(Vec2.__unpack_args(args))

    def __eq__(self, other):
        if not isinstance(other, Vec2):
            return False
        if not (self.x == other.x):
            return False
        if not (self.y == other.y):
            return False
        return True

    def __hash__(self):
        return hash((self.x, self.y))

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __copy__(self):
        return Vec2(self.x, self.y)

    copy = __copy__

    def __str__(self):
        return f"{{\"x\": {self.__xy[0]:20}, \"y\": {self.__xy[1]:20}}}"

    ##########################
    #####  + operetor   ######
    ##########################

    def __add__(self, *args):
        other = self.__unpack_args(args)
        return Vec2(self.x + other[0], self.y + other[1])

    def __iadd__(self, *args):
        other = self.__unpack_args(args)
        self.x += other[0]
        self.y += other[1]
        return self

    __radd__ = __add__
    ##########################
    #####  - operetor   ######
    ##########################

    def __sub__(self, *args):
        other = self.__unpack_args(args)
        return Vec2(self.x - other[0], self.y - other[1])

    def __isub__(self, *args):
        other = self.__unpack_args(args)
        self.x -= other[0]
        self.y -= other[1]
        return self

    def __rsub__(self, *args):
        other = self.__unpack_args(args)
        return Vec2(other[0] - self.x, other[1] - self.y)
    ##########################
    #####  * operetor   ######
    ##########################

    def __mul__(self, *args):
        other = self.__unpack_args(args)
        return Vec2(self.x * other[0], self.y * other[1])

    def __imul__(self, *args):
        other = self.__unpack_args(args)
        self.x *= other[0]
        self.y *= other[1]
        return self

    __rmul__ = __mul__
    ##########################
    #####  / operetor   ######
    ##########################

    def __truediv__(self, *args):
        other = self.__unpack_args(args)
        return Vec2(self.x / other[0], self.y / other[1])

    def __rtruediv__(self,  *args):
        other = self.__unpack_args(args)
        return Vec2(other[0] / self.x, other[1] / self.y)

    def __itruediv__(self, *args):
        other = self.__unpack_args(args)
        self.x /= other[0]
        self.y /= other[1]
        return self

    def __getitem__(self, index):
        if index < 0 or index >= 2:
            raise IndexError(f"Vec2 :: trying to access index: {index}")
        return self.__xy[index]

    def __setitem__(self, index: int, value: float):
        if index < 0 or index >= 2:
            raise IndexError(f"Vec2 :: trying to access index: {index}")
        self.__xy[index] = value

    @classmethod
    def smooth_step(cls, t: float, a, b):
        return cls(_smooth_step(t, a.x, b.x),
                   _smooth_step(t, a.y, b.y))

    @staticmethod
    def dot(a, b) -> float:
        return a.x * b.x + a.y * b.y

    @staticmethod
    def cross(a, b) -> float:
        return a.y * b.x - a.x * b.y

    @classmethod
    def max(cls, a,  b):
        return cls(max(a.x, b.x), max(a.y, b.y))

    @classmethod
    def min(cls, a, b):
        return cls(min(a.x, b.x), min(a.y, b.y))

    @staticmethod
    def reflect(n, e):
        return e - 2.0 * (Vec2.dot(n, e)) * n

    @staticmethod
    def refract(n, e, ri_ratio: float):
        pass

    @property
    def unique_id(self) -> int:
        return id(self)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.__xy[0] * self.__xy[0] + self.__xy[1] * self.__xy[1])

    @property
    def np_array(self) -> np.ndarray:
        return np.array(self.__xy, dtype=np.float32)

    @property
    def as_list(self) -> List[float]:
        return self.__xy

    @property
    def as_tuple(self) -> Tuple[float, float]:
        return self.__xy[0], self.__xy[1]

    @property
    def x(self) -> float: return self.__xy[0]

    @property
    def y(self) -> float: return self.__xy[1]

    @x.setter
    def x(self, x: float): self.__xy[0] = x

    @y.setter
    def y(self, y: float): self.__xy[1] = y

    @property
    def u(self) -> float: return self.__xy[1]

    @property
    def v(self) -> float: return self.__xy[0]

    @u.setter
    def u(self, y: float): self.__xy[1] = y

    @v.setter
    def v(self, x: float): self.__xy[0] = x

    @property
    def magnitude_sqr(self) -> float:
        return self.__xy[0] * self.__xy[0] + self.__xy[1] * self.__xy[1]

    def from_np_array(self, data: np.ndarray) -> None:
        for element_id, element in enumerate(data.flat):
            self.__xy[element_id] = element
            if element_id == 1:
                break

    def normalize(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        nrm = 1.0 / nrm
        self.__xy[0] *= nrm
        self.__xy[1] *= nrm
        return self

    def normalized(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        nrm = 1.0 / nrm
        return Vec2(self.__xy[0] * nrm, self.__xy[1] * nrm)


class Vec3:

    @staticmethod
    def __unpack_args(*args) -> Tuple[float, float, float]:
        args = args[0]
        n_args = len(args)
        if n_args == 1:  # one argument
            args = args[0]
            if isinstance(args, Vec3):
                return args.x, args.y, args.z
            if isinstance(args, float) or isinstance(args, int):
                return args, args, args
        if n_args == 3:
            return args[0], args[1], args[2]  # x, y and z passed in
        if n_args == 0:
            return 0.0, 0.0, 0.0
        raise TypeError(f'Invalid Input: {args}')

    __slots__ = "__xyz"

    def __init__(self, *args):
        self.__xyz: List[float] = list(Vec3.__unpack_args(args))

    def __sizeof__(self):
        return ctypes.c_float * 3

    def __eq__(self, other):
        if not isinstance(other, Vec3):
            return False
        if not (self.x == other.x):
            return False
        if not (self.y == other.y):
            return False
        if not (self.z == other.z):
            return False
        return True

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __copy__(self):
        return Vec3(self.x, self.y, self.z)

    copy = __copy__

    def __str__(self):
        return f"{{\"x\": {self.__xyz[0]:20}, \"y\": {self.__xyz[1]:20}, \"z\": {self.__xyz[2]:20}}}"

    ##########################
    #####  + operetor   ######
    ##########################

    def __add__(self, *args):
        other = self.__unpack_args(args)
        return Vec3(self.x + other[0], self.y + other[1], self.z + other[2])

    def __iadd__(self, *args):
        other = self.__unpack_args(args)
        self.x += other[0]
        self.y += other[1]
        self.z += other[2]
        return self

    __radd__ = __add__
    ##########################
    #####  - operetor   ######
    ##########################

    def __sub__(self, *args):
        other = self.__unpack_args(args)
        return Vec3(self.x - other[0], self.y - other[1], self.z - other[2])

    def __isub__(self, *args):
        other = self.__unpack_args(args)
        self.x -= other[0]
        self.y -= other[1]
        self.z -= other[2]
        return self

    def __rsub__(self, *args):
        other = self.__unpack_args(args)
        return Vec3(other[0] - self.x, other[1] - self.y, other[2] - self.z)
    ##########################
    #####  * operetor   ######
    ##########################

    def __mul__(self, *args):
        other = self.__unpack_args(args)
        return Vec3(self.x * other[0], self.y * other[1], self.z * other[2])

    def __imul__(self, *args):
        other = self.__unpack_args(args)
        self.x *= other[0]
        self.y *= other[1]
        self.z *= other[2]
        return self

    __rmul__ = __mul__
    ##########################
    #####  / operetor   ######
    ##########################

    def __truediv__(self, *args):
        other = self.__unpack_args(args)
        return Vec3(self.x / other[0], self.y / other[1], self.z / other[2])

    def __rtruediv__(self,  *args):
        other = self.__unpack_args(args)
        return Vec3(other[0] / self.x, other[1] / self.y, other[2] / self.z)

    def __itruediv__(self, *args):
        other = self.__unpack_args(args)
        self.x /= other[0]
        self.y /= other[1]
        self.z /= other[2]
        return self

    def __getitem__(self, index):
        if index < 0 or index >= 3:
            raise IndexError(f"Vec3 :: trying to access index: {index}")
        return self.__xyz[index]

    def __setitem__(self, index: int, value: float):
        if index < 0 or index >= 3:
            raise IndexError(f"Vec3 :: trying to access index: {index}")
        self.__xyz[index] = value

    @classmethod
    def smooth_step(cls, t: float, a, b):
        return cls(_smooth_step(t, a.x, b.x),
                   _smooth_step(t, a.y, b.y),
                   _smooth_step(t, a.z, b.z))

    @staticmethod
    def dot(a, b) -> float:
        return a.x * b.x + a.y * b.y + a.z * b.z

    @classmethod
    def cross(cls, a, b):
        return cls(a.z * b.y - a.y * b.z, a.x * b.z - a.z * b.x, a.y * b.x - a.x * b.y)

    @classmethod
    def max(cls, a, b):
        return cls(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))

    @classmethod
    def min(cls, a, b):
        return cls(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))

    @staticmethod
    def reflect(n, e):
        return e - 2.0 * (Vec3.dot(n, e)) * n

    @property
    def unique_id(self) -> int:
        return id(self)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.__xyz[0] * self.__xyz[0] + self.__xyz[1] * self.__xyz[1] + self.__xyz[2] * self.__xyz[2])

    @property
    def np_array(self) -> np.ndarray:
        return np.array(self.__xyz, dtype=np.float32)

    @property
    def as_list(self) -> List[float]:
        return self.__xyz

    @property
    def as_tuple(self) -> Tuple[float, float, float]:
        return self.__xyz[0], self.__xyz[1], self.__xyz[2]

    @property
    def x(self) -> float: return self.__xyz[0]

    @property
    def y(self) -> float: return self.__xyz[1]

    @property
    def z(self) -> float: return self.__xyz[2]

    @x.setter
    def x(self, x: float): self.__xyz[0] = x

    @y.setter
    def y(self, y: float): self.__xyz[1] = y

    @z.setter
    def z(self, z: float): self.__xyz[2] = z

    @property
    def u(self) -> float: return self.__xyz[1]

    @property
    def v(self) -> float: return self.__xyz[0]

    @property
    def w(self) -> float: return self.__xyz[2]

    @u.setter
    def u(self, x: float): self.__xyz[1] = x

    @v.setter
    def v(self, y: float): self.__xyz[0] = y

    @w.setter
    def w(self, z: float): self.__xyz[2] = z

    @property
    def magnitude_sqr(self) -> float:
        return self.__xyz[0] * self.__xyz[0] + self.__xyz[1] * self.__xyz[1] + self.__xyz[2] * self.__xyz[2]

    def normalize(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        nrm = 1.0 / nrm
        self.__xyz[0] *= nrm
        self.__xyz[1] *= nrm
        self.__xyz[2] *= nrm
        return self

    def normalized(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        nrm = 1.0 / nrm
        return Vec3(self.__xyz[0] * nrm, self.__xyz[1] * nrm, self.__xyz[2] * nrm)

    def from_np_array(self, data: np.ndarray) -> None:
        for element_id, element in enumerate(data.flat):
            self.__xyz[element_id] = element
            if element_id == 2:
                break
