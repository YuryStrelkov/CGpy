import numpy as np
import ctypes
import math


class Vec2(object):

    @staticmethod
    def __unpack_values(*args) -> tuple:
        args = args[0]
        number_of_args = len(args)
        if number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is Vec2:
                return args[0].x, args[0].y

            if arg_type is float or arg_type is int:  # single int or float argument
                return args[0], args[0]

        if number_of_args == 0:
            return 0, 0  # no arguments

        if number_of_args == 2:
            return args[0], args[1]  # x, y and z passed in

        raise TypeError(f'Invalid Input: {args}')

    def from_np_array(self, data: np.ndarray) -> None:
        i: int = 0
        for element in data.ravel():
            self.__xy[i] = element
            i += 1
            if i == 2:
                break

    @staticmethod
    def dot(a, b) -> float:
        return a.x * b.x + a.y * b.y

    @staticmethod
    def cross(a, b) -> float:
        return a.y * b.x - a.x * b.y

    def unique_id(self) -> int:
        return id(self)

    def normalize(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        self.__xy[0] /= nrm
        self.__xy[1] /= nrm
        return self

    def normalized(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        return Vec2(self.__xy[0] / nrm, self.__xy[1] / nrm)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.__xy[0] * self.__xy[0] + self.__xy[1] * self.__xy[1])

    @property
    def np_array(self) -> np.ndarray:
        return np.array(self.__xy, dtype=np.float32)

    @property
    def as_array(self):
        return self.__xy

    @property
    def x(self) -> float: return self.__xy[0]

    @property
    def y(self) -> float: return self.__xy[1]

    @x.setter
    def x(self, x: float): self.__xy[0] = x

    @y.setter
    def y(self, y: float): self.__xy[1] = y

    @property
    def magnitude_sqr(self) -> float:
        return self.__xy[0] * self.__xy[0] + self.__xy[1] * self.__xy[1]

    __slots__ = "__xy"

    def __init__(self, x: float = 0, y: float = 0):
        self.__xy: [float] = [x, y]

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

    def __repr__(self):
        return f"<vec2[{self.__xy[0]:20},{self.__xy[1]:20}]>"

    def __str__(self):
        return f"{{\"x\": {self.__xy[0]:20}, \"y\": {self.__xy[1]:20}}}"

    ##########################
    #####  + operetor   ######
    ##########################

    def __add__(self, *args):
        other = self.__unpack_values(args)
        return Vec2(self.x + other[0], self.y + other[1])

    def __iadd__(self, *args):
        other = self.__unpack_values(args)
        self.x += other[0]
        self.y += other[1]
        return self

    __radd__ = __add__
    ##########################
    #####  - operetor   ######
    ##########################

    def __sub__(self, *args):
        other = self.__unpack_values(args)
        return Vec2(self.x - other[0], self.y - other[1])

    def __isub__(self, *args):
        other = self.__unpack_values(args)
        self.x -= other[0]
        self.y -= other[1]
        return self

    def __rsub__(self, *args):
        other = self.__unpack_values(args)
        return Vec2(other[0] - self.x, other[1] - self.y)
    ##########################
    #####  * operetor   ######
    ##########################

    def __mul__(self, *args):
        other = self.__unpack_values(args)
        return Vec2(self.x * other[0], self.y * other[1])

    def __imul__(self, *args):
        other = self.__unpack_values(args)
        self.x *= other[0]
        self.y *= other[1]
        return self

    __rmul__ = __mul__
    ##########################
    #####  / operetor   ######
    ##########################

    def __truediv__(self, *args):
        other = self.__unpack_values(args)
        return Vec2(self.x / other[0], self.y / other[1])

    def __rtruediv__(self,  *args):
        other = self.__unpack_values(args)
        return Vec2(other[0] / self.x, other[1] / self.y)

    def __itruediv__(self, *args):
        other = self.__unpack_values(args)
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


def dot2(a: Vec2, b: Vec2) -> float:
    return a.x * b.x + a.y * b.y


def max2(a: Vec2, b: Vec2) -> Vec2:
    return Vec2(max(a.x, b.x), max(a.y, b.y))


def min2(a: Vec2, b: Vec2) -> Vec2:
    return Vec2(min(a.x, b.x), min(a.y, b.y))


class Vec3(object):

    @staticmethod
    def __unpack_values(*args) -> tuple:
        args = args[0]

        number_of_args = len(args)

        if number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is Vec3:
                return args[0].x, args[0].y, args[0].z

            if arg_type is float or arg_type is int:  # single int or float argument
                return args[0], args[0], args[0]

        if number_of_args == 0:
            return 0, 0, 0  # no arguments

        if number_of_args == 3:
            return args[0], args[1], args[2]  # x, y and z passed in

        raise TypeError(f'Invalid Input: {args}')

    def from_np_array(self, data: np.ndarray) -> None:
        i: int = 0
        for element in data.ravel():
            self.__xyz[i] = element
            i += 1
            if i == 3:
                break

    @staticmethod
    def dot(a, b) -> float:
        return a.x * b.x + a.y * b.y + a.z * b.z

    @staticmethod
    def cross(a, b):
        return Vec3(a.z * b.y - a.y * b.z, a.x * b.z - a.z * b.x, a.y * b.x - a.x * b.y)

    def unique_id(self) -> int:
        return id(self)

    def normalize(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        self.__xyz[0] /= nrm
        self.__xyz[1] /= nrm
        self.__xyz[2] /= nrm
        return self

    def normalized(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        return Vec3(self.__xyz[0] / nrm, self.__xyz[1] / nrm, self.__xyz[2] / nrm)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.__xyz[0] * self.__xyz[0] + self.__xyz[1] * self.__xyz[1] + self.__xyz[2] * self.__xyz[2])

    @property
    def np_array(self) -> np.ndarray:
        return np.array(self.__xyz, dtype=np.float32)

    @property
    def as_array(self):
        return self.__xyz

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
    def magnitude_sqr(self) -> float:
        return self.__xyz[0] * self.__xyz[0] + self.__xyz[1] * self.__xyz[1] + self.__xyz[2] * self.__xyz[2]

    __slots__ = "__xyz"

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.__xyz: [float] = [x, y, z]

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

    def __repr__(self):
        return f"<vec3[{self.__xyz[0]:20},{self.__xyz[1]:20},{self.__xyz[2]:20}]>"

    def __str__(self):
        return f"{{\"x\": {self.__xyz[0]:20}, \"y\": {self.__xyz[1]:20}, \"z\": {self.__xyz[2]:20}}}"

    ##########################
    #####  + operetor   ######
    ##########################

    def __add__(self, *args):
        other = self.__unpack_values(args)
        return Vec3(self.x + other[0], self.y + other[1], self.z + other[2])

    def __iadd__(self, *args):
        other = self.__unpack_values(args)
        self.x += other[0]
        self.y += other[1]
        self.z += other[2]
        return self

    __radd__ = __add__
    ##########################
    #####  - operetor   ######
    ##########################

    def __sub__(self, *args):
        other = self.__unpack_values(args)
        return Vec3(self.x - other[0], self.y - other[1], self.z - other[2])

    def __isub__(self, *args):
        other = self.__unpack_values(args)
        self.x -= other[0]
        self.y -= other[1]
        self.z -= other[2]
        return self

    def __rsub__(self, *args):
        other = self.__unpack_values(args)
        return Vec3(other[0] - self.x, other[1] - self.y, other[2] - self.z)
    ##########################
    #####  * operetor   ######
    ##########################

    def __mul__(self, *args):
        other = self.__unpack_values(args)
        return Vec3(self.x * other[0], self.y * other[1], self.z * other[2])

    def __imul__(self, *args):
        other = self.__unpack_values(args)
        self.x *= other[0]
        self.y *= other[1]
        self.z *= other[2]
        return self

    __rmul__ = __mul__
    ##########################
    #####  / operetor   ######
    ##########################

    def __truediv__(self, *args):
        other = self.__unpack_values(args)
        return Vec3(self.x / other[0], self.y / other[1], self.z / other[2])

    def __rtruediv__(self,  *args):
        other = self.__unpack_values(args)
        return Vec3(other[0] / self.x, other[1] / self.y, other[2] / self.z)

    def __itruediv__(self, *args):
        other = self.__unpack_values(args)
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


def dot3(a: Vec3, b: Vec3) -> float: return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3: return Vec3(a.z * b.y - a.y * b.z, a.x * b.z - a.z * b.x, a.y * b.x - a.x * b.y)


def max3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(max(a.x, b.x), max(a.y, b.y), min(a.z, b.z))


def min3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))