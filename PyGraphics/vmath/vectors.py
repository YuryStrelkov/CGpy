import math

import numpy as np


class Vec2(object):
    @staticmethod
    def __unpack_values(*args) -> [float]:
        args = args[0]

        number_of_args = len(args)

        if number_of_args == 0:
            return [0, 0]  # no arguments

        elif number_of_args == 2:
            return [args[0], args[1]]  # x, y and z passed in

        elif number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is float or arg_type is int:  # single int or float argument
                return [args[0], args[0]]

            if arg_type is Vec2:
                return [args[0].x, args[0].y]

        raise TypeError(f'Invalid Input: {args}')

    def __init__(self, x: float = 0, y: float = 0):
        self.__xy: [float] = [x, y]

    def __eq__(self, other) -> bool:
        if not (type(other) is Vec2):
            return False
        if not (self.x == other.x):
            return False
        if not (self.y == other.y):
            return False
        return True

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str: return "<vec2 x:%s y:%s>" % (self.__xy[0], self.__xy[1])

    def __str__(self) -> str: return "[%s, %s]" % (self.__xy[0], self.__xy[1])

    def __add__(self, *args):
        other = self.__unpack_values(args)
        return Vec2(self.x + other[0], self.y + other[1])

    def __sub__(self, *args):
        other = self.__unpack_values(args)
        return Vec2(self.x - other[0], self.y - other[1])

    def __mul__(self, *args):
        other = self.__unpack_values(args)
        return Vec2(self.x * other[0], self.y * other[1])

    def __truediv__(self, *args):
        other = self.__unpack_values(args)
        return Vec2(self.x / other[0], self.y / other[1])

    def __getitem__(self, index):
        if index < 0 or index >= 2:
            raise IndexError(f"vec2 :: trying to access index: {index}")
        return self.__xy[index]

    def norm(self) -> float: return math.sqrt(self.__xy[0] * self.__xy[0] + self.__xy[1] * self.__xy[1])

    def normalize(self):
        nrm = self.norm()
        if abs(nrm) < 1e-12:
            raise ArithmeticError("vec2::zero length vector")
        self.__xy[0] /= nrm
        self.__xy[1] /= nrm
        return self

    @property
    def magnitude(self) -> float: return math.sqrt(self.__xy[0] * self.__xy[0] + self.__xy[1] * self.__xy[1])

    @property
    def x(self) -> float: return self.__xy[0]

    @property
    def y(self) -> float: return self.__xy[1]

    @x.setter
    def x(self, x_: float): self.__xy[0] = x_

    @y.setter
    def y(self, y_: float): self.__xy[1] = y_


def dot2(a: Vec2, b: Vec2) -> float: return a.x * b.x + a.y * b.y


def max2(a: Vec2, b: Vec2) -> Vec2:
    return Vec2(max(a.x, b.x), max(a.y, b.y))


def min2(a: Vec2, b: Vec2) -> Vec2:
    return Vec2(min(a.x, b.x), min(a.y, b.y))


class Vec3(object):
    @staticmethod
    def __unpack_values(*args) -> [float]:
        args = args[0]

        number_of_args = len(args)

        if number_of_args == 0:
            return [0, 0, 0]  # no arguments

        elif number_of_args == 3:
            return [args[0], args[1], args[2]]  # x, y and z passed in

        elif number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is float or arg_type is int:  # single int or float argument
                return [args[0], args[0], args[0]]

            if arg_type is Vec3:
                return [args[0].x, args[0].y, args[0].z]

        raise TypeError(f'Invalid Input: {args}')

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.__xyz: [float] = [x, y, z]

    def __eq__(self, other):
        if not (type(other) is Vec3):
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

    def __repr__(self): return "<vec3 x:%s y:%s z:%s>" % (self.__xyz[0], self.__xyz[1], self.__xyz[2])

    def __str__(self): return "[%s, %s, %s]" % (self.__xyz[0], self.__xyz[1], self.__xyz[2])

    def __add__(self, *args):
        other = self.__unpack_values(args)
        return Vec3(self.x + other[0], self.y + other[1], self.z + other[2])

    def __sub__(self, *args):
        other = self.__unpack_values(args)
        return Vec3(self.x - other[0], self.y - other[1], self.z - other[2])

    def __mul__(self, *args):
        other = self.__unpack_values(args)
        return Vec3(self.x * other[0], self.y * other[1], self.z * other[2])

    def __truediv__(self, *args):
        other = self.__unpack_values(args)
        return Vec3(self.x / other[0], self.y / other[1], self.z / other[2])

    def __getitem__(self, index):
        if index < 0 or index >= 3:
            raise IndexError(f"vec3 :: trying to access index: {index}")
        return self.__xyz[index]

    def norm(self) -> float: return math.sqrt(
        self.__xyz[0] * self.__xyz[0] + self.__xyz[1] * self.__xyz[1] + self.__xyz[2] * self.__xyz[2])

    def normalize(self):
        nrm = self.norm()
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        self.__xyz[0] /= nrm
        self.__xyz[1] /= nrm
        self.__xyz[2] /= nrm
        return self

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.__xyz[0] * self.__xyz[0] + self.__xyz[1] * self.__xyz[1] + self.__xyz[2] * self.__xyz[2])

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


def dot3(a: Vec3, b: Vec3) -> float: return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3: return Vec3(a.z * b.y - a.y * b.z, a.x * b.z - a.z * b.x, a.y * b.x - a.x * b.y)


def max3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(max(a.x, b.x), max(a.y, b.y), min(a.z, b.z))


def min3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
