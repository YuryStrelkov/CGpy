import numpy as np


class Vec2(object):
    def __init__(self, x: float = 0, y: float = 0): self.xy: [float] = [x, y]

    def __repr__(self): return "<vec2 x:%s y:%s>" % (self.xy[0], self.xy[1])

    def __str__(self): return "[%s, %s]" % (self.xy[0], self.xy[1])

    def __add__(self, other): return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other): return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other): return Vec2(self.x * other.x, self.y * other.y)

    def __truediv__(self, other): return Vec2(self.x / other.x, self.y / other.y)

    def __mul__(self, other: float): return Vec2(self.x * other, self.y * other)

    def __truediv__(self, other: float): return Vec2(self.x / other, self.y / other)

    def norm(self) -> float: return np.sqrt(self.xy[0] * self.xy[0] + self.xy[1] * self.xy[1])

    def normalize(self):
        nrm = self.norm()
        if abs(nrm) < 1e-12:
            raise ArithmeticError("vec2::zero length vector")
        self.xy[0] /= nrm
        self.xy[1] /= nrm
        return self

    @property
    def magnitude(self) -> float: return np.sqrt(self.xy[0] * self.xy[0] + self.xy[1] * self.xy[1])

    @property
    def x(self) -> float: return self.xy[0]

    @property
    def y(self) -> float: return self.xy[1]

    @x.setter
    def x(self, x_: float): self.xy[0] = x_

    @y.setter
    def y(self, y_: float): self.xy[1] = y_


def dot2(a: Vec2, b: Vec2) -> float: return a.x * b.x + a.y * b.y


def max2(a: Vec2, b: Vec2) -> Vec2:
    return Vec2(max(a.x, b.x), max(a.y, b.y))


def min2(a: Vec2, b: Vec2) -> Vec2:
    return Vec2(min(a.x, b.x), min(a.y, b.y))


class Vec3(object):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0): self.xyz: [float] = [x, y, z]

    def norm(self) -> float: return np.sqrt(
        self.xyz[0] * self.xyz[0] + self.xyz[1] * self.xyz[1] + self.xyz[2] * self.xyz[2])

    def __repr__(self): return "<vec3 x:%s y:%s z:%s>" % (self.xyz[0], self.xyz[1], self.xyz[2])

    def __str__(self): return "[%s, %s, %s]" % (self.xyz[0], self.xyz[1], self.xyz[2])

    def __add__(self, other): return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other): return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other): return Vec3(self.x * other.x, self.y * other.y, self.z / other.z)

    def __truediv__(self, other): return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)

    def __mul__(self, other: float): return Vec3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float): return Vec3(self.x / other, self.y / other, self.z / other)

    def normalize(self):
        nrm = self.norm()
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        self.xyz[0] /= nrm
        self.xyz[1] /= nrm
        self.xyz[2] /= nrm
        return self

    @property
    def magnitude(self) -> float:
        return np.sqrt(self.xyz[0] * self.xyz[0] + self.xyz[1] * self.xyz[1] + self.xyz[2] * self.xyz[2])

    @property
    def x(self) -> float: return self.xyz[0]

    @property
    def y(self) -> float: return self.xyz[1]

    @property
    def z(self) -> float: return self.xyz[2]

    @x.setter
    def x(self, x: float): self.xyz[0] = x

    @y.setter
    def y(self, y: float): self.xyz[1] = y

    @z.setter
    def z(self, z: float): self.xyz[2] = z


def dot3(a: Vec3, b: Vec3) -> float: return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3: return Vec3(a.z * b.y - a.y * b.z, a.x * b.z - a.z * b.x, a.y * b.x - a.x * b.y)

