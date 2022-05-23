import numpy as np
import math


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


class Mat3(object):
    def __init__(self,
                 m0: float = 0, m1: float = 0, m2: float = 0,
                 m3: float = 0, m4: float = 0, m5: float = 0,
                 m6: float = 0, m7: float = 0, m8: float = 0):
        self.data: [float] = [m0, m1, m2, m3, m4, m5, m6, m7, m8]

    def __getitem__(self, key: int) -> float: return self.data[key]

    def __setitem__(self, key: int, value: float): self.data[key] = value

    def __repr__(self):
        res: str = "mat4:\n"
        res += "[[%s, %s, %s],\n" % (self.data[0], self.data[1], self.data[2])
        res += " [%s, %s, %s],\n" % (self.data[3], self.data[4], self.data[5])
        res += " [%s, %s, %s],\n" % (self.data[6], self.data[7], self.data[8])
        return res

    def __str__(self):
        res: str = ""
        res += "[[%s, %s, %s],\n" % (self.data[0], self.data[1], self.data[2])
        res += " [%s, %s, %s],\n" % (self.data[3], self.data[4], self.data[5])
        res += " [%s, %s, %s],\n" % (self.data[6], self.data[7], self.data[8])
        return res

    # row 1 set/get
    @property
    def m00(self) -> float: return self.data[0]

    @m00.setter
    def m00(self, val: float): self.data[0] = val

    @property
    def m01(self) -> float: return self.data[1]

    @m01.setter
    def m01(self, val: float): self.data[1] = val

    @property
    def m02(self) -> float: return self.data[2]

    @m02.setter
    def m02(self, val: float): self.data[2] = val

    # row 2 set/get
    @property
    def m10(self) -> float: return self.data[3]

    @m10.setter
    def m10(self, val: float): self.data[3] = val

    @property
    def m11(self) -> float: return self.data[4]

    @m11.setter
    def m11(self, val: float): self.data[4] = val

    @property
    def m12(self) -> float: return self.data[5]

    @m12.setter
    def m12(self, val: float): self.data[5] = val

    # row 3 set/get
    @property
    def m20(self) -> float: return self.data[6]

    @m20.setter
    def m20(self, val: float): self.data[6] = val

    @property
    def m21(self) -> float: return self.data[7]

    @m21.setter
    def m21(self, val: float): self.data[7] = val

    @property
    def m22(self) -> float: return self.data[8]

    @m22.setter
    def m22(self, val: float): self.data[8] = val


def mat_mul_3(a: Mat3, b: Mat3) -> Mat3:
    return Mat3(
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],

        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],

        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
    )


class Mat4(object):

    def __init__(self,
                 m0: float = 0, m1: float = 0, m2: float = 0, m3: float = 0,
                 m4: float = 0, m5: float = 0, m6: float = 0, m7: float = 0,
                 m8: float = 0, m9: float = 0, m10: float = 0, m11: float = 0,
                 m12: float = 0, m13: float = 0, m14: float = 0, m15: float = 0):
        self.data: [float] = [m0, m1, m2, m3,
                              m4, m5, m6, m7,
                              m8, m9, m10, m11,
                              m12, m13, m14, m15]

    def __getitem__(self, key: int) -> float: return self.data[key]

    def __setitem__(self, key: int, value: float): self.data[key] = value

    def __repr__(self):
        res: str = "mat4:\n"
        res += "[[%s, %s, %s, %s],\n" % (self.data[0], self.data[1], self.data[2], self.data[3])
        res += " [%s, %s, %s, %s],\n" % (self.data[4], self.data[5], self.data[6], self.data[7])
        res += " [%s, %s, %s, %s],\n" % (self.data[8], self.data[9], self.data[10], self.data[11])
        res += " [%s, %s, %s, %s]]" % (self.data[12], self.data[13], self.data[14], self.data[15])
        return res

    def __str__(self):
        res: str = ""
        res += "[[%s, %s, %s, %s],\n" % (self.data[0], self.data[1], self.data[2], self.data[3])
        res += " [%s, %s, %s, %s],\n" % (self.data[4], self.data[5], self.data[6], self.data[7])
        res += " [%s, %s, %s, %s],\n" % (self.data[8], self.data[9], self.data[10], self.data[11])
        res += " [%s, %s, %s, %s]]" % (self.data[12], self.data[13], self.data[14], self.data[15])
        return res

    # row 1 set/get
    @property
    def m00(self) -> float: return self.data[0]

    @m00.setter
    def m00(self, val: float): self.data[0] = val

    @property
    def m01(self) -> float: return self.data[1]

    @m01.setter
    def m01(self, val: float): self.data[1] = val

    @property
    def m02(self) -> float: return self.data[2]

    @m02.setter
    def m02(self, val: float): self.data[2] = val

    @property
    def m03(self) -> float: return self.data[3]

    @m03.setter
    def m03(self, val: float): self.data[3] = val

    # row 2 set/get
    @property
    def m10(self) -> float: return self.data[4]

    @m10.setter
    def m10(self, val: float): self.data[4] = val

    @property
    def m11(self) -> float: return self.data[5]

    @m11.setter
    def m11(self, val: float): self.data[5] = val

    @property
    def m12(self) -> float: return self.data[6]

    @m12.setter
    def m12(self, val: float): self.data[6] = val

    @property
    def m13(self) -> float: return self.data[7]

    @m13.setter
    def m13(self, val: float): self.data[7] = val

    # row 3 set/get
    @property
    def m20(self) -> float: return self.data[8]

    @m20.setter
    def m20(self, val: float): self.data[8] = val

    @property
    def m21(self) -> float: return self.data[9]

    @m21.setter
    def m21(self, val: float): self.data[9] = val

    @property
    def m22(self) -> float: return self.data[10]

    @m22.setter
    def m22(self, val: float): self.data[10] = val

    @property
    def m23(self) -> float: return self.data[11]

    @m23.setter
    def m23(self, val: float): self.data[11] = val

    # row 4 set/get
    @property
    def m30(self) -> float: return self.data[12]

    @m30.setter
    def m30(self, val: float): self.data[12] = val

    @property
    def m31(self) -> float: return self.data[13]

    @m31.setter
    def m31(self, val: float): self.data[13] = val

    @property
    def m32(self) -> float: return self.data[14]

    @m32.setter
    def m32(self, val: float): self.data[14] = val

    @property
    def m33(self) -> float: return self.data[15]

    @m33.setter
    def m33(self, val: float): self.data[15] = val


def mat_mul_4(a: Mat4, b: Mat4) -> Mat4:
    return Mat4(a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12],
                a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13],
                a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14],
                a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15],

                a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12],
                a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13],
                a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14],
                a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15],

                a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12],
                a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13],
                a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14],
                a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15],

                a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12],
                a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13],
                a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14],
                a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15])


def rotate_x(angle: float) -> Mat4:
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return Mat4(1, 0, 0, 0,
                0, cos_a, -sin_a, 0,
                0, sin_a, cos_a, 0,
                0, 0, 0, 1)


def rotate_y(angle: float) -> Mat4:
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return Mat4(cos_a, 0, -sin_a, 0,
                0, 1, 0, 0,
                sin_a, 0, cos_a, 0,
                0, 0, 0, 1)


def rotate_z(angle: float) -> Mat4:
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return Mat4(cos_a, -sin_a, 0, 0,
                sin_a, cos_a, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1)


def deg_to_rad(deg: float) -> float: return deg / 180.0 * np.pi


def rad_to_deg(deg: float) -> float: return deg / np.pi * 180.0


def rot_m_to_euler_angles(rot: Mat4) -> Vec3:
    if rot.m02 + 1 < 1e-6:
        return Vec3(0, np.pi * 0.5, math.atan2(rot.m10, rot.m20))

    if rot.m02 - 1 < 1e-6:
        return Vec3(0, -np.pi * 0.5, math.atan2(-rot.m10, -rot.m20))

    x1 = -np.asin(rot.z)
    x2 = np.pi - x1
    y1 = math.atan2(rot.m12 / np.cos(x1), rot.m22 / np.cos(x1))
    y2 = math.atan2(rot.m12 / np.cos(x2), rot.m22 / np.cos(x2))
    z1 = math.atan2(rot.m01 / np.cos(x1), rot.m00 / np.cos(x1))
    z2 = math.atan2(rot.m01 / np.cos(x2), rot.m00 / np.cos(x2))
    if (abs(x1) + abs(y1) + abs(z1)) <= (abs(x2) + abs(y2) + abs(z2)):
        return Vec3(x1, y1, z1)

    return Vec3(x2, y2, z2)


def look_at(target: Vec3, eye: Vec3, up: Vec3 = Vec3(0, 1, 0)) -> Mat4:
    zaxis = target - eye  # The "forward" vector.
    zaxis.normalize()
    xaxis = cross(up, zaxis)  # The "right" vector.
    xaxis.normalize()
    yaxis = cross(zaxis, xaxis)  # The "up" vector.

    return Mat4(xaxis.x, -yaxis.x, zaxis.x, eye.x,
                -xaxis.y, -yaxis.y, zaxis.y, eye.y,
                xaxis.z, -yaxis.z, zaxis.z, eye.z,
                0, 0, 0, 1)


def clamp(min_: float, max_: float, val: float) -> float:
    if val < min_:
        return min_
    if val > max_:
        return max_
    return val


def lerp_vec_2(a: Vec2, b: Vec2, t: float) -> Vec2:
    return Vec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)


def lerp_vec_3(a: Vec3, b: Vec3, t: float) -> Vec3:
    return Vec3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t)


def lerp_mat_3(a: Mat3, b: Mat3, t: float) -> Mat3:
    return Mat3(a.m00 + (b.m00 - a.m00) * t, a.m01 + (b.m01 - a.m01) * t, a.m02 + (b.m02 - a.m02) * t,
                a.m10 + (b.m10 - a.m10) * t, a.m11 + (b.m11 - a.m11) * t, a.m12 + (b.m12 - a.m12) * t,
                a.m20 + (b.m20 - a.m20) * t, a.m21 + (b.m21 - a.m21) * t, a.m22 + (b.m22 - a.m22) * t)


def lerp_mat_4(a: Mat4, b: Mat4, t: float) -> Mat4:
    return Mat4(
        a.m00 + (b.m00 - a.m00) * t, a.m01 + (b.m01 - a.m01) * t, a.m02 + (b.m02 - a.m02) * t,
        a.m03 + (b.m03 - a.m03) * t,
        a.m10 + (b.m10 - a.m10) * t, a.m11 + (b.m11 - a.m11) * t, a.m12 + (b.m12 - a.m12) * t,
        a.m13 + (b.m13 - a.m13) * t,
        a.m20 + (b.m20 - a.m20) * t, a.m21 + (b.m21 - a.m21) * t, a.m22 + (b.m22 - a.m22) * t,
        a.m23 + (b.m23 - a.m23) * t,
        a.m30 + (b.m30 - a.m30) * t, a.m31 + (b.m31 - a.m31) * t, a.m32 + (b.m32 - a.m32) * t,
        a.m33 + (b.m33 - a.m33) * t)


def perpendicular_2(v: Vec2) -> Vec2:
    if v.x == 0:
        return Vec2(np.sign(v.y), 0)
    if v.y == 0:
        return Vec2(0, -np.sign(v.x))
    sign: float = np.sign(v.x / v.y)
    dx: float = 1.0 / v.x
    dy: float = -1.0 / v.y
    sign /= np.sqrt(dx * dx + dy * dy)
    return Vec2(dx * sign, dy * sign)
