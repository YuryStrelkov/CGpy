import numpy as np
import math
import vmath.vectors as vectors
from vmath.matrices import Mat4, Mat3
from vmath.vectors import Vec3, Vec2


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
    xaxis = vectors.cross(up, zaxis)  # The "right" vector.
    xaxis.normalize()
    yaxis = vectors.cross(zaxis, xaxis)  # The "up" vector.

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


def perpendicular_3(v: Vec3) -> Vec3:
    s: float = np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    g: float = np.copysign(s, v.z)  # note s instead of 1
    h: float = v.z + g
    return Vec3(g * h - v.x * v.x, -v.x * v.y, -v.x * h)
