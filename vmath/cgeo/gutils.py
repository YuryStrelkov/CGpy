from cgeo.matrices import Mat4, Mat3
from cgeo.vectors import Vec3, Vec2
from typing import Tuple, List
from cgeo import mutils
import numpy as np
# import numba
import math

numerical_precision: float = 1e-9
Vector2 = Tuple[float, float]


def rotate_x(angle: float) -> Mat4:
    """
    :param angle: угол поворота вокру оси х
    :return: матрица поворота вокруг оси x
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Mat4(1, 0, 0, 0,
                0, cos_a, -sin_a, 0,
                0, sin_a, cos_a, 0,
                0, 0, 0, 1)


def rotate_y(angle: float) -> Mat4:
    """
     :param angle: угол поворота вокру оси y
     :return: матрица поворота вокруг оси y
     """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Mat4(cos_a, 0, sin_a, 0,
                0, 1, 0, 0,
                -sin_a, 0, cos_a, 0,
                0, 0, 0, 1)


def rotate_z(angle: float) -> Mat4:
    """
     :param angle: угол поворота вокру оси z
     :return: матрица поворота вокруг оси z
     """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Mat4(cos_a, -sin_a, 0, 0,
                sin_a, cos_a, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1)


def rotate(angle_x: float, angle_y: float, angle_z: float) -> Mat4:
    """
    :param angle_x: угол поворота вокру оси x
    :param angle_y: угол поворота вокру оси y
    :param angle_z: угол поворота вокру оси z
    :return: матрица поворота
    """
    return rotate_x(angle_x) * rotate_y(angle_y) * rotate_z(angle_z)


# @numba.njit(fastmath=True)
def deg_to_rad(deg: float) -> float:
    """
    :param deg: угол в градусах
    :return: угол в радианах
    """
    return deg / 180.0 * math.pi


# @numba.njit(fastmath=True)
def rad_to_deg(deg: float) -> float:
    """
    :param deg: угол в радианах
    :return: угол в градусах
    """
    return deg / math.pi * 180.0


def rot_m_to_euler_angles(rot: Mat4) -> Vec3:
    """
    :param rot: матрица поворота
    :return: углы поворота по осям
    """

    # psi, theta, phi = x, y, z

    if math.fabs(rot.m20 + 1) < 1e-6:
        return Vec3(0, math.pi * 0.5, math.atan2(rot.m01, rot.m02))

    if math.fabs(rot.m20 - 1) < 1e-6:
        return Vec3(0, -math.pi * 0.5, math.atan2(-rot.m01, -rot.m02))
    """
        y = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        x = math.atan2(rot[2,1]/cos_theta, rot[2,2]/cos_theta)
        z = math.atan2(rot[1,0]/cos_theta, rot[0,0]/cos_theta)
    """
    x1 = -math.asin(rot.m20)
    inv_cos_x1 = 1.0 / math.cos(x1)
    x2 = math.pi + x1
    inv_cos_x2 = 1.0 / math.cos(x1)

    y1 = math.atan2(rot.m21 * inv_cos_x1, rot.m22 * inv_cos_x1)
    y2 = math.atan2(rot.m21 * inv_cos_x2, rot.m22 * inv_cos_x2)
    z1 = math.atan2(rot.m10 * inv_cos_x1, rot.m00 * inv_cos_x1)
    z2 = math.atan2(rot.m10 * inv_cos_x2, rot.m00 * inv_cos_x2)
    if (abs(x1) + abs(y1) + abs(z1)) <= (abs(x2) + abs(y2) + abs(z2)):
        return Vec3(y1, x1, z1)

    return Vec3(y2, x2, z2)


def look_at(target: Vec3, eye: Vec3, up: Vec3 = Vec3(0, 1, 0)) -> Mat4:
    """
    :param target: цель на которую смотрим
    :param eye: положение глаза в пространстве
    :param up: вектор вверх
    :return: матрица взгляда
    """
    zaxis = target - eye  # The "forward" vector.
    zaxis.normalize()
    xaxis = Vec3.cross(up, zaxis)  # The "right" vector.
    xaxis.normalize()
    yaxis = Vec3.cross(zaxis, xaxis)  # The "up" vector.

    return Mat4(xaxis.x, -yaxis.x, zaxis.x, eye.x,
                -xaxis.y, -yaxis.y, zaxis.y, eye.y,
                xaxis.z, -yaxis.z, zaxis.z, eye.z,
                0, 0, 0, 1)


def lin_interp_vec2(a: Vec2, b: Vec2, t: float) -> Vec2:
    """
    :param a: вектор начала
    :param b: вектор конца
    :param t: параметр в пределах от 0 до 1
    :return: возвращает линейную интерполяцию между двухмерными векторами a и b
    """
    return Vec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)


def lin_interp_vec3(a: Vec3, b: Vec3, t: float) -> Vec3:
    """
    :param a: вектор начала
    :param b: вектор конца
    :param t: параметр в пределах от 0 до 1
    :return: возвращает линейную интерполяцию между трёхмерными векторами a и b
    """
    return Vec3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t)


def lin_interp_mat3(a: Mat3, b: Mat3, t: float) -> Mat3:
    """
    :param a: матрица начала
    :param b: матрица конца
    :param t: параметр в пределах от 0 до 1
    :return: возвращает линейную интерполяцию между трёхмернми матрицами a и b
    """
    return Mat3(a.m00 + (b.m00 - a.m00) * t, a.m01 + (b.m01 - a.m01) * t, a.m02 + (b.m02 - a.m02) * t,
                a.m10 + (b.m10 - a.m10) * t, a.m11 + (b.m11 - a.m11) * t, a.m12 + (b.m12 - a.m12) * t,
                a.m20 + (b.m20 - a.m20) * t, a.m21 + (b.m21 - a.m21) * t, a.m22 + (b.m22 - a.m22) * t)


def lin_interp_mat4(a: Mat4, b: Mat4, t: float) -> Mat4:
    """
    :param a: матрица начала
    :param b: матрица конца
    :param t: параметр в пределах от 0 до 1
    :return: возвращает линейную интерполяцию между четырёхмерными матрицами a и b
    """
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
    """
    :param v:
    :return: возвращает единичный вектор пермендикулярный заданному
    """
    if v.x == 0:
        return Vec2(mutils.signum(v.y), 0)
    if v.y == 0:
        return Vec2(0, -mutils.signum(v.x))
    sign: float = mutils.signum(v.x / v.y)
    dx: float = 1.0 / v.x
    dy: float = -1.0 / v.y
    sign /= math.sqrt(dx * dx + dy * dy)
    return Vec2(dx * sign, dy * sign)


def perpendicular_3(v: Vec3) -> Vec3:
    """
    :param v:
    :return: возвращает единичный вектор пермендикулярный заданному
    """
    s: float = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    g: float = math.copysign(s, v.z)  # note s instead of 1
    h: float = v.z + g
    return Vec3(g * h - v.x * v.x, -v.x * v.y, -v.x * h)


def build_projection_matrix(fov: float = 70, aspect: float = 1, z_near: float = 0.01, z_far: float = 1000) -> Mat4:
    """
    :param fov: угол обзора
    :param aspect: соотношение сторон
    :param z_near: ближняя плоскость отсечения
    :param z_far: дальняя плоскость отсечения
    :return: матрица перспективной проекции
    """
    projection = Mat4.identity()
    scale = 1.0 / math.tan(fov * 0.5 * math.pi / 180)
    projection.m00 = scale * aspect  # scale the x coordinates of the projected point
    projection.m11 = scale  # scale the y coordinates of the projected point
    projection.m22 = z_far / (z_near - z_far)  # used to remap z to [0,1]
    projection.m32 = z_far * z_near / (z_near - z_far)  # used to remap z [0,1]
    projection.m23 = -1  # set w = -z
    projection.m33 = 0
    return projection


def build_orthogonal_basis(right: Vec3, up: Vec3, front: Vec3, main_axes=3) -> Mat4:
    """
    :param right:
    :param up:
    :param front:
    :param main_axes:
    :return:
    """
    x_: Vec3
    y_: Vec3
    z_: Vec3
    while True:
        if main_axes == 1:
            f_mag = right.magnitude
            z_ = right.normalized()
            y_ = (up - Vec3.dot(z_, up) * z_ / (f_mag * f_mag)).normalized()
            x_ = Vec3.cross(z_, y_).normalized()
            break
        if main_axes == 2:
            f_mag = up.magnitude
            z_ = up.normalized()
            y_ = (front - Vec3.dot(z_, front) * front / (f_mag * f_mag)).normalized()
            x_ = Vec3.cross(z_, y_).normalized()
            break
        if main_axes == 3:
            f_mag = front.magnitude
            z_ = front.normalized()
            y_ = (up - Vec3.dot(z_, up) * z_ / (f_mag * f_mag)).normalized()
            x_ = Vec3.cross(z_, y_).normalized()
            break
        raise RuntimeError("wrong parameter main_axes")
    m = Mat4(x_.x, y_.x, z_.x, 0,
             x_.y, y_.y, z_.y, 0,
             x_.z, y_.z, z_.z, 0,
             0, 0, 0, 1)
    return m


def _bezier_coordinate(t: float, x1: float, x2: float, x3: float, x4: float) -> float:
    one_min_t: float = 1.0 - t
    return x1 * one_min_t * one_min_t * one_min_t + \
           x2 * 3.0 * one_min_t * one_min_t * t + \
           x3 * 3.0 * one_min_t * t * t + x4 * t * t * t


def _section_bounds_1d(x1: float, x2: float, x3: float, x4: float) -> Tuple[float, float]:
    a: float = -3 * x1 + 9 * x2 - 9 * x3 + 3 * x4
    b: float =  6 * x1 - 12 * x2 + 6 * x3
    c: float = -3 * x1 + 3 * x2
    if abs(a) < 1e-6:
        if abs(b) < 1e-6:
            return min(x1, x4), max(x1, x4)
        t = _bezier_coordinate(-c / b, x1, x2, x3, x4)
        return min(min(t, x1), x4), max(max(t, x1), x4)
    det: float = b * b - 4.0 * a * c
    if det < 0:
        return min(min(min(x1, x2), x3), x4), max(max(max(x1, x2), x3), x4)
    det = math.sqrt(det)
    x_1: float = _bezier_coordinate(max(0.0, min(1.0, (-b + det) * 0.5 / a)), x1, x2, x3, x4)
    x_2: float = _bezier_coordinate(max(0.0, min(1.0, (-b - det) * 0.5 / a)), x1, x2, x3, x4)
    return min(min(min(x_1, x_2), x1), x4), max(max(max(x_1, x_2), x1), x4)


def bezier_2_section_bounds(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Tuple[Vec2, Vec2]:
    x_0, x_1 = _section_bounds_1d(p1.x, p2.x, p3.x, p4.x)
    y_0, y_1 = _section_bounds_1d(p1.y, p2.y, p3.y, p4.y)
    return Vec2(min(x_0, x_1), min(y_0, y_1)), Vec2(max(x_0, x_1), max(y_0, y_1))


def bezier_3_section_bounds(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3) -> Tuple[Vec3, Vec3]:
    x_0, x_1 = _section_bounds_1d(p1.x, p2.x, p3.x, p4.x)
    y_0, y_1 = _section_bounds_1d(p1.y, p2.y, p3.y, p4.y)
    z_0, z_1 = _section_bounds_1d(p1.z, p2.z, p3.z, p4.z)
    return Vec3(min(x_0, x_1), min(y_0, y_1), min(z_0, z_1)), Vec3(max(x_0, x_1), max(y_0, y_1), max(z_0, z_1))


def bezier_2_cubic(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, t: float) -> Vec2:
    """
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :param t:
    :return: координаты точки на кривой
    """
    one_min_t: float = 1.0 - t
    a: float = one_min_t * one_min_t * one_min_t
    b: float = 3.0 * one_min_t * one_min_t * t
    c: float = 3.0 * one_min_t * t * t
    d: float = t * t * t
    return Vec2(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                p1.y * a + p2.y * b + p3.y * c + p4.y * d)


def bezier_2_tangent(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, t: float) -> Vec2:
    """
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :param t:
    :return: касательная для точки на кривой
    """
    d: float = 3.0 * t * t
    a: float = -3.0 + 6.0 * t - d
    b: float = 3.0 - 12.0 * t + 3.0 * d
    c: float = 6.0 * t - 3.0 * d
    return Vec2(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                p1.y * a + p2.y * b + p3.y * c + p4.y * d)


def bezier_3_cubic(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3, t: float) -> Vec3:
    """
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :param t:
    :return: координаты точки на кривой
    """
    one_min_t: float = 1.0 - t
    a: float = one_min_t * one_min_t * one_min_t
    b: float = 3.0 * one_min_t * one_min_t * t
    c: float = 3.0 * one_min_t * t * t
    d: float = t * t * t
    return Vec3(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                p1.y * a + p2.y * b + p3.y * c + p4.y * d,
                p1.z * a + p2.z * b + p3.z * c + p4.z * d)


def bezier_3_tangent(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3, t: float) -> Vec3:
    """
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :param t:
    :return: касательная для точки на кривой
    """
    d: float = 3.0 * t * t
    a: float = -3.0 + 6.0 * t - d
    b: float = 3.0 - 12.0 * t + 3.0 * d
    c: float = 6.0 * t - 3.0 * d
    return Vec3(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                p1.y * a + p2.y * b + p3.y * c + p4.y * d,
                p1.z * a + p2.z * b + p3.z * c + p4.z * d)


def distance_to_bezier_3(point: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3, iterations: int = 2, slices: int = 32):
    t: float
    dt: float
    best_t: float = 0.0
    min_dist: float = 0.0
    curr_dist: float = 0.0
    _start: float = 0.0
    _end: float = 1.0
    while iterations >= 0:
        dt = (_end - _start) * 1.0 / slices
        t = _start
        best_t = 0.0
        min_dist = 1e12
        while t <= _end:
            curr_dist = (bezier_3_cubic(p1, p2, p3, p4, t) - point).magnitude
            if curr_dist < min_dist:
                min_dist = curr_dist
                best_t = t
            t += dt
        iterations -= 1
        _start = max(best_t - dt, 0.0)
        _end = min(best_t + dt, 1.0)
    return Vec2((_start + _end) * 0.5, math.sqrt(min_dist))


def distance_to_bezier_2(point: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, iterations: int = 2, slices: int = 32):
    t: float
    dt: float
    best_t: float = 0.0
    min_dist: float = 0.0
    curr_dist: float = 0.0
    _start: float = 0.0
    _end: float = 1.0
    while iterations >= 0:
        dt = (_end - _start) * 1.0 / slices
        t = _start
        best_t = 0.0
        min_dist = 1e12
        while t <= _end:
            curr_dist = (bezier_2_cubic(p1, p2, p3, p4, t) - point).magnitude
            if curr_dist < min_dist:
                min_dist = curr_dist
                best_t = t
            t += dt
        iterations -= 1
        _start = max(best_t - dt, 0.0)
        _end = min(best_t + dt, 1.0)
    return Vec2((_start + _end) * 0.5, math.sqrt(min_dist))


def split_bezier_section_2(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, t: float) -> Tuple[Vec2, Vec2, Vec2, Vec2]:
    x12 = (p2.x - p1.x) * t + p1.x
    y12 = (p2.y - p1.y) * t + p1.y

    x23 = (p3.x - p2.x) * t + p2.x
    y23 = (p3.y - p2.y) * t + p2.y

    x34 = (p4.x - p3.x) * t + p3.x
    y34 = (p4.y - p3.y) * t + p3.y

    x123 = (x23 - x12) * t + x12
    y123 = (y23 - y12) * t + y12

    x234 = (x34 - x23) * t + x23
    y234 = (y34 - y23) * t + y23

    x1234 = (x234 - x123) * t + x123
    y1234 = (y234 - y123) * t + y123

    return Vec2(p1.x, p1.y), Vec2(x12, y12), Vec2(x123, y123), Vec2(x1234, y1234)


def split_bezier_section_3(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3, t: float) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    x12 = (p2.x - p1.x) * t + p1.x
    y12 = (p2.y - p1.y) * t + p1.y
    z12 = (p2.z - p1.z) * t + p1.z

    x23 = (p3.x - p2.x) * t + p2.x
    y23 = (p3.y - p2.y) * t + p2.y
    z23 = (p3.z - p2.z) * t + p2.z

    x34 = (p4.x - p3.x) * t + p3.x
    y34 = (p4.y - p3.y) * t + p3.y
    z34 = (p4.z - p3.z) * t + p3.z

    x123 = (x23 - x12) * t + x12
    y123 = (y23 - y12) * t + y12
    z123 = (z23 - z12) * t + z12

    x234 = (x34 - x23) * t + x23
    y234 = (y34 - y23) * t + y23
    z234 = (z34 - z23) * t + z23

    x1234 = (x234 - x123) * t + x123
    y1234 = (y234 - y123) * t + y123
    z1234 = (z234 - z123) * t + z123

    return Vec3(p1.x, p1.y, p1.z), Vec3(x12, y12, z12), Vec3(x123, y123, z123), Vec3(x1234, y1234, z1234)


def quadratic_bezier_patch(p1: Vec3, p2: Vec3, p3: Vec3,
                           p4: Vec3, p5: Vec3, p6: Vec3,
                           p7: Vec3, p8: Vec3, p9: Vec3, u: float, v: float) -> Tuple[Vec3, Vec3]:
    """
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :param p5:
    :param p6:
    :param p7:
    :param p8:
    :param p9:
    :param u:
    :param v:
    :return:
    """
    phi1: float = (1.0 - u) * (1.0 - u)
    phi3: float = u * u
    phi2: float = -2.0 * phi3 + 2.0 * u

    psi1: float = (1.0 - v) * (1.0 - v)
    psi3: float = v * v
    psi2: float = -2.0 * psi3 + 2.0 * v

    p: Vec3 = p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + \
              p4 * phi2 * psi1 + p5 * phi2 * psi2 + p6 * phi2 * psi3 + \
              p7 * phi3 * psi1 + p8 * phi3 * psi2 + p9 * phi3 * psi3

    dp1: float = -2.0 + u * 2.0
    dp2: float = 2.0 * u
    dp3: float = -2.0 * psi3 + 2.0

    du: Vec3 = p1 * dp1 * psi1 + p2 * dp1 * psi2 + p3 * dp1 * psi3 + \
               p4 * dp2 * psi1 + p5 * dp2 * psi2 + p6 * dp2 * psi3 + \
               p7 * dp3 * psi1 + p8 * dp3 * psi2 + p9 * dp3 * psi3

    dp1 = -2.0 + v * 2.0
    dp2 = 2.0 * v
    dp3 = -2.0 * psi3 + 2.0

    dv: Vec3 = p1 * phi1 * dp1 + p2 * phi1 * dp2 + p3 * phi1 * dp3 + \
               p4 * phi2 * dp1 + p5 * phi2 * dp2 + p6 * phi2 * dp3 + \
               p7 * phi3 * dp1 + p8 * phi3 * dp2 + p9 * phi3 * dp3

    return p, Vec3.cross(dv, du).normalize()


def cubic_bezier_patch(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3,
                       p5: Vec3, p6: Vec3, p7: Vec3, p8: Vec3,
                       p9: Vec3, p10: Vec3, p11: Vec3, p12: Vec3,
                       p13: Vec3, p14: Vec3, p15: Vec3, p16: Vec3, u: float, v: float) -> Tuple[Vec3, Vec3]:
    """
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :param p5:
    :param p6:
    :param p7:
    :param p8:
    :param p9:
    :param p10:
    :param p11:
    :param p12:
    :param p13:
    :param p14:
    :param p15:
    :param p16:
    :param u:
    :param v:
    :return:
    """

    phi1: float = (1.0 - u) * (1.0 - u) * (1.0 - u)
    phi4: float = u * u * u
    phi2: float = 3.0 * phi4 - 6.0 * u * u + 3.0 * u
    phi3: float = -3.0 * phi4 + 3.0 * u * u

    psi1: float = (1.0 - v) * (1.0 - v) * (1.0 - v)
    psi4: float = v * v * v
    psi2: float = 3.0 * psi4 - 6.0 * v * v + 3.0 * v
    psi3: float = -3.0 * psi4 + 3.0 * v * v

    p: Vec3 = p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + p4 * phi1 * psi4 + \
              p5 * phi2 * psi1 + p6 * phi2 * psi2 + p7 * phi2 * psi3 + p8 * phi2 * psi4 + \
              p9 * phi3 * psi1 + p10 * phi3 * psi2 + p11 * phi3 * psi3 + p12 * phi3 * psi4 + \
              p13 * phi4 * psi1 + p14 * phi4 * psi2 + p15 * phi4 * psi3 + p16 * phi4 * psi4

    d4: float = 3.0 * u * u
    d1: float = -3.0 + 6.0 * u - d4
    d2: float = 3.0 * phi4 - 12.0 * u + 3.0
    d3: float = -3.0 * phi4 + 6.0 * u

    dpu: Vec3 = p1 * d1 * psi1 + p2 * d1 * psi2 + p3 * d1 * psi3 + p4 * d1 * psi4 + \
                p5 * d2 * psi1 + p6 * d2 * psi2 + p7 * d2 * psi3 + p8 * d2 * psi4 + \
                p9 * d3 * psi1 + p10 * d3 * psi2 + p11 * d3 * psi3 + p12 * d3 * psi4 + \
                p13 * d4 * psi1 + p14 * d4 * psi2 + p15 * d4 * psi3 + p16 * d4 * psi4

    d4 = 3.0 * v * v
    d1 = -3.0 + 6.0 * v - d4
    d2 = 3.0 * phi4 - 12.0 * v + 3.0
    d3 = -3.0 * phi4 + 6.0 * v

    dpv: Vec3 = p1 * phi1 * d1 + p2 * phi1 * d2 + p3 * phi1 * d3 + p4 * phi1 * d4 + \
                p5 * phi2 * d1 + p6 * phi2 * d2 + p7 * phi2 * d3 + p8 * phi2 * d4 + \
                p9 * phi3 * d1 + p10 * phi3 * d2 + p11 * phi3 * d3 + p12 * phi3 * d4 + \
                p13 * phi4 * d1 + p14 * phi4 * d2 + p15 * phi4 * d3 + p16 * phi4 * d4

    return p, Vec3.cross(dpv, dpu).normalize()


def point_to_line_dist(point: Vec3, origin: Vec3, direction: Vec3) -> float:
    """
    Расстояние от точки до прямой.
    :arg point точка до которой ищем расстоняие
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return расстояние между точкой и прямой
    """
    return Vec3.cross(direction, (origin - point)).magnitude


def line_to_line_dist(origin_1: Vec3, direction_1: Vec3, origin_2: Vec3, direction_2: Vec3) -> float:
    """
    :arg origin_1 начало первого луча
    :arg direction_1 направление первого луча (единичный вектор)
    :arg origin_2 начало второго луча
    :arg direction_2 направление второго луча (единичный вектор)
    :return расстояние между первой и второй прямой
    """
    temp = Vec3.cross(direction_1, direction_2) * (origin_2 - origin_1)
    temp1 = Vec3.cross(direction_1, direction_2)
    return math.sqrt(Vec3.dot(temp, temp)) / math.sqrt(Vec3.dot(temp1, temp1))


# @numba.njit(fastmath=True)
def _rect_intersection(min_1: Vector2, max_1: Vector2,
                       min_2: Vector2, max_2: Vector2) -> Tuple[bool, Vector2, Vector2]:
    """
    AABB rectangles intersection test
    :param min_1: rect 1 min bound
    :param max_1: rect 1 max bound
    :param min_2: rect 2 min bound
    :param max_2: rect 2 max bound
    :return: true if intersection occurs / min max bound of overlap area
    """
    if min_1[0] > max_1[0]:
        t = min_1[0]
        min_1 = (max_1[0], min_1[1])
        max_1 = (t, max_1[1])

    if min_1[1] > max_1[1]:
        t = min_1[1]
        min_1 = (min_1[0], max_1[1])
        max_1 = (max_1[1], t)

    if min_2[0] > max_2[0]:
        t = min_2[0]
        min_2 = (max_2[0], min_2[1])
        max_2 = (t, max_2[1])

    if min_2[1] > max_2[1]:
        t = min_2[1]
        min_2 = (min_2[0], max_2[1])
        max_2 = (max_2[1], t)

    pt_max = (min(max_1[0], max_2[0]), min(max_1[1], max_2[1]))

    pt_min = (max(min_1[0], min_2[0]), max(min_1[1], min_2[1]))

    if pt_max[0] - pt_min[0] <= 0:
        return False, (0.0, 0.0), (0.0, 0.0)

    if pt_max[1] - pt_min[1] <= 0:
        return False, (0.0, 0.0), (0.0, 0.0)

    return True, pt_min, pt_max


def rect_intersection(min_1: Vec2, max_1: Vec2,
                      min_2: Vec2, max_2: Vec2) -> Tuple[bool, Vec2, Vec2]:
    """
    AABB rectangles intersection test
    :param min_1: rect 1 min bound
    :param max_1: rect 1 max bound
    :param min_2: rect 2 min bound
    :param max_2: rect 2 max bound
    :return: true if intersection occurs / min max bound of overlap area
    """
    flag, p1, p2 = _rect_intersection((min_1.x, min_1.y), (max_1.x, max_1.y),
                                      (min_2.x, min_2.y), (max_2.x, max_2.y))
    return flag, Vec2(p1[0], p1[1]), Vec2(p2[0], p2[0])


# @numba.njit(fastmath=True)
def _in_between(pt: Vector2, _min: Vector2, _max: Vector2) -> bool:
    """
    Определяет принадлежность точки прямоугольной области, ограниченнной точками _min и _max.\n
    :param pt: вектор - пара (x, y), точка для которой ищем принадлежность к области
    :param _min: вектор - пара (x, y), минимальная граница области
    :param _max: вектор - пара (x, y), максимальная граница области
    :return:
    """
    if abs((_min[0] + _max[0]) * 0.5 - pt[0]) > abs(_max[0] - _min[0]) * 0.5:
        return False
    if abs((_min[1] + _max[1]) * 0.5 - pt[1]) > abs(_max[1] - _min[1]) * 0.5:
        return False
    return True


def in_between(pt: Vec2, _min: Vec2, _max: Vec2) -> bool:
    """
    Определяет принадлежность точки прямоугольной области, ограниченнной точками _min и _max.\n
    :param pt: вектор - пара (x, y), точка для которой ищем принадлежность к области
    :param _min: вектор - пара (x, y), минимальная граница области
    :param _max: вектор - пара (x, y), максимальная граница области
    :return:
    """
    return _in_between((pt.x, pt.y), (_min.x, _min.y), (_max.x, _max.y))


# @numba.njit(fastmath=True)
def _cross(a: Vector2, b: Vector2) -> float:
    """
    Косое векторное произведение.\n
    :param a: первый вектор - пара (x, y)
    :param b: второй вектор - пара (x, y)
    :return:
    """
    return a[0] * b[1] - a[1] * b[0]


# @numba.njit(fastmath=True)
def _intersect_lines(pt1: Vector2, pt2: Vector2, pt3: Vector2, pt4: Vector2) -> Tuple[bool, Vector2]:
    """
    Определяет точку пересечения двух линий, проходящих через точки pt1, pt2 и pt3, pt4 для первой и второй\n
    соответственно.\n
    :param pt1: вектор - пара (x, y), первая точка первой линии
    :param pt2: вектор - пара (x, y), вторая точка первой линии
    :param pt3: вектор - пара (x, y), первая точка второй линии
    :param pt4: вектор - пара (x, y), вторая точка второй линии
    :return: переселись или нет, вектор - пара (x, y)
    """
    da = (pt2[0] - pt1[0], pt2[1] - pt1[1])

    db = (pt4[0] - pt3[0], pt4[1] - pt3[1])

    det = _cross(da, db)

    if abs(det) < numerical_precision:
        return False, (0, 0)

    det = 1.0 / det

    x = _cross(pt1, da)

    y = _cross(pt3, db)

    return True, ((y * da[0] - x * db[0]) * det, (y * da[1] - x * db[1]) * det)


def intersect_lines(pt1: Vec2, pt2: Vec2, pt3: Vec2, pt4: Vec2) -> Tuple[bool, Vec2]:
    """
    Определяет точку пересечения двух линий, проходящих через точки pt1, pt2 и pt3, pt4 для первой и второй\n
    соответственно.\n
    :param pt1: вектор - пара (x, y), первая точка первой линии
    :param pt2: вектор - пара (x, y), вторая точка первой линии
    :param pt3: вектор - пара (x, y), первая точка второй линии
    :param pt4: вектор - пара (x, y), вторая точка второй линии
    :return: переселись или нет, вектор - пара (x, y)
    """
    flag, (x, y) = _intersect_lines((pt1.x, pt1.y),
                                    (pt2.x, pt2.y),
                                    (pt3.x, pt3.y),
                                    (pt4.x, pt4.y))
    return flag, Vec2(x, y)


def intersect_sects(pt1: Vec2, pt2: Vec2, pt3: Vec2, pt4: Vec2) -> Tuple[bool, Vec2]:
    """
    Определяет точку пересечения двух отрезков, проходящих через точки pt1, pt2 и pt3, pt4 для первого и второго\n
    соответственно.\n
    :param pt1: вектор - пара (x, y), первая точка первого отрезка
    :param pt2: вектор - пара (x, y), вторая точка первого отрезка
    :param pt3: вектор - пара (x, y), первая точка второго отрезка
    :param pt4: вектор - пара (x, y), вторая точка второго отрезка
    :return: переселись или нет, вектор - пара (x, y)
    """
    flag, int_point = intersect_lines(pt1, pt2, pt3, pt4)
    if not flag:
        return flag, int_point

    if not in_between(int_point, pt1, pt2):
        return False, int_point

    if not in_between(int_point, pt3, pt4):
        return False, int_point

    return flag, int_point


# @numba.njit(fastmath=True)
def _cw(a: Vector2, b: Vector2, c: Vector2) -> bool:
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) < 0


def cw(a: Vec2, b: Vec2, c: Vec2) -> bool:
    return _cw((a.x, a.y), (b.x, b.y), (c.x, c.y))


def ccw(a: Vec2, b: Vec2, c: Vec2) -> bool:
    return not cw(a, b, c)


def bin_search(points: List[Vec2], x_val: float) -> int:
    """
    Бинарный поиск вхождения позиции x_val внутри points.\n
    :param points: список точек среди которых ищем вхождение
    :param x_val: значение координаты по х для которой ищем вхождение
    :return: индекс точки из points слева от x_val
    """
    left_id = 0

    right_id = len(points) - 1

    if points[right_id][0] < x_val:
        return right_id

    if points[left_id][0] > x_val:
        return left_id

    while right_id - left_id != 1:
        mid = (right_id + left_id) // 2
        if points[mid][0] < x_val:
            left_id = mid
            continue
        right_id = mid
    return right_id


def point_within_polygon(point: Vec2, polygon: List[Vec2]) -> bool:
    """
    https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    :param point:
    :param polygon:
    :return:
    """
    n = len(polygon)

    inside = False

    px: float
    py: float

    px, py = point.as_tuple

    p1x: float
    p1y: float

    p2x: float
    p2y: float

    x_ints: float = 0.0

    p1x, p1y = polygon[0].as_tuple
    for i in range(n + 1):
        p2x, p2y = polygon[i % n].as_tuple
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        x_ints = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= x_ints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_within_polygons(point: Vec2, polygons: List[List[Vec2]]) -> bool:
    intersections: int = sum(1 if point_within_polygon(point, polygon) else 0 for polygon in polygons)
    return intersections % 2 != 0


def polygon_perimeter(polygon: List[Vec2]) -> float:
    perimeter: float = 0.0
    for i in range(len(polygon) - 1):
        perimeter += math.sqrt((polygon[i].x - polygon[i + 1].x) ** 2 + (polygon[i].y - polygon[i + 1].y) ** 2)
    return perimeter


def polygon_area(polygon: List[Vec2]) -> float:
    area: float = 0.0
    for i in range(len(polygon) - 1):
        area += Vec2.cross(polygon[i], polygon[i + 1])
    return area * 0.5


def polygon_centroid(polygon: List[Vec2]) -> Vec2:
    cx: float = 0.0
    cy: float = 0.0
    area = 1.0 / 6.0 / polygon_area(polygon)
    for i in range(len(polygon) - 1):
        xi, yi = polygon[i].as_tuple
        xj, yj = polygon[i + 1].as_tuple
        t = (xi * yj - xj * yi)
        cx += (xi + xj) * t
        cy += (yi + yj) * t
    return Vec2(cx * area, cy * area)


def polygon_bounds(polygon: List[Vec2]) -> Tuple[Vec2, Vec2]:
    min_b = Vec2(1e12, 1e12)
    max_b = Vec2(-1e12, -1e12)
    for p in polygon:
        if p.x > max_b.x:
            max_b.x = p.x
        if p.y > max_b.y:
            max_b.y = p.y
        if p.x < min_b.x:
            min_b.x = p.x
        if p.y < min_b.y:
            min_b.y = p.y
    return min_b, max_b


def cw_polygon(polygon: List[Vec2]) -> bool:
    return polygon_area(polygon) >= 0


def ccw_polygon(polygon: List[Vec2]) -> bool:
    return not cw_polygon(polygon)


def angle2(a: Vec2, b: Vec2, c: Vec2) -> float:
    v1 = (b.x - a.x, b.y - a.y)
    v2 = (b.x - c.x, b.y - c.y)
    rho1 = math.sqrt(v1[0]**2 + v1[1]**2)
    rho2 = math.sqrt(v2[0]**2 + v2[1]**2)
    return np.arccos(mutils.clamp((v1[0] * v2[0] + v1[1] * v2[1]) / rho1 / rho2, -1.0, 1.0))


def angle3(a: Vec3, b: Vec3, c: Vec3) -> float:
    v1 = (b.x - a.x, b.y - a.y, b.z - a.z)
    v2 = (b.x - c.x, b.y - c.y, b.z - a.z)
    rho1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
    rho2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    return np.arccos(mutils.clamp((v1[0] * v2[0] + v1[1] * v2[1] +  v1[2] * v2[2]) / rho1 / rho2, -1.0, 1.0))


def plane_to_point_dist(r_0: Vec3, n: Vec3, point: Vec3) -> float:
    """
    Уравнение плоскости (r - r_0, n) = 0
    :arg r_0 точка через которую проходит плоскость
    :arg n нормаль к плоскости (единичный вектор)
    :arg point точка для которой ищем расстояние
    :return расстояние между точкой и плоскостью
    """
    return Vec3.dot(point - r_0, n)


def point_to_line_seg_distance(point: Vec2, point1: Vec2, point2: Vec2, threshold: float = 0.01) -> float:
    return _point_to_line_seg_distance(point.as_tuple, point1.as_tuple, point2.as_tuple, threshold)


# @numba.njit(fastmath=True)
def _point_to_line_seg_distance_sign(point: Vector2, point0: Vector2, point1: Vector2, threshold: float = 0.01) -> float:

    p1_p0 = (point1[0] - point0[0], point1[1] - point0[1])
    p_p0  = (point[0]  - point0[0], point[1]  - point0[1])

    if abs(p_p0[0]) > threshold:
        return 1e9

    if abs(p_p0[1]) > threshold:
        return 1e9

    len_p1_p0 = math.sqrt(p1_p0[0] * p1_p0[0] + p1_p0[1] * p1_p0[1])

    if len_p1_p0 < threshold:
        return math.sqrt(p_p0[0] * p_p0[0] + p_p0[1] * p_p0[1])

    return (_cross(p1_p0, p_p0)) / len_p1_p0


# @numba.njit(fastmath=True)
def _point_to_line_seg_distance(point: Vector2, point0: Vector2, point1: Vector2, threshold: float = 0.01) -> float:
    return abs(_point_to_line_seg_distance_sign(point, point0, point1, threshold))


def point_to_polygon_distance(point: Vec2, polygon: List[Vec2], threshold: float = 0.01) -> Tuple[float, int]:
    m_dist = 1e32
    m_dist_sect_id: int = -1
    curr_dist: float
    n_points: int = len(polygon)

    for i in range(n_points):
        curr_dist = point_to_line_seg_distance(point, polygon[i], polygon[(i + 1) % n_points], threshold)
        if curr_dist < m_dist:
            m_dist = curr_dist
            m_dist_sect_id = i
    return m_dist, m_dist_sect_id


def point_to_polygons_distance(point: Vec2, polygons: List[List[Vec2]], threshold: float = 0.01) -> \
        Tuple[float, int, int]:
    m_dist = 1e32
    m_dist_sect_id: int = -1
    m_dist_poly_id: int = -1
    curr_dist: float
    curr_id: int

    for i, poly in enumerate(polygons):
        curr_dist, curr_id = point_to_polygon_distance(point, poly, threshold)
        if curr_dist < m_dist:
            m_dist = curr_dist
            m_dist_sect_id = curr_id
            m_dist_poly_id = i
    return m_dist, m_dist_sect_id, m_dist_poly_id


def ray_plane_intersect(r_0: Vec3, n: Vec3, origin: Vec3, direction: Vec3) -> Tuple[bool, float]:
    # (r - r_0, n) = 0
    """
    :arg r_0 точка через которую проходит плоскость
    :arg n нормаль к плоскости (единичный вектор)
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return длина луча вдоль его направления до пересечения с плоскостью
    """
    return False, 0.0


def ray_sphere_intersect(r_0: Vec3, r: float, origin: Vec3, direction: Vec3) -> Tuple[bool, float, float]:
    """
    :arg r_0 точка центра сферы
    :arg r радиус сферы
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return длина луча вдоль его направления до первого и воторого пересечений со сферой, если они есть
    """
    return False, 0.0, 0.0


def ray_box_intersect(box_min: Vec3, box_max: Vec3, origin: Vec3, direction: Vec3) -> Tuple[bool, float, float]:
    # r = r_0 + e * t
    """
    :arg box_min минимальная точка бокса
    :arg box_max  максимальная точка бокса
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return длина луча вдоль его направления до первого и воторого пересечений со боксом, если они есть
    """
    return False, 0.0, 0.0


def ray_triangle_intersect(p1: Vec3, p2: Vec3, p3: Vec3, origin: Vec3, direction: Vec3) -> Tuple[bool, float]:
    """
       :arg p1 первая вершина трекугольника
       :arg p2 вторая вершина трекугольника
       :arg p3 третья вершина трекугольника
       :arg origin начало луча
       :arg direction направление луча (единичный вектор)
       :return длина луча вдоль его направления до пересечения с треугольником, если оно
    """
    return False, 0.0
