from core.matrices import Mat4, Mat3
from core.vectors import Vec3, Vec2
from typing import Tuple
# import numba
import math


# @numba.njit(fastmath=True)
def square_equation(a: float, b: float, c: float) -> Tuple[bool, float, float]:
    det: float = b * b - 4.0 * a * c
    if det < 0.0:
        return False, 0.0, 0.0
    det = math.sqrt(det)
    return True, (-b + det) / 2.0 / a, (-b - det) / 2.0 / a


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
    return Mat4(cos_a, 0, -sin_a, 0,
                0, 1, 0, 0,
                sin_a, 0, cos_a, 0,
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
    if math.fabs(rot.m20 + 1) < 1e-6:
        return Vec3(0, -math.pi * 0.5, math.atan2(rot.m01, rot.m02))

    if math.fabs(rot.m20 - 1) < 1e-6:
        return Vec3(0, math.pi * 0.5, math.atan2(-rot.m01, -rot.m02))

    x1 = math.asin(rot.m20)
    x2 = math.pi + x1
    y1 = math.atan2(rot.m21 / math.cos(x1), rot.m22 / math.cos(x1))
    y2 = math.atan2(rot.m21 / math.cos(x2), rot.m22 / math.cos(x2))
    z1 = math.atan2(rot.m10 / math.cos(x1), rot.m00 / math.cos(x1))
    z2 = math.atan2(rot.m10 / math.cos(x2), rot.m00 / math.cos(x2))
    if (abs(x1) + abs(y1) + abs(z1)) <= (abs(x2) + abs(y2) + abs(z2)):
        return Vec3(x1, y1, z1)

    return Vec3(x2, y2, z2)


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


# @numba.njit(fastmath=True)
def clamp(min_: float, max_: float, val: float) -> float:
    """
    :param min_: минимальная граница
    :param max_: максимальная граница
    :param val: значение
    :return: возвращает указанное значение val в границах от min до max
    """
    if val < min_:
        return min_
    if val > max_:
        return max_
    return val


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


# @numba.njit(fastmath=True)
def signum(value) -> float:
    """
    :param value:
    :return: возвращает знач числа
    """
    if value < 0:
        return -1.0
    return 1.0


def perpendicular_2(v: Vec2) -> Vec2:
    """
    :param v:
    :return: возвращает единичный вектор пермендикулярный заданному
    """
    if v.x == 0:
        return Vec2(signum(v.y), 0)
    if v.y == 0:
        return Vec2(0, -signum(v.x))
    sign: float = signum(v.x / v.y)
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


def build_projection_matrix(fov: float = 70, aspect: float = 1, znear: float = 0.01, zfar: float = 1000) -> Mat4:
    """
    :param fov: угол обзора
    :param aspect: соотношение сторон
    :param znear: ближняя плоскость отсечения
    :param zfar: дальняя плоскость отсечения
    :return: матрица перспективной проекции
    """
    projection = Mat4.identity()
    scale = 1.0 / math.tan(fov * 0.5 * math.pi / 180)
    projection.m00 = scale * aspect  # scale the x coordinates of the projected point
    projection.m11 = scale  # scale the y coordinates of the projected point
    projection.m22 = zfar / (znear - zfar)  # used to remap z to [0,1]
    projection.m32 = zfar * znear / (znear - zfar)  # used to remap z [0,1]
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
    return x1 * one_min_t * one_min_t * one_min_t +\
           x2 * 3.0 * one_min_t * one_min_t * t +\
           x3 * 3.0 * one_min_t * t * t + x4 * t * t * t


def _section_bounds_1d(x1: float, x2: float, x3: float, x4: float) -> Tuple[float, float]:
    a: float = -3 * x1 + 9 * x2 - 9 * x3 + 3 * x4
    b: float = 6 * x1 - 12 * x2 + 6 * x3
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
    x_1: float = _bezier_coordinate(clamp(0.0, 1.0, (-b + det) * 0.5 / a), x1, x2, x3, x4)
    x_2: float = _bezier_coordinate(clamp(0.0, 1.0, (-b - det) * 0.5 / a), x1, x2, x3, x4)
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
    dp2: float =  2.0 * u
    dp3: float = -2.0 * psi3 + 2.0

    du: Vec3 = p1 * dp1 * psi1 + p2 * dp1 * psi2 + p3 * dp1 * psi3 + \
               p4 * dp2 * psi1 + p5 * dp2 * psi2 + p6 * dp2 * psi3 + \
               p7 * dp3 * psi1 + p8 * dp3 * psi2 + p9 * dp3 * psi3

    dp1 = -2.0 + v * 2.0
    dp2 =  2.0 * v
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
    phi4: float =  u * u * u
    phi2: float =  3.0 * phi4 - 6.0 * u * u + 3.0 * u
    phi3: float = -3.0 * phi4 + 3.0 * u * u

    psi1: float = (1.0 - v) * (1.0 - v) * (1.0 - v)
    psi4: float =  v * v * v
    psi2: float =  3.0 * psi4 - 6.0 * v * v + 3.0 * v
    psi3: float = -3.0 * psi4 + 3.0 * v * v

    p: Vec3 = p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + p4 * phi1 * psi4 + \
              p5 * phi2 * psi1 + p6 * phi2 * psi2 + p7 * phi2 * psi3 + p8 * phi2 * psi4 + \
              p9 * phi3 * psi1 + p10 * phi3 * psi2 + p11 * phi3 * psi3 + p12 * phi3 * psi4 + \
              p13 * phi4 * psi1 + p14 * phi4 * psi2 + p15 * phi4 * psi3 + p16 * phi4 * psi4

    d4: float =  3.0 * u * u
    d1: float = -3.0 + 6.0 * u - d4
    d2: float =  3.0 * phi4 - 12.0 * u + 3.0
    d3: float = -3.0 * phi4 + 6.0 * u

    dpu: Vec3 = p1  * d1 * psi1 + p2  * d1 * psi2 + p3  * d1 * psi3 + p4  * d1 * psi4 + \
                p5  * d2 * psi1 + p6  * d2 * psi2 + p7  * d2 * psi3 + p8  * d2 * psi4 + \
                p9  * d3 * psi1 + p10 * d3 * psi2 + p11 * d3 * psi3 + p12 * d3 * psi4 + \
                p13 * d4 * psi1 + p14 * d4 * psi2 + p15 * d4 * psi3 + p16 * d4 * psi4

    d4 =  3.0 * v * v
    d1 = -3.0 + 6.0 * v - d4
    d2 =  3.0 * phi4 - 12.0 * v + 3.0
    d3 = -3.0 * phi4 + 6.0 * v

    dpv: Vec3 = p1  * phi1 * d1 + p2  * phi1 * d2 + p3  * phi1 * d3 + p4  * phi1 * d4 + \
                p5  * phi2 * d1 + p6  * phi2 * d2 + p7  * phi2 * d3 + p8  * phi2 * d4 + \
                p9  * phi3 * d1 + p10 * phi3 * d2 + p11 * phi3 * d3 + p12 * phi3 * d4 + \
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


def plane_to_point_dist(r_0: Vec3, n: Vec3, point: Vec3) -> float:
    """
    Уравнение плоскости (r - r_0, n) = 0
    :arg r_0 точка через которую проходит плоскость
    :arg n нормаль к плоскости (единичный вектор)
    :arg point точка для которой ищем расстояние
    :return расстояние между точкой и плоскостью
    """
    return Vec3.dot(point - r_0, n)


def ray_plane_intersect(r_0: Vec3, n: Vec3, origin: Vec3, direction: Vec3) -> Tuple[bool, float]:
    # (r - r_0, n) = 0
    """
    :arg r_0 точка через которую проходит плоскость
    :arg n нормаль к плоскости (единичный вектор)
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return длина луча вдоль его направления до пересечения с плоскостью
    """
    pass


def ray_sphere_intersect(r_0: Vec3, r: float, origin: Vec3, direction: Vec3) -> Tuple[bool, float, float]:
    """
    :arg r_0 точка центра сферы
    :arg r радиус сферы
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return длина луча вдоль его направления до первого и воторого пересечений со сферой, если они есть
    """
    pass


def ray_box_intersect(box_min: Vec3, box_max: Vec3, origin: Vec3, direction: Vec3) -> Tuple[bool, float, float]:
    # r = r_0 + e * t
    """
    :arg box_min минимальная точка бокса
    :arg box_max  максимальная точка бокса
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return длина луча вдоль его направления до первого и воторого пересечений со боксом, если они есть
    """
    pass


def ray_triangle_intersect(p1: Vec3, p2: Vec3, p3: Vec3, origin: Vec3, direction: Vec3) -> Tuple[bool, float]:
    """
       :arg p1 первая вершина трекугольника
       :arg p2 вторая вершина трекугольника
       :arg p3 третья вершина трекугольника
       :arg origin начало луча
       :arg direction направление луча (единичный вектор)
       :return длина луча вдоль его направления до пересечения с треугольником, если оно
    """
    pass
