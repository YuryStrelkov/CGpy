from core.matrices import Mat4, Mat3
from core.vectors import Vec3, Vec2
import core.matrices as matrices
import math


def square_equation(a: float, b: float, c: float) -> (bool, float, float):
    det: float = b * b - 4.0 * a * c
    if det < 0.0:
        return False, 0.0, 0.0
    det = math.sqrt(det)
    return True, (-b + det) / 2.0 / a, (-b - det) / 2.0 / a


def cube_equation(a: float, b: float, c: float, d: float) -> (bool, float, float, float):
    if a == 0 and b == 0 and c == 0 and d == 0:
        print('Ошибка')
        exit()
    p = (3.0 * a * c - b ** 2.0) / (3.0 * a ** 2.0)
    q = (2.0 * b ** 3.0 - 9.0 * a * b * c + 27.0 * a ** 2.0 * d) / (27.0 * a ** 3.0)
    det = (p / 3.0) ** 3.0 + (q / 2.0) ** 2.0
    if det < 0:
        return False, 0.0, 0.0, 0.0
    elif det == 0:
        return True, 2.0 * (-q / 2.0) ** (1.0 / 3.0) - b / (3.0 * a),\
                           (-q / 2.0) ** (-1.0 / 3.0) - b / (3.0 * a),\
                           (-q / 2.0) ** (-1.0 / 3.0) - b / (3.0 * a)
    elif det > 0:
        alfa = (-q / 2 + det ** 0.5) ** (1 / 3)
        beta = -abs((-q / 2 - det ** 0.5) ** (1 / 3))
        y1 = alfa + beta
        y2 = complex(-((alfa + beta) / 2), (alfa - beta) / 2 * 3 ** 0.5)
        y3 = complex(-((alfa + beta) / 2), -(alfa - beta) / 2 * 3 ** 0.5)
        x1 = y1 - b / (3 * a)
        x1 = round(x1, 5)
        x2 = y2 - b / (3 * a)
        x2 = round(x2.real, 5) + round(x2.imag, 5) * 1j
        x3 = y3 - b / (3 * a)
        x3 = round(x3.real, 5) + round(x3.imag, 5) * 1j
        print('\nalfa =', alfa, '\nbeta =', beta, '\n\nx1 =', x1, '\nx2 =', x2, '\nx3 =', x3)


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


def deg_to_rad(deg: float) -> float:
    """
    :param deg: угол в градусах
    :return: угол в радианах
    """
    return deg / 180.0 * math.pi


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


def lerp_vec_2(a: Vec2, b: Vec2, t: float) -> Vec2:
    """
    :param a: вектор начала
    :param b: вектор конца
    :param t: параметр в пределах от 0 до 1
    :return: возвращает линейную интерполяцию между двухмерными векторами a и b
    """
    return Vec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)


def lerp_vec_3(a: Vec3, b: Vec3, t: float) -> Vec3:
    """
    :param a: вектор начала
    :param b: вектор конца
    :param t: параметр в пределах от 0 до 1
    :return: возвращает линейную интерполяцию между трёхмерными векторами a и b
    """
    return Vec3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t)


def lerp_mat_3(a: Mat3, b: Mat3, t: float) -> Mat3:
    """
    :param a: матрица начала
    :param b: матрица конца
    :param t: параметр в пределах от 0 до 1
    :return: возвращает линейную интерполяцию между трёхмернми матрицами a и b
    """
    return Mat3(a.m00 + (b.m00 - a.m00) * t, a.m01 + (b.m01 - a.m01) * t, a.m02 + (b.m02 - a.m02) * t,
                a.m10 + (b.m10 - a.m10) * t, a.m11 + (b.m11 - a.m11) * t, a.m12 + (b.m12 - a.m12) * t,
                a.m20 + (b.m20 - a.m20) * t, a.m21 + (b.m21 - a.m21) * t, a.m22 + (b.m22 - a.m22) * t)


def lerp_mat_4(a: Mat4, b: Mat4, t: float) -> Mat4:
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
    projection = matrices.identity_4()
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
    d: float = 3 * t * t
    a: float = -3 + 6 * t - d
    b: float = 3 - 12 * t + 3 * d
    c: float = 6 * t - 3 * d
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
    d: float = 3 * t * t
    a: float = -3 + 6 * t - d
    b: float = 3 - 12 * t + 3 * d
    c: float = 6 * t - 3 * d
    return Vec3(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                p1.y * a + p2.y * b + p3.y * c + p4.y * d,
                p1.z * a + p2.z * b + p3.z * c + p4.z * d)


def quadratic_bezier_patch(p1: Vec3, p2: Vec3, p3: Vec3,
                           p4: Vec3, p5: Vec3, p6: Vec3,
                           p7: Vec3, p8: Vec3, p9: Vec3, u: float, v: float) -> (Vec3, Vec3):
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
    phi1: float = (1 - u) * (1 - u)
    phi3: float = u * u
    phi2: float = -2 * phi3 + 2 * u

    psi1: float = (1 - v) * (1 - v)
    psi3: float = v * v
    psi2: float = -2 * psi3 + 2 * v

    p: Vec3 = p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + \
              p4 * phi2 * psi1 + p5 * phi2 * psi2 + p6 * phi2 * psi3 + \
              p7 * phi3 * psi1 + p8 * phi3 * psi2 + p9 * phi3 * psi3

    dp1: float = -2 + u * 2
    dp2: float =  2 * u
    dp3: float = -2 * psi3 + 2

    du: Vec3 = p1 * dp1 * psi1 + p2 * dp1 * psi2 + p3 * dp1 * psi3 + \
               p4 * dp2 * psi1 + p5 * dp2 * psi2 + p6 * dp2 * psi3 + \
               p7 * dp3 * psi1 + p8 * dp3 * psi2 + p9 * dp3 * psi3

    dp1 = -2 + v * 2
    dp2 =  2 * v
    dp3 = -2 * psi3 + 2

    dv: Vec3 = p1 * phi1 * dp1 + p2 * phi1 * dp2 + p3 * phi1 * dp3 + \
               p4 * phi2 * dp1 + p5 * phi2 * dp2 + p6 * phi2 * dp3 + \
               p7 * phi3 * dp1 + p8 * phi3 * dp2 + p9 * phi3 * dp3

    return [p, Vec3.cross(dv, du).normalize()]


def cubic_bezier_patch(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3,
                       p5: Vec3, p6: Vec3, p7: Vec3, p8: Vec3,
                       p9: Vec3, p10: Vec3, p11: Vec3, p12: Vec3,
                       p13: Vec3, p14: Vec3, p15: Vec3, p16: Vec3, u: float, v: float) -> (Vec3, Vec3):
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

    phi1: float = (1 - u) * (1 - u) * (1 - u)
    phi4: float =  u * u * u
    phi2: float =  3 * phi4 - 6 * u * u + 3 * u
    phi3: float = -3 * phi4 + 3 * u * u

    psi1: float = (1 - v) * (1 - v) * (1 - v)
    psi4: float =  v * v * v
    psi2: float =  3 * psi4 - 6 * v * v + 3 * v
    psi3: float = -3 * psi4 + 3 * v * v

    p: Vec3 = p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + p4 * phi1 * psi4 + \
              p5 * phi2 * psi1 + p6 * phi2 * psi2 + p7 * phi2 * psi3 + p8 * phi2 * psi4 + \
              p9 * phi3 * psi1 + p10 * phi3 * psi2 + p11 * phi3 * psi3 + p12 * phi3 * psi4 + \
              p13 * phi4 * psi1 + p14 * phi4 * psi2 + p15 * phi4 * psi3 + p16 * phi4 * psi4

    d4: float =  3 * u * u
    d1: float = -3 + 6 * u - d4
    d2: float =  3 * phi4 - 12 * u + 3
    d3: float = -3 * phi4 + 6 * u

    dpu: Vec3 = p1  * d1 * psi1 + p2  * d1 * psi2 + p3  * d1 * psi3 + p4  * d1 * psi4 + \
                p5  * d2 * psi1 + p6  * d2 * psi2 + p7  * d2 * psi3 + p8  * d2 * psi4 + \
                p9  * d3 * psi1 + p10 * d3 * psi2 + p11 * d3 * psi3 + p12 * d3 * psi4 + \
                p13 * d4 * psi1 + p14 * d4 * psi2 + p15 * d4 * psi3 + p16 * d4 * psi4

    d4 =  3 * v * v
    d1 = -3 + 6 * v - d4
    d2 =  3 * phi4 - 12 * v + 3
    d3 = -3 * phi4 + 6 * v

    dpv: Vec3 = p1  * phi1 * d1 + p2  * phi1 * d2 + p3  * phi1 * d3 + p4  * phi1 * d4 + \
                p5  * phi2 * d1 + p6  * phi2 * d2 + p7  * phi2 * d3 + p8  * phi2 * d4 + \
                p9  * phi3 * d1 + p10 * phi3 * d2 + p11 * phi3 * d3 + p12 * phi3 * d4 + \
                p13 * phi4 * d1 + p14 * phi4 * d2 + p15 * phi4 * d3 + p16 * phi4 * d4

    return [p, Vec3.cross(dpv, dpu).normalize()]


def point_to_line_dist(point: Vec3, origin: Vec3, direction: Vec3) -> float:
    """
    :arg point точка до которой ищем расстоняие
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return расстояние между точкой и прямой
    """
    pass


def line_to_line_dist(origin_1: Vec3, direction_1: Vec3, origin_2: Vec3, direction_2: Vec3) -> float:
    """
    :arg origin_1 начало первого луча
    :arg direction_1 направление первого луча (единичный вектор)
    :arg origin_2 начало второго луча
    :arg direction_2 направление второго луча (единичный вектор)
    :return расстояние между первой и второй прямой
    """
    pass


def plane_to_point_dist(r_0: Vec3, n: Vec3, point: Vec3) -> float:
    # (r - r_0, n) = 0
    """
    :arg r_0 точка через которую проходит плоскость
    :arg n нормаль к плоскости (единичный вектор)
    :arg point точка для которой ищем расстояние
    :return расстояние между точкой и плоскостью
    """
    pass


def ray_plane_intersect(r_0: Vec3, n: Vec3, origin: Vec3, direction: Vec3) -> (bool, float):
    # (r - r_0, n) = 0
    """
    :arg r_0 точка через которую проходит плоскость
    :arg n нормаль к плоскости (единичный вектор)
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return длина луча вдоль его направления до пересечения с плоскостью
    """
    pass


def ray_sphere_intersect(r_0: Vec3, r: float, origin: Vec3, direction: Vec3) -> (bool, float, float):
    """
    :arg r_0 точка центра сферы
    :arg r радиус сферы
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return длина луча вдоль его направления до первого и воторого пересечений со сферой, если они есть
    """
    pass


def ray_box_intersect(box_min: Vec3, box_max: Vec3, origin: Vec3, direction: Vec3) -> (bool, float, float):
    # r = r_0 + e * t
    """
    :arg box_min минимальная точка бокса
    :arg box_max  максимальная точка бокса
    :arg origin начало луча
    :arg direction направление луча (единичный вектор)
    :return длина луча вдоль его направления до первого и воторого пересечений со боксом, если они есть
    """
    pass


def ray_triangle_intersect(p1: Vec3, p2: Vec3, p3: Vec3, origin: Vec3, direction: Vec3) -> (bool, float):
    """
       :arg p1 первая вершина трекугольника
       :arg p2 вторая вершина трекугольника
       :arg p3 третья вершина трекугольника
       :arg origin начало луча
       :arg direction направление луча (единичный вектор)
       :return длина луча вдоль его направления до пересечения с треугольником, если оно
    """
    pass
