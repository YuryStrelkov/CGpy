from vmath.matrices import Mat4, Mat3
from vmath.vectors import Vec3, Vec2
import vmath.vectors as vectors
import vmath.matrices
import numpy as np
import math


def rotate_x(angle: float) -> Mat4:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Mat4(1, 0, 0, 0,
                0, cos_a, -sin_a, 0,
                0, sin_a, cos_a, 0,
                0, 0, 0, 1)


def rotate_y(angle: float) -> Mat4:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Mat4(cos_a, 0, -sin_a, 0,
                0, 1, 0, 0,
                sin_a, 0, cos_a, 0,
                0, 0, 0, 1)


def rotate_z(angle: float) -> Mat4:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Mat4(cos_a, -sin_a, 0, 0,
                sin_a, cos_a, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1)


def rotate(angle_x: float, angle_y: float, angle_z: float) -> Mat4:
    return rotate_x(angle_x) * rotate_y(angle_y) * rotate_z(angle_z)


def deg_to_rad(deg: float) -> float: return deg / 180.0 * math.pi


def rad_to_deg(deg: float) -> float: return deg / math.pi * 180.0


def rot_m_to_euler_angles(rot: Mat4) -> Vec3:
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


def signum(value) -> float:
    if value < 0:
        return -1.0
    return 1.0


def perpendicular_2(v: Vec2) -> Vec2:
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
    s: float = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    g: float = math.copysign(s, v.z)  # note s instead of 1
    h: float = v.z + g
    return Vec3(g * h - v.x * v.x, -v.x * v.y, -v.x * h)


def form_transform(rotation, translation):
    """
        Makes a transformation matrix from the given rotation matrix and translation vector
        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector
        Returns
        -------
        T (ndarray): The transformation matrix
        """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def build_projection_matrix(fov: float = 70, aspect: float = 1, znear: float = 0.01, zfar: float = 1000) -> Mat4:
    projection = vmath.matrices.identity_4()
    scale = 1.0 / math.tan(fov * 0.5 * math.pi / 180)
    projection.m00 = scale * aspect  # scale the x coordinates of the projected point
    projection.m11 = scale  # scale the y coordinates of the projected point
    projection.m22 = zfar / (znear - zfar)  # used to remap z to [0,1]
    projection.m32 = zfar * znear / (znear - zfar)  # used to remap z [0,1]
    projection.m23 = -1  # set w = -z
    projection.m33 = 0
    return projection


def build_orthogonal_basis(right: Vec3, up: Vec3, front: Vec3, main_axes=3) -> Mat4:
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


