import math

import numpy as np
from vmath import mathUtils, matrices
from vmath.matrices import  Mat4
from vmath.vectors import  Vec3, Vec2


class Transform(object):

    def __init__(self):
        self.transformM = Mat4(1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0)
        self.eulerAngles = Vec3(0.0, 0.0, 0.0)

    def __repr__(self) -> str:
        res: str = "<Transform \n"
        res += f"origin   :{self.origin}\n"
        res += f"scale    :{self.scale}\n"
        res += f"rotate   :{self.eulerAngles}\n"
        res += f"t-matrix :\n{self.transformM}\n"
        res += ">"
        return res

    def __str__(self) -> str:
        res: str = "Transform \n"
        res += f"origin   :{self.origin}\n"
        res += f"scale    :{self.scale}\n"
        res += f"rotate   :{self.eulerAngles}\n"
        res += f"t-matrix :\n{self.transformM}\n"
        return res

    def __eq__(self, other) -> bool:
        if not(type(other) is Transform):
            return False
        if not(self.transformM == other.transformM):
            return False
        return True

    def __hash__(self) -> int:
        return hash(self.transformM)

    @property
    def front(self) -> Vec3:
        return Vec3(self.transformM.m02,
                    self.transformM.m12,
                    self.transformM.m22).normalize()

    @property
    def up(self) -> Vec3:
        return Vec3(self.transformM.m01,
                    self.transformM.m11,
                    self.transformM.m21).normalize()

    @property
    def right(self) -> Vec3:
        return Vec3(self.transformM.m00,
                    self.transformM.m10,
                    self.transformM.m20).normalize()

    # масштаб по Х
    @property
    def sx(self) -> float:
        x = self.transformM.m00
        y = self.transformM.m10
        z = self.transformM.m20
        return math.sqrt(x * x + y * y + z * z)

    # масштаб по Y
    @property
    def sy(self) -> float:
        x = self.transformM.m01
        y = self.transformM.m11
        z = self.transformM.m21
        return math.sqrt(x * x + y * y + z * z)
        # масштаб по Z

    @property
    def sz(self) -> float:
        x = self.transformM.m02
        y = self.transformM.m12
        z = self.transformM.m22
        return math.sqrt(x * x + y * y + z * z)
        # установить масштаб по Х

    @sx.setter
    def sx(self, s_x: float):
        if s_x == 0:
            return
        scl = self.sx
        self.transformM.m00 /= scl / s_x
        self.transformM.m10 /= scl / s_x
        self.transformM.m20 /= scl / s_x

    # установить масштаб по Y
    @sy.setter
    def sy(self, s_y: float):
        if s_y == 0:
            return
        scl = self.sy
        self.transformM.m01 /= scl / s_y
        self.transformM.m11 /= scl / s_y
        self.transformM.m21 /= scl / s_y

    # установить масштаб по Z
    @sz.setter
    def sz(self, s_z: float):
        if s_z == 0:
            return
        scl = self.sz
        self.transformM.m02 /= scl / s_z
        self.transformM.m12 /= scl / s_z
        self.transformM.m22 /= scl / s_z

    @property
    def scale(self) -> Vec3:
        return Vec3(self.sx, self.sy, self.sz)

    @scale.setter
    def scale(self, xyz: Vec3):
        self.sx = xyz.x
        self.sy = xyz.y
        self.sz = xyz.z

    @property
    def x(self) -> float:
        return self.transformM.m03

    @property
    def y(self) -> float:
        return self.transformM.m13

    @property
    def z(self) -> float:
        return self.transformM.m23

    @x.setter
    def x(self, x: float):
        self.transformM.m03 = x

    @y.setter
    def y(self, y: float):
        self.transformM.m13 = y

    @z.setter
    def z(self, z: float):
        self.transformM.m23 = z

    @property
    def origin(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)

    @origin.setter
    def origin(self, xyz: Vec3):
        self.x = xyz.x
        self.y = xyz.y
        self.z = xyz.z

    @property
    def angles(self) -> Vec3:
        return self.eulerAngles

    @angles.setter
    def angles(self, xyz: Vec3):
        if self.eulerAngles.x == xyz.x and self.eulerAngles.y == xyz.y and self.eulerAngles.z == xyz.z:
            return
        self.eulerAngles.x = xyz.x
        self.eulerAngles.y = xyz.y
        self.eulerAngles.z = xyz.z

        i = mathUtils.rotate_x(xyz.x)
        i = i * mathUtils.rotate_y(xyz.y)
        i = i * mathUtils.rotate_z(xyz.y)

        scl = self.scale
        orig = self.origin
        self.transformM = i
        self.scale = scl
        self.origin = orig

    @property
    def ax(self) -> float:
        return self.eulerAngles.x

    @property
    def ay(self) -> float:
        return self.eulerAngles.y

    @property
    def az(self) -> float:
        return self.eulerAngles.z

    @ax.setter
    def ax(self, x: float):
        self.angles = Vec3(mathUtils.deg_to_rad(x), self.eulerAngles.y, self.eulerAngles.z)

    @ay.setter
    def ay(self, y: float):
        self.angles = Vec3(self.eulerAngles.x, mathUtils.deg_to_rad(y), self.eulerAngles.z)

    @az.setter
    def az(self, z: float):
        self.angles = Vec3(self.eulerAngles.x, self.eulerAngles.y, mathUtils.deg_to_rad(z))

    def rotation_mat(self) -> Mat4:
        scl = self.scale
        return Mat4(self.transformM.m00 / scl.x, self.transformM.m01 / scl.y, self.transformM.m02 / scl.z, 0,
                    self.transformM.m10 / scl.x, self.transformM.m11 / scl.y, self.transformM.m12 / scl.z, 0,
                    self.transformM.m20 / scl.x, self.transformM.m21 / scl.y, self.transformM.m22 / scl.z, 0,
                    0, 0, 0, 1)

    def look_at(self, target: Vec3, eye: Vec3, up: Vec3 = Vec3(0, 1, 0)):
        self.transformM = mathUtils.look_at(target, eye, up)
        self.eulerAngles = mathUtils.rot_m_to_euler_angles(self.rotation_mat())

    # переводит вектор в собственное пространство координат
    def transform_vect(self, vec: Vec3, w) -> Vec3:
        if w == 0:
            return Vec3(self.transformM.m00 * vec.x + self.transformM.m01 * vec.y + self.transformM.m02 * vec.z,
                        self.transformM.m10 * vec.x + self.transformM.m11 * vec.y + self.transformM.m12 * vec.z,
                        self.transformM.m20 * vec.x + self.transformM.m21 * vec.y + self.transformM.m22 * vec.z)

        return Vec3(
            self.transformM.m00 * vec.x + self.transformM.m01 * vec.y + self.transformM.m02 * vec.z + self.transformM.m03,
            self.transformM.m10 * vec.x + self.transformM.m11 * vec.y + self.transformM.m12 * vec.z + self.transformM.m13,
            self.transformM.m20 * vec.x + self.transformM.m21 * vec.y + self.transformM.m22 * vec.z + self.transformM.m23)

    # не переводит вектор в собственное пространство координат =)
    def inv_transform_vect(self, vec: Vec3, w) -> Vec3:
        scl: Vec3 = self.scale
        if w == 0:
            return Vec3((self.transformM.m00 * vec.x + self.transformM.m10 * vec.y + self.transformM.m20 * vec.z) / scl.x / scl.x,
                        (self.transformM.m01 * vec.x + self.transformM.m11 * vec.y + self.transformM.m21 * vec.z) / scl.y / scl.y,
                        (self.transformM.m02 * vec.x + self.transformM.m12 * vec.y + self.transformM.m22 * vec.z) / scl.z / scl.z)

        vec_ = Vec3(vec.x - self.x, vec.y - self.y, vec.z - self.z)
        return Vec3((self.transformM.m00 * vec_.x + self.transformM.m10 * vec_.y + self.transformM.m20 * vec_.z) / scl.x / scl.x,
                    (self.transformM.m01 * vec_.x + self.transformM.m11 * vec_.y + self.transformM.m21 * vec_.z) / scl.y / scl.y,
                    (self.transformM.m02 * vec_.x + self.transformM.m12 * vec_.y + self.transformM.m22 * vec_.z) / scl.z / scl.z)

    # row major 2D transform
