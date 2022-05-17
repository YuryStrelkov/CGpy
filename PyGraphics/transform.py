import numpy as np
import mathUtils
from mathUtils import Vec3, Vec2, Mat4, Mat3


# row major 3D transform
class Transform(object):
    def __init__(self):
        self.transformM = Mat4(1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0)
        self.eulerAngles = Vec3(0.0, 0.0, 0.0)

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
        return np.sqrt(x * x + y * y + z * z)

    # масштаб по Y
    @property
    def sy(self) -> float:
        x = self.transformM.m01
        y = self.transformM.m11
        z = self.transformM.m21
        return np.sqrt(x * x + y * y + z * z)
        # масштаб по Z

    @property
    def sz(self) -> float:
        x = self.transformM.m02
        y = self.transformM.m12
        z = self.transformM.m22
        return np.sqrt(x * x + y * y + z * z)
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
        i = mathUtils.mat_mul_4(i, mathUtils.rotate_y(xyz.y))
        i = mathUtils.mat_mul_4(i, mathUtils.rotate_z(xyz.z))

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


class Transform2(object):
    def __init__(self):
        self.transformM = Mat3(1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0)
        self.zAngle = 0.0

    @property
    def front(self) -> Vec2:
        return Vec2(self.transformM.m00, self.transformM.m10).normalize()

    @property
    def up(self) -> Vec2:
        return Vec2(self.transformM.m01, self.transformM.m11).normalize()

    @property
    def scale(self) -> Vec2:
        return Vec2(self.sx, self.sy)

    # масштаб по Х
    @property
    def sx(self) -> float:
        x = self.transformM.m00
        y = self.transformM.m10
        return np.sqrt(x * x + y * y)

    # масштаб по Y
    @property
    def sy(self) -> float:
        x = self.transformM.m01
        y = self.transformM.m11
        return np.sqrt(x * x + y * y)
        # установить масштаб по Х

    @sx.setter
    def sx(self, s_x: float):
        if s_x == 0:
            return
        scl = self.sx
        self.transformM.m00 /= scl / s_x
        self.transformM.m10 /= scl / s_x

    # установить масштаб по Y
    @sy.setter
    def sy(self, s_y: float):
        if s_y == 0:
            return
        scl = self.sy
        self.transformM.m01 /= scl / s_y
        self.transformM.m11 /= scl / s_y

    @scale.setter
    def scale(self, sxy: Vec2):
        self.sx = sxy.x
        self.sy = sxy.y

    @property
    def x(self) -> float:
        return self.transformM.m02

    @property
    def y(self) -> float:
        return self.transformM.m12

    @property
    def origin(self) -> Vec2:
        return Vec2(self.x, self.y)

    @x.setter
    def x(self, x: float):
        self.transformM.m02 = x

    @y.setter
    def y(self, y: float):
        self.transformM.m12 = y

    @origin.setter
    def origin(self, xy: Vec2):
        self.x = xy.x
        self.y = xy.y

    @property
    def az(self) -> float:
        return self.zAngle

    @az.setter
    def az(self, angle: float):
        if self.zAngle == angle:
            return
        self.zAngle = angle
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rz = Mat3(cos_a, -sin_a, 0,
                  sin_a, cos_a, 0,
                  0, 0, 1)
        scl = self.scale
        orig = self.origin
        self.transformM = rz
        self.scale = scl
        self.origin = orig

    # переводит вектор в собственное пространство координат
    def transform_vect(self, vec: Vec2, w) -> Vec2:
        if w == 0:
            return Vec2(self.transformM.m00 * vec.x + self.transformM.m01 * vec.y,
                        self.transformM.m10 * vec.x + self.transformM.m11 * vec.y)

        return Vec2(self.transformM.m00 * vec.x + self.transformM.m01 * vec.y + self.transformM.m02,
                    self.transformM.m10 * vec.x + self.transformM.m11 * vec.y + self.transformM.m12)

    # не переводит вектор в собственное пространство координат =)
    def inv_transform_vect(self, vec: Vec2, w) -> Vec2:
        scl: Vec2 = self.scale
        if w == 0:
            return Vec2((self.transformM.m00 * vec.x + self.transformM.m10 * vec.y) / scl.x / scl.x,
                        (self.transformM.m01 * vec.x + self.transformM.m11 * vec.y) / scl.y / scl.y)

        vec_ = Vec2(vec.x - self.x, vec.y - self.y)
        return Vec2((self.transformM.m00 * vec.x + self.transformM.m10 * vec.y) / scl.x / scl.x,
                    (self.transformM.m01 * vec.x + self.transformM.m11 * vec.y) / scl.y / scl.y)
