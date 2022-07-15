import vmath.vectors as vectors
from vmath.matrices import Mat4
from vmath.vectors import Vec3
from vmath import mathUtils
import math


class Transform(object):

    def __init__(self):
        self.__m_transform = Mat4(1.0, 0.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0, 0.0,
                                  0.0, 0.0, 0.0, 1.0)

    def __repr__(self) -> str:
        res: str = "<Transform \n"
        res += f"origin   :{self.origin}\n"
        res += f"scale    :{self.scale}\n"
        res += f"rotate   :{self.angles / math.pi * 180}\n"
        res += f"t-matrix :\n{self.__m_transform}\n"
        res += ">"
        return res

    def __str__(self) -> str:
        res: str = f"Transform : 0x{id(self)} \n"
        res += f"origin    : {self.origin}\n"
        res += f"scale     : {self.scale}\n"
        res += f"rotate    : {self.angles / math.pi * 180}\n"
        res += f"t-matrix  :\n{self.__m_transform}\n"
        return res

    def __eq__(self, other) -> bool:
        if not(type(other) is Transform):
            return False
        if not(self.__m_transform == other.__m_transform):
            return False
        return True

    def __hash__(self) -> int:
        return hash(self.__m_transform)

    def __build_basis(self, ex: Vec3, ey: Vec3, ez: Vec3) -> None:
        self.__m_transform.m00 = ex.x
        self.__m_transform.m10 = ex.y
        self.__m_transform.m20 = ex.z

        self.__m_transform.m01 = ey.x
        self.__m_transform.m11 = ey.y
        self.__m_transform.m21 = ey.z

        self.__m_transform.m02 = ez.x
        self.__m_transform.m12 = ez.y
        self.__m_transform.m22 = ez.z

        # self.eulerAngles = mathUtils.rot_m_to_euler_angles(self.rotation_mat())

    @property
    def transform_matrix(self) -> Mat4:
        return self.__m_transform

    @transform_matrix.setter
    def transform_matrix(self, t: Mat4) -> None:
        self.__m_transform.m00 = t.m00
        self.__m_transform.m10 = t.m10
        self.__m_transform.m20 = t.m20

        self.__m_transform.m01 = t.m01
        self.__m_transform.m11 = t.m11
        self.__m_transform.m21 = t.m21

        self.__m_transform.m02 = t.m02
        self.__m_transform.m12 = t.m12
        self.__m_transform.m22 = t.m22

        self.__m_transform.m03 = t.m03
        self.__m_transform.m13 = t.m13
        self.__m_transform.m23 = t.m23

    @property
    def front(self) -> Vec3:
        return Vec3(self.__m_transform.m02,
                    self.__m_transform.m12,
                    self.__m_transform.m22).normalize()

    @front.setter
    def front(self, front_: Vec3) -> None:
        length_ = front_.magnitude
        if length_ < 1e-9:
            raise ArithmeticError("Error transform front set")
        front_dir_ = front_ / length_
        right_ = vectors. cross(front_dir_, Vec3(0, 1, 0)).normalized()
        up_ = vectors. cross(right_, front_dir_).normalized()
        self.__build_basis(right_ * self.sx, up_ * self.sy, front_)

    @property
    def up(self) -> Vec3:
        return Vec3(self.__m_transform.m01,
                    self.__m_transform.m11,
                    self.__m_transform.m21).normalize()

    @up.setter
    def up(self, up_: Vec3) -> None:
        length_ = up_.magnitude
        if length_ < 1e-9:
            raise ArithmeticError("Error transform up set")
        up_dir_ = up_ / length_
        front_ = vectors.cross(up_dir_, Vec3(1, 0, 0)).normalized()
        right_ = vectors.cross(up_dir_, front_).normalized()
        self.__build_basis(right_ * self.sx, up_, front_ * self.sz)

    @property
    def right(self) -> Vec3:
        return Vec3(self.__m_transform.m00,
                    self.__m_transform.m10,
                    self.__m_transform.m20).normalize()

    @right.setter
    def right(self, right_: Vec3) -> None:
        length_ = right_.magnitude
        if length_ < 1e-9:
            raise ArithmeticError("Error transform up set")
        right_dir_ = right_ / length_
        front_ = vectors.cross(right_dir_, Vec3(0, 1, 0)).normalized()
        up_ = vectors.cross(front_, right_dir_).normalized()
        self.__build_basis(right_, up_ * self.sy, front_ * self.sz)

    # масштаб по Х
    @property
    def sx(self) -> float:
        x = self.__m_transform.m00
        y = self.__m_transform.m10
        z = self.__m_transform.m20
        return math.sqrt(x * x + y * y + z * z)

    # масштаб по Y
    @property
    def sy(self) -> float:
        x = self.__m_transform.m01
        y = self.__m_transform.m11
        z = self.__m_transform.m21
        return math.sqrt(x * x + y * y + z * z)
        # масштаб по Z

    @property
    def sz(self) -> float:
        x = self.__m_transform.m02
        y = self.__m_transform.m12
        z = self.__m_transform.m22
        return math.sqrt(x * x + y * y + z * z)
        # установить масштаб по Х

    @sx.setter
    def sx(self, s_x: float):
        if s_x == 0:
            return
        scl = self.sx
        self.__m_transform.m00 *= s_x / scl
        self.__m_transform.m10 *= s_x / scl
        self.__m_transform.m20 *= s_x / scl

    # установить масштаб по Y
    @sy.setter
    def sy(self, s_y: float):
        if s_y == 0:
            return
        scl = self.sy
        self.__m_transform.m01 *= s_y / scl
        self.__m_transform.m11 *= s_y / scl
        self.__m_transform.m21 *= s_y / scl

    # установить масштаб по Z
    @sz.setter
    def sz(self, s_z: float):
        if s_z == 0:
            return
        scl = self.sz
        self.__m_transform.m02 *= s_z / scl
        self.__m_transform.m12 *= s_z / scl
        self.__m_transform.m22 *= s_z / scl

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
        return self.__m_transform.m03

    @property
    def y(self) -> float:
        return self.__m_transform.m13

    @property
    def z(self) -> float:
        return self.__m_transform.m23

    @x.setter
    def x(self, x: float):
        self.__m_transform.m03 = x

    @y.setter
    def y(self, y: float):
        self.__m_transform.m13 = y

    @z.setter
    def z(self, z: float):
        self.__m_transform.m23 = z

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
        return mathUtils.rot_m_to_euler_angles(self.rotation_mat())

    @angles.setter
    def angles(self, xyz: Vec3):
        i = mathUtils.rotate_x(xyz.x)
        i = mathUtils.rotate_y(xyz.y) * i
        i = mathUtils.rotate_z(xyz.z) * i
        scl = self.scale
        orig = self.origin
        self.__m_transform = i
        self.scale = scl
        self.origin = orig

    @property
    def ax(self) -> float:
        return mathUtils.rot_m_to_euler_angles(self.rotation_mat()).x

    @property
    def ay(self) -> float:
        return mathUtils.rot_m_to_euler_angles(self.rotation_mat()).y

    @property
    def az(self) -> float:
        return mathUtils.rot_m_to_euler_angles(self.rotation_mat()).z

    @ax.setter
    def ax(self, x: float):
        _angles = self.angles
        self.angles = Vec3(mathUtils.deg_to_rad(x), _angles.y, _angles.z)

    @ay.setter
    def ay(self, y: float):
        _angles = self.angles
        self.angles = Vec3(_angles.x, mathUtils.deg_to_rad(y), _angles.z)

    @az.setter
    def az(self, z: float):
        _angles = self.angles
        self.angles = Vec3(_angles.x, _angles.y, mathUtils.deg_to_rad(z))

    def rotation_mat(self) -> Mat4:
        scl = self.scale
        return Mat4(self.__m_transform.m00 / scl.x, self.__m_transform.m01 / scl.y, self.__m_transform.m02 / scl.z, 0,
                    self.__m_transform.m10 / scl.x, self.__m_transform.m11 / scl.y, self.__m_transform.m12 / scl.z, 0,
                    self.__m_transform.m20 / scl.x, self.__m_transform.m21 / scl.y, self.__m_transform.m22 / scl.z, 0,
                    0, 0, 0, 1)

    def look_at(self, target: Vec3, eye: Vec3, up: Vec3 = Vec3(0, 1, 0)):
        self.__m_transform = mathUtils.look_at(target, eye, up)

    # переводит вектор в собственное пространство координат
    def transform_vect(self, vec: Vec3, w) -> Vec3:
        if w == 0:
            return Vec3(self.__m_transform.m00 * vec.x + self.__m_transform.m01 * vec.y + self.__m_transform.m02 * vec.z,
                        self.__m_transform.m10 * vec.x + self.__m_transform.m11 * vec.y + self.__m_transform.m12 * vec.z,
                        self.__m_transform.m20 * vec.x + self.__m_transform.m21 * vec.y + self.__m_transform.m22 * vec.z)

        return Vec3(
            self.__m_transform.m00 * vec.x + self.__m_transform.m01 * vec.y + self.__m_transform.m02 * vec.z + self.__m_transform.m03,
            self.__m_transform.m10 * vec.x + self.__m_transform.m11 * vec.y + self.__m_transform.m12 * vec.z + self.__m_transform.m13,
            self.__m_transform.m20 * vec.x + self.__m_transform.m21 * vec.y + self.__m_transform.m22 * vec.z + self.__m_transform.m23)

    # не переводит вектор в собственное пространство координат =)
    @property
    def inv_transform_matrix(self) -> Mat4:
        scl: Vec3 = self.scale
        scl *= scl
        return Mat4(self.__m_transform.m00 * scl.x, self.__m_transform.m10 * scl.x, self.__m_transform.m20 * scl.x, -self.__m_transform.m03,
                    self.__m_transform.m01 * scl.y, self.__m_transform.m11 * scl.y, self.__m_transform.m21 * scl.y, -self.__m_transform.m13,
                    self.__m_transform.m02 * scl.z, self.__m_transform.m12 * scl.z, self.__m_transform.m22 * scl.z, -self.__m_transform.m23,
                    0, 0, 0, 1)

    def inv_transform_vect(self, vec: Vec3, w) -> Vec3:
        scl: Vec3 = self.scale
        if w == 0:
            return Vec3((self.__m_transform.m00 * vec.x + self.__m_transform.m10 * vec.y + self.__m_transform.m20 * vec.z) / scl.x / scl.x,
                        (self.__m_transform.m01 * vec.x + self.__m_transform.m11 * vec.y + self.__m_transform.m21 * vec.z) / scl.y / scl.y,
                        (self.__m_transform.m02 * vec.x + self.__m_transform.m12 * vec.y + self.__m_transform.m22 * vec.z) / scl.z / scl.z)

        vec_ = Vec3(vec.x - self.x, vec.y - self.y, vec.z - self.z)
        return Vec3((self.__m_transform.m00 * vec_.x + self.__m_transform.m10 * vec_.y + self.__m_transform.m20 * vec_.z) / scl.x / scl.x,
                    (self.__m_transform.m01 * vec_.x + self.__m_transform.m11 * vec_.y + self.__m_transform.m21 * vec_.z) / scl.y / scl.y,
                    (self.__m_transform.m02 * vec_.x + self.__m_transform.m12 * vec_.y + self.__m_transform.m22 * vec_.z) / scl.z / scl.z)

    # row major 2D transform
