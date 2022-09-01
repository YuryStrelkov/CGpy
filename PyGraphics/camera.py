from transforms.transform import Transform
from vmath.math_utils import Vec3, Mat4
import math


# определяет направление и положение с которого мы смотрим на 3D сцену
# определяет так же перспективное искажение
class Camera(object):

    __slots__ = "__transform", "__zfar", "__znear", "__projection", "__inv_projection"

    def __init__(self):
        self.__transform: Transform = Transform()
        self.__zfar: float = 1000
        self.__znear: float = 0.01
        self.__projection: Mat4 = Mat4(1, 0, 0, 0,
                                       0, 1, 0, 0,
                                       0, 0, 1, 0,
                                       0, 0, 0, 1)
        self.__inv_projection: Mat4 = Mat4(1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1, 0,
                                           0, 0, 0, 1)
        self.fov = 60
        self.aspect = 1
        self.__build_projection()

    def __repr__(self) -> str:
        res: str = "<Camera \n"
        res += f"z far     :{self.__zfar}\n"
        res += f"z near    :{self.__znear}\n"
        res += f"fov       :{self.fov}\n"
        res += f"aspect    :{self.aspect}\n"
        res += f"projection:\n{self.__projection}\n"
        res += f"transform :\n{self.__transform}\n"
        res += ">"
        return res

    def __str__(self) -> str:
        res: str = "Camera \n"
        res += f"z far     :{self.__zfar}\n"
        res += f"z near    :{self.__znear}\n"
        res += f"fov       :{self.fov}\n"
        res += f"aspect    :{self.aspect}\n"
        res += f"projection:\n{self.__projection}\n"
        res += f"transform :\n{self.__transform}\n"
        return res

    def __eq__(self, other) -> bool:
        if not (type(other) is Camera):
            return False
        if not (self.__transform == other.__transform):
            return False
        if not (self.__projection == other.__projection):
            return False
        return True

    def __hash__(self) -> int:
        return hash((self.__transform, self.__projection))

    # Строит матрицу перспективного искажения
    def __build_projection(self):
        self.__projection.m22 = self.__zfar / (self.__znear - self.__zfar)  # used to remap z to [0,1]
        self.__projection.m32 = self.__zfar * self.__znear / (self.__znear - self.__zfar)  # used to remap z [0,1]
        self.__projection.m23 = -1  # set w = -z
        self.__projection.m33 = 0
        self.__inv_projection = self.__projection.copy()
        self.__inv_projection.invert()

    @property
    def unique_id(self) -> int:
        return id(self)

    @property
    def transform(self) -> Transform:
        return self.__transform

    @property
    def projection(self) -> Mat4:
        return self.__projection

    @property
    def inv_projection(self) -> Mat4:
        return self.__inv_projection

    @property
    def zfar(self) -> float:
        return self.__zfar

    @zfar.setter
    def zfar(self, far_plane: float) -> None:
        self.__zfar = far_plane
        self.__build_projection()

    @property
    def znear(self) -> float:
        return self.__znear

    @znear.setter
    def znear(self, near_plane: float) -> None:
        self.__znear = near_plane
        self.__build_projection()

    @property
    def fov(self) -> float:
        return math.atan(1.0 / self.__projection.m11) * 2 / math.pi * 180

    @fov.setter
    def fov(self, fov_: float) -> None:
        scale = 1.0 / math.tan(fov_ * 0.5 * math.pi / 180)
        self.__projection.m00 *= (scale / self.__projection.m11)
        self.__projection.m11 = scale  # scale the y coordinates of the projected point
        self.__inv_projection = self.__projection.copy()
        self.__inv_projection.invert()

    @property
    def aspect(self) -> float:
        return self.__projection.m00 / self.__projection.m11

    @aspect.setter
    def aspect(self, aspect_: float) -> None:
        self.__projection.m00 *= (aspect_ / self.aspect)
        self.__inv_projection = self.__projection.copy()
        self.__inv_projection.invert()

    @property
    def front(self) -> Vec3:
        return self.__transform.front

    # ось Y системы координат камеры
    @property
    def up(self) -> Vec3:
        return self.__transform.up

    # ось Z системы координат камеры
    @property
    def right(self) -> Vec3:
        return self.__transform.right

    # Cтроит матрицу вида
    def look_at(self, target: Vec3, eye: Vec3, up: Vec3 = Vec3(0, 1, 0)) -> None:
        self.__transform.look_at(target, eye, up)

    # Переводит точку в пространстве в собственную систему координат камеры
    def to_camera_space(self, v: Vec3) -> Vec3:
        return self.__transform.inv_transform_vect(v, 1)

    # Переводит точку в пространстве сперва в собственную систему координат камеры,
    # а после в пространство перспективной проекции
    def to_clip_space(self, vect: Vec3) -> Vec3:
        v = self.to_camera_space(vect)
        out = Vec3(
            v.x * self.__projection.m00 + v.y * self.__projection.m10 + v.z * self.__projection.m20 + self.__projection.m30,
            v.x * self.__projection.m01 + v.y * self.__projection.m11 + v.z * self.__projection.m21 + self.__projection.m31,
            v.x * self.__projection.m02 + v.y * self.__projection.m12 + v.z * self.__projection.m22 + self.__projection.m32)
        w = v.x * self.__projection.m03 + v.y * self.__projection.m13 + v.z * self.__projection.m23 + self.__projection.m33
        if w != 1:  # normalize if w is different from 1 (convert from homogeneous to Cartesian coordinates)
            out.x /= w
            out.y /= w
            out.z /= w
        return out
