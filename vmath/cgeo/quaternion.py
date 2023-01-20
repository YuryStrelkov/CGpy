from cgeo.trigonometry import trigonometry
from cgeo.matrices import Mat4
from typing import List, Tuple
from cgeo.vectors import Vec3
from cgeo import gutils
import numpy as np
import ctypes
import math


class Quaternion:
    @staticmethod
    def __build_from_matrix(m: Mat4) -> Tuple[float, float, float, float]:
        tr = m.m00 + m.m11 + m.m22
        if tr > 0.0:
            s: float = math.sqrt(tr + 1.0)
            ew: float = s * 0.5
            s = 0.5 / s
            ex: float = (m.m12 - m.m21) * s
            ey: float = (m.m20 - m.m02) * s
            ez: float = (m.m01 - m.m10) * s
            return ex, ey, ez, ew
        i: int
        j: int
        k: int
        if m.m11 > m.m00:
            if m.m22 > m.m11:
                i = 2
                j = 0
                k = 1
            else:
                i = 1
                j = 2
                k = 0
        elif m.m22 > m.m00:
            i = 2
            j = 0
            k = 1
        else:
            i = 0
            j = 1
            k = 2
        quaternion = [0.0, 0.0, 0.0, 0.0]
        s = math.sqrt((m[i * 4 + i] - (m[j * 4 + j] + m[k * 4 + k])) + 1.0)
        quaternion[i] = s * 0.5
        if s != 0.0:
            s = 0.5  / s
        quaternion[j] = (m[i * 4 + j] + m[j * 4 + i]) * s
        quaternion[k] = (m[i * 4 + k] + m[k * 4 + i]) * s
        quaternion[3] = (m[j * 4 + k] - m[k * 4 + j]) * s
        return quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    @staticmethod
    def __build_from_angles(ax: float, ay: float, az: float) -> Tuple[float, float, float, float]:
        return Quaternion.__build_from_matrix(gutils.rotate(ax, ay, az))

    @staticmethod
    def __build_from_quaternion(quaternion) -> Tuple[float, float, float, float]:
        return quaternion.as_tuple

    @staticmethod
    def __build_from_axis_and_angle(axis: Vec3, angle: float) -> Tuple[float, float, float, float]:
        sin_half: float = trigonometry.sin(angle * 0.5)
        cos_half: float = trigonometry.cos(angle * 0.5)
        return axis.x * sin_half, axis.y * sin_half, axis.z * sin_half, cos_half

    @staticmethod
    def __unpack_args(*args) -> Tuple[float, float, float, float]:
        args = args[0]
        n_args = len(args)
        if n_args == 4:
            return args[0], args[1], args[2], args[3]
        if n_args == 1:
            args = args[0]
            if isinstance(args, Quaternion):
                return Quaternion.__build_from_quaternion(args)
            if isinstance(args, Vec3):
                return Quaternion.__build_from_angles(args.x, args.y, args.z)
            if isinstance(args, float) or isinstance(args, int):
                return 0.0, 0.0, 0.0, args
            if isinstance(args, Mat4):
                return Quaternion.__build_from_matrix(args)
        if n_args == 2:
            if isinstance(args[0], Vec3) and (isinstance(args[1], float) or isinstance(args[1], int)):
                return Quaternion.__build_from_axis_and_angle(args[0], args[1])
        if n_args == 3:
            return Quaternion.__build_from_angles(args[0], args[1], args[2])
        if n_args == 0:
            return 0.0, 0.0, 0.0, 0.0
        raise TypeError(f'Invalid Input: {args}')

    __slots__ = "__quaternion"

    def __init__(self, *args):
        self.__quaternion: List[float] = list(Quaternion.__unpack_args(args))

    def __sizeof__(self):
        return ctypes.c_float * 4

    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return False
        if not (self.ex == other.ex):
            return False
        if not (self.ey == other.ey):
            return False
        if not (self.ez == other.ez):
            return False
        if not (self.ew == other.ew):
            return False
        return True

    def __hash__(self):
        return hash((self.ex, self.ey, self.ez, self.ew))

    def __neg__(self):
        return Quaternion(-self.ex, -self.ey, -self.ez, -self.ew)

    def __copy__(self):
        return Quaternion(self.ex, self.ey, self.ez, self.ew)

    copy = __copy__

    def __str__(self):
        return f"{{\"ex\": {self.ex:20}, \"ey\": {self.ey:20}, \"ez\": {self.ez:20}, \"angle\": {self.ew:20}}}"

    def __getitem__(self, index):
        if index < 0 or index >= 4:
            raise IndexError(f"Quaternion :: trying to access index: {index}")
        return self.__quaternion[index]

    def __setitem__(self, index: int, value: float):
        if index < 0 or index >= 4:
            raise IndexError(f"Quaternion :: trying to access index: {index}")
        self.__quaternion[index] = value

    ##########################
    #####  + operetor   ######
    ##########################
    def __iadd__(self, *args):
        other = self.__unpack_args(args)
        self.ex += other[0]
        self.ey += other[1]
        self.ez += other[2]
        self.ew += other[3]
        return self

    def __add__(self, *args):
        other = Quaternion.__unpack_args(args)
        return Quaternion(self.ex + other[0],
                          self.ey + other[1],
                          self.ez + other[2],
                          self.ew + other[3])

    __radd__ = __add__

    ##########################
    #####  - operetor   ######
    ##########################
    def __isub__(self, *args):
        other = self.__unpack_args(args)
        self.ex -= other[0]
        self.ey -= other[1]
        self.ez -= other[2]
        self.ew -= other[3]
        return self

    def __sub__(self, *args):
        other = Quaternion.__unpack_args(args)
        return Quaternion(self.ex - other[0],
                          self.ey - other[1],
                          self.ez - other[2],
                          self.ew - other[3])

    def __rsub__(self, *args):
        other = Quaternion.__unpack_args(args)
        return Quaternion(other[0] - self.ex,
                          other[1] - self.ey,
                          other[2] - self.ez,
                          other[3] - self.ew)

    ##########################
    #####  * operetor   ######
    ##########################
    def __imul__(self, *args):
        """
        https://github.com/BennyQBD/3DGameEngine/blob/master/src/com/base/engine/core/Quaternion.java
        """
        ex, ey, ez, ew = Quaternion.__unpack_args(args)
        w_ = self.ew * ew - self.ex * ex - self.ey * ey - self.ez * ez
        x_ = self.ex * ew + self.ew * ex + self.ey * ez - self.ez * ey
        y_ = self.ey * ew + self.ew * ey + self.ez * ex - self.ex * ez
        z_ = self.ez * ew + self.ew * ez + self.ex * ey - self.ey * ex
        self.ex = x_
        self.ey = y_
        self.ez = z_
        self.ew = w_
        return self

    def __mul__(self, *args):
        q = Quaternion(self)
        q *= args
        return q

    def __rmul__(self, *args):
        q1 = Quaternion(self)
        q2 = Quaternion(args)
        q2 *= q1
        return q2

    @staticmethod
    def dot(a, b) -> float:
        return a.ex * b.ex + a.ey * b.ey + a.ez * b.ez + a.ew * b.ew

    @classmethod
    def max(cls, a, b):
        return cls(max(a.ex, b.ex), max(a.ey, b.ey), max(a.ez, b.ez), max(a.ew, b.ew))

    @classmethod
    def min(cls, a, b):
        return cls(min(a.ex, b.ex), min(a.ey, b.ey), min(a.ez, b.ez), min(a.ew, b.ew))

    @classmethod
    def s_lerp(cls, q_start, q_destination, blend_factor: float):
        cos_omega: float  = cls.dot(q_start, q_destination)
        k: float = 1.0
        if cos_omega < 0.0:
            cos_omega *= -1.0
            k *= -1.0
        if cos_omega > 1.0:
            cos_omega = 1.0
        k0: float
        k1: float
        if 1.0 - cos_omega > 0.1:
            omega = trigonometry.a_cos(cos_omega)
            sin_omega = 1.0 / trigonometry.sin(omega)
            k0 = trigonometry.sin(omega * (1.0 - blend_factor)) * sin_omega
            k1 = trigonometry.sin(omega * blend_factor) * sin_omega
        else:
            k0 = 1.0 - blend_factor
            k1 = blend_factor
        k0 *= k
        return cls(q_start.ex * k0 + q_destination.ex * k1,
                   q_start.ey * k0 + q_destination.ey * k1,
                   q_start.ez * k0 + q_destination.ez * k1,
                   q_start.ew * k0 + q_destination.ew * k1)

    @staticmethod
    def rotate_vec(q, vec: Vec3) -> Vec3:
        return q.rot_vec(vec)

    def unique_id(self) -> int:
        return id(self)

    def from_np_array(self, data: np.ndarray) -> None:
        for element_id, element in enumerate(data.flat):
            self.__quaternion[element_id] = element
            if element_id == 3:
                break

    def normalize(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        nrm = 1.0 / nrm
        self.ex *= nrm
        self.ey *= nrm
        self.ez *= nrm
        self.ew *= nrm
        return self

    def normalized(self):
        nrm = self.magnitude
        if abs(nrm) < 1e-12:
            raise ArithmeticError("zero length vector")
        nrm = 1.0 / nrm
        return Quaternion(self.ex * nrm, self.ey * nrm, self.ez * nrm, self.ew * nrm)

    def conj(self):
        """
        conj self
        :return:
        """
        self.ex *= -1.0
        self.ey *= -1.0
        self.ez *= -1.0
        return self

    def conjugate(self):
        """
        returns new conjugated Quaternion
        :return:
        """
        return Quaternion(self.ex * -1.0,
                          self.ey * -1.0,
                          self.ez * -1.0,
                          self.ew)

    def inv(self):
        self.conj()
        self.normalize()
        return self

    def inverse(self):
        q = self.conjugate()
        q.normalize()
        return q

    def rot_vec(self, vec: Vec3) -> Vec3:
        q_conj = self.conjugate()
        q = self * vec * q_conj
        return Vec3(q.x, q.y, q.z)

    @property
    def as_rot_mat(self) -> Mat4:
        xx = self.ex * self.ex * 2.0
        xy = self.ex * self.ey * 2.0
        xz = self.ex * self.ez * 2.0

        yy = self.ey * self.ey * 2.0
        yz = self.ey * self.ez * 2.0
        zz = self.ez * self.ez * 2.0

        wx = self.ew * self.ex * 2.0
        wy = self.ew * self.ey * 2.0
        wz = self.ew * self.ez * 2.0
        rot = Mat4()
        rot.m00 = 1.0 - (yy + zz)
        rot.m10 = xy + wz
        rot.m20 = xz - wy
        rot.m30 = 0.0

        rot.m01 = xy - wz
        rot.m11 = 1.0 - (xx + zz)
        rot.m21 = yz + wx
        rot.m31 = 0.0

        rot.m02 = xz + wy
        rot.m12 = yz - wx
        rot.m22 = 1.0 - (xx + yy)
        rot.m32 = 0.0

        rot.m03 = 0.0
        rot.m13 = 0.0
        rot.m23 = 0.0
        rot.m33 = 1.0
        return rot

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.ex * self.ex +
                         self.ey * self.ey +
                         self.ez * self.ez +
                         self.ew * self.ew)

    @property
    def np_array(self) -> np.ndarray:
        return np.array(self.__quaternion, dtype=np.float32)

    @property
    def as_list(self) -> List[float]:
        return self.__quaternion

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.__quaternion[0], self.__quaternion[1], self.__quaternion[2], self.__quaternion[3]

    @property
    def ex(self) -> float: return self.__quaternion[0]

    @property
    def ey(self) -> float: return self.__quaternion[1]

    @property
    def ez(self) -> float: return self.__quaternion[2]

    @property
    def ew(self) -> float: return self.__quaternion[3]

    @ex.setter
    def ex(self, x: float) -> None: self.__quaternion[0] = x

    @ey.setter
    def ey(self, y: float) -> None: self.__quaternion[1] = y

    @ez.setter
    def ez(self, z: float) -> None: self.__quaternion[2] = z

    @ew.setter
    def ew(self, angle: float)  -> None: self.__quaternion[3] = angle
