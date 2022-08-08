import math
from vmath.matrices import Mat3
from vmath.vectors import Vec2


class Transform2(object):
    def __init__(self):
        self.__transform_m = Mat3(1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0)
        self.zAngle = 0.0

    def __repr__(self) -> str:
        res: str = "<Transform \n"
        res += f"origin   :{self.origin}\n"
        res += f"scale    :{self.scale}\n"
        res += f"rotate   :{self.zAngle}\n"
        res += f"t-matrix :\n{self.__transform_m}\n"
        res += ">"
        return res

    def __str__(self) -> str:
        res: str = "Transform \n"
        res += f"origin   :{self.origin}\n"
        res += f"scale    :{self.scale}\n"
        res += f"rotate   :{self.zAngle}\n"
        res += f"t-matrix :\n{self.__transform_m}\n"
        return res

    def __eq__(self, other) -> bool:
        if not(type(other) is Transform2):
            return False
        if not(self.__transform_m == other.__m_transform):
            return False
        return True

    def __hash__(self) -> int:
        return hash(self.__transform_m)

    @property
    def transform_matrix(self) -> Mat3:
        return self.__transform_m

    @property
    def front(self) -> Vec2:
        return Vec2(self.__transform_m.m00, self.__transform_m.m10).normalize()

    @property
    def up(self) -> Vec2:
        return Vec2(self.__transform_m.m01, self.__transform_m.m11).normalize()

    @property
    def scale(self) -> Vec2:
        return Vec2(self.sx, self.sy)

    # масштаб по Х
    @property
    def sx(self) -> float:
        x = self.__transform_m.m00
        y = self.__transform_m.m10
        return math.sqrt(x * x + y * y)

    # масштаб по Y
    @property
    def sy(self) -> float:
        x = self.__transform_m.m01
        y = self.__transform_m.m11
        return math.sqrt(x * x + y * y)
        # установить масштаб по Х

    @sx.setter
    def sx(self, s_x: float):
        if s_x == 0:
            return
        scl = self.sx
        self.__transform_m.m00 *= s_x / scl
        self.__transform_m.m10 *= s_x / scl

    # установить масштаб по Y
    @sy.setter
    def sy(self, s_y: float):
        if s_y == 0:
            return
        scl = self.sy
        self.__transform_m.m01 *= s_y / scl
        self.__transform_m.m11 *= s_y / scl

    @scale.setter
    def scale(self, sxy: Vec2):
        self.sx = sxy.x
        self.sy = sxy.y

    @property
    def x(self) -> float:
        return self.__transform_m.m02

    @property
    def y(self) -> float:
        return self.__transform_m.m12

    @property
    def origin(self) -> Vec2:
        return Vec2(self.x, self.y)

    @x.setter
    def x(self, x: float):
        self.__transform_m.m02 = x

    @y.setter
    def y(self, y: float):
        self.__transform_m.m12 = y

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
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rz = Mat3(cos_a, -sin_a, 0,
                  sin_a, cos_a, 0,
                  0, 0, 1)
        scl = self.scale
        orig = self.origin
        self.__transform_m = rz
        self.scale = scl
        self.origin = orig

    # переводит вектор в собственное пространство координат
    def transform_vect(self, vec: Vec2, w) -> Vec2:
        if w == 0:
            return Vec2(self.__transform_m.m00 * vec.x + self.__transform_m.m01 * vec.y,
                        self.__transform_m.m10 * vec.x + self.__transform_m.m11 * vec.y)

        return Vec2(self.__transform_m.m00 * vec.x + self.__transform_m.m01 * vec.y + self.__transform_m.m02,
                    self.__transform_m.m10 * vec.x + self.__transform_m.m11 * vec.y + self.__transform_m.m12)

    # не переводит вектор в собственное пространство координат =)
    def inv_transform_vect(self, vec: Vec2, w) -> Vec2:
        scl: Vec2 = self.scale
        if w == 0:
            return Vec2((self.__transform_m.m00 * vec.x + self.__transform_m.m10 * vec.y) / scl.x / scl.x,
                        (self.__transform_m.m01 * vec.x + self.__transform_m.m11 * vec.y) / scl.y / scl.y)

        vec_ = Vec2(vec.x - self.x, vec.y - self.y)
        return Vec2((self.__transform_m.m00 * vec_.x + self.__transform_m.m10 * vec_.y) / scl.x / scl.x,
                    (self.__transform_m.m01 * vec_.x + self.__transform_m.m11 * vec_.y) / scl.y / scl.y)