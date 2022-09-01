from vmath.vectors import Vec3, Vec2
from vmath import math_utils
from camera import Camera
import frameBuffer


class Vertex:
    @staticmethod
    def __unpack_values(*args) -> tuple:
        args = args[0]

        number_of_args = len(args)

        if number_of_args == 0:
            return 0, 0, 0, 0, 0, 0, 0, 0  # no arguments

        elif number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is float or arg_type is int:  # single int or float argument
                return args[0], args[0], args[0], args[0], args[0], args[0], args[0], args[0]

            if arg_type is Vertex:
                return args[0].v.x, args[0].v.y, args[0].v.z, \
                       args[0].n.x, args[0].n.y, args[0].n.z, \
                       args[0].uv.x, args[0].uv.y

        raise TypeError(f'Invalid Input: {args}')

    __slots__ = "__v", "__n", "__uv"

    def __init__(self, v_: Vec3, n_: Vec3, uv_: Vec2):
        self.__v: Vec3 = v_
        self.__n: Vec3 = n_
        self.__uv: Vec2 = uv_

    def __add__(self, *args):
        other = self.__unpack_values(args)
        return Vertex(self.v + Vec3(other[0], other[1], other[2]),
                      self.n + Vec3(other[3], other[4], other[5]),
                      self.uv + Vec2(other[6], other[7]))

    def __sub__(self, *args):
        other = self.__unpack_values(args)
        return Vertex(self.v - Vec3(other[0], other[1], other[2]),
                      self.n - Vec3(other[3], other[4], other[5]),
                      self.uv - Vec2(other[6], other[7]))

    def __mul__(self, *args):
        other = self.__unpack_values(args)
        return Vertex(self.v * Vec3(other[0], other[1], other[2]),
                      self.n * Vec3(other[3], other[4], other[5]),
                      self.uv * Vec2(other[6], other[7]))

    def __truediv__(self, *args):
        other = self.__unpack_values(args)
        return Vertex(self.v / Vec3(other[0], other[1], other[2]),
                      self.n / Vec3(other[3], other[4], other[5]),
                      self.uv / Vec2(other[6], other[7]))

    @property
    def v(self) -> Vec3:
        return self.__v

    @property
    def n(self) -> Vec3:
        return self.__n

    @property
    def uv(self) -> Vec2:
        return self.__uv

    def to_clip_space(self, cam: Camera) -> None:
        self.__v = cam.to_clip_space(self.v)

    def to_screen_space(self, fb: frameBuffer) -> None:
        self.__v = Vec3(round(math_utils.clamp(0, fb.width - 1, round(fb.width * (self.v.x * 0.5 + 0.5)))),
                        round(math_utils.clamp(0, fb.height - 1, round(fb.height * (-self.v.y * 0.5 + 0.5)))),
                        self.v.z)

    def camera_screen_transform(self, cam: Camera, fb: frameBuffer) -> None:
        self.to_clip_space(cam)
        self.to_screen_space(fb)


def lerp_vertex(a: Vertex, b: Vertex, val: float) -> Vertex:
    return a + (b - a) * val
