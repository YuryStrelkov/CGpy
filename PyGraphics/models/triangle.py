from transforms.transform import Transform
from vmath.vectors import Vec3, Vec2
from models.vertex import Vertex
from vmath import math_utils
from camera import Camera
import frameBuffer


class Triangle(object):

    __slots__ = "__p1", "__p2", "__p3", "__n1", "__n2", "__n3", "__uv1", "__uv2", "__uv3"

    def __init__(self,
                 _p1: Vec3, _p2: Vec3, _p3: Vec3,
                 _n1: Vec3, _n2: Vec3, _n3: Vec3,
                 _uv1: Vec2, _uv2: Vec2, _uv3: Vec2):
        self.__p1: Vec3 = _p1
        self.__p2: Vec3 = _p2
        self.__p3: Vec3 = _p3
        self.__n1: Vec3 = _n1
        self.__n2: Vec3 = _n2
        self.__n3: Vec3 = _n3
        self.__uv1: Vec2 = _uv1
        self.__uv2: Vec2 = _uv2
        self.__uv3: Vec2 = _uv3

    def __str__(self):
        return f"{{\n\t\"p1\":  {self.p1},\n\t\"p2\":  {self.p2},\n\t\"p3\":  {self.p3},\n" \
                  f"\t\"n1\":  {self.n1},\n\t\"n2\":  {self.n2},\n\t\"n3\":  {self.n3},\n" \
                  f"\t\"uv1\": {self.uv1},\n\t\"uv2\": {self.uv2},\n\t\"uv3\": {self.uv3}\n}}"

    @property
    def p1(self) -> Vec3:
        return self.__p1

    @property
    def p2(self) -> Vec3:
        return self.__p2

    @property
    def p3(self) -> Vec3:
        return self.__p3

    @property
    def n1(self) -> Vec3:
        return self.__n1

    @property
    def n2(self) -> Vec3:
        return self.__n2

    @property
    def n3(self) -> Vec3:
        return self.__n3

    @property
    def uv1(self) -> Vec2:
        return self.__uv1

    @property
    def uv2(self) -> Vec2:
        return self.__uv2

    @property
    def uv3(self) -> Vec2:
        return self.__uv3

    @property
    def vertex1(self):
        return Vertex(self.__p1, self.__n1, self.__uv1)

    @property
    def vertex2(self):
        return Vertex(self.__p2, self.__n2, self.__uv2)

    @property
    def vertex3(self):
        return Vertex(self.__p3, self.__n3, self.__uv3)

    def transform(self, tm: Transform) -> None:
        self.__p1 = tm.transform_vect(self.__p1, 1.0)
        self.__p2 = tm.transform_vect(self.__p2, 1.0)
        self.__p3 = tm.transform_vect(self.__p3, 1.0)

        self.__n1 = tm.transform_vect(self.__n1, 0.0)
        self.__n2 = tm.transform_vect(self.__n2, 0.0)
        self.__n3 = tm.transform_vect(self.__n3, 0.0)
        self.__n1.normalize()
        self.__n2.normalize()
        self.__n3.normalize()

    def inv_transform(self, tm: Transform) -> None:
        self.__p1 = tm.inv_transform_vect(self.__p1, 1.0)
        self.__p2 = tm.inv_transform_vect(self.__p2, 1.0)
        self.__p3 = tm.inv_transform_vect(self.__p3, 1.0)

        self.__n1 = tm.inv_transform_vect(self.__n1, 0.0)
        self.__n2 = tm.inv_transform_vect(self.__n2, 0.0)
        self.__n3 = tm.inv_transform_vect(self.__n3, 0.0)
        self.__n1.normalize()
        self.__n2.normalize()
        self.__n3.normalize()

    def to_clip_space(self, cam: Camera) -> None:
        self.__p1 = cam.to_clip_space(self.__p1)
        self.__p2 = cam.to_clip_space(self.__p2)
        self.__p3 = cam.to_clip_space(self.__p3)

    def to_screen_space(self, fb: frameBuffer) -> None:
        self.__p1 = Vec3(round(math_utils.clamp(0, fb.width - 1, round(fb.width * (self.__p1.x * 0.5 + 0.5)))),
                         round(math_utils.clamp(0, fb.height - 1, round(fb.height * (-self.__p1.y * 0.5 + 0.5)))),
                         self.__p1.z)
        self.__p2 = Vec3(round(math_utils.clamp(0, fb.width - 1, round(fb.width * (self.__p2.x * 0.5 + 0.5)))),
                         round(math_utils.clamp(0, fb.height - 1, round(fb.height * (-self.__p2.y * 0.5 + 0.5)))),
                         self.__p2.z)
        self.__p3 = Vec3(round(math_utils.clamp(0, fb.width - 1, round(fb.width * (self.__p3.x * 0.5 + 0.5)))),
                         round(math_utils.clamp(0, fb.height - 1, round(fb.height * (-self.__p3.y * 0.5 + 0.5)))),
                         self.__p3.z)

    def camera_screen_transform(self, cam: Camera, fb: frameBuffer) -> None:
        self.to_clip_space(cam)
        self.to_screen_space(fb)
