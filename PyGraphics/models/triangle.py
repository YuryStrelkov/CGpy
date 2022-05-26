import frameBuffer
from camera import Camera
from models.vertex import Vertex
from transforms.transform import Transform
from vmath import mathUtils
from vmath.vectors import Vec3, Vec2


class Triangle(object):
    def __init__(self,
                 _p1: Vec3, _p2: Vec3, _p3: Vec3,
                 _n1: Vec3, _n2: Vec3, _n3: Vec3,
                 _uv1: Vec2, _uv2: Vec2, _uv3: Vec2):
        self.p1: Vec3 = _p1
        self.p2: Vec3 = _p2
        self.p3: Vec3 = _p3
        self.n1: Vec3 = _n1
        self.n2: Vec3 = _n2
        self.n3: Vec3 = _n3
        self.uv1: Vec2 = _uv1
        self.uv2: Vec2 = _uv2
        self.uv3: Vec2 = _uv3

    @property
    def vertex1(self):
        return Vertex(self.p1, self.n1, self.uv1)

    @property
    def vertex2(self):
        return Vertex(self.p2, self.n2, self.uv2)

    @property
    def vertex3(self):
        return Vertex(self.p3, self.n3, self.uv3)

    def transform(self, tm: Transform) -> None:
        self.p1 = tm.transform_vect(self.p1, 1.0)
        self.p2 = tm.transform_vect(self.p2, 1.0)
        self.p3 = tm.transform_vect(self.p3, 1.0)

        self.n1 = tm.transform_vect(self.n1, 0.0)
        self.n2 = tm.transform_vect(self.n2, 0.0)
        self.n3 = tm.transform_vect(self.n3, 0.0)
        self.n1.normalize()
        self.n2.normalize()
        self.n3.normalize()

    def inv_transform(self, tm: Transform) -> None:
        self.p1 = tm.inv_transform_vect(self.p1, 1.0)
        self.p2 = tm.inv_transform_vect(self.p2, 1.0)
        self.p3 = tm.inv_transform_vect(self.p3, 1.0)

        self.n1 = tm.inv_transform_vect(self.n1, 0.0)
        self.n2 = tm.inv_transform_vect(self.n2, 0.0)
        self.n3 = tm.inv_transform_vect(self.n3, 0.0)
        self.n1.normalize()
        self.n2.normalize()
        self.n3.normalize()

    def to_clip_space(self, cam: Camera) -> None:
        self.p1 = cam.to_clip_space(self.p1)
        self.p2 = cam.to_clip_space(self.p2)
        self.p3 = cam.to_clip_space(self.p3)

    def to_screen_space(self, fb: frameBuffer) -> None:
        self.p1 = Vec3(round(mathUtils.clamp(0, fb.width - 1, round(fb.width * (self.p1.x * 0.5 + 0.5)))),
                       round(mathUtils.clamp(0, fb.height - 1, round(fb.height * (-self.p1.y * 0.5 + 0.5)))),
                       self.p1.z)
        self.p2 = Vec3(round(mathUtils.clamp(0, fb.width - 1, round(fb.width * (self.p2.x * 0.5 + 0.5)))),
                       round(mathUtils.clamp(0, fb.height - 1, round(fb.height * (-self.p2.y * 0.5 + 0.5)))),
                       self.p2.z)
        self.p3 = Vec3(round(mathUtils.clamp(0, fb.width - 1, round(fb.width * (self.p3.x * 0.5 + 0.5)))),
                       round(mathUtils.clamp(0, fb.height - 1, round(fb.height * (-self.p3.y * 0.5 + 0.5)))),
                       self.p3.z)

    def camera_screen_transform(self, cam: Camera, fb: frameBuffer) -> None:
        self.to_clip_space(cam)
        self.to_screen_space(fb)
