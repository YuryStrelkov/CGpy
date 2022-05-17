import numpy as np
from transform import Transform
from mathUtils import Vec3, Mat4
from frameBuffer import FrameBuffer

# определяет направление и положение с которого мы смотрим на 3D сцену
# определяет так же перспективное искажение
class Camera(object):
    def __init__(self):
        self.lookAtTransform: Transform = Transform()
        self.zfar: float = 1000
        self.znear: float = 0.01
        self.fov: float = 60
        self.aspect: float = 1
        self.projection: Mat4 = Mat4(1, 0, 0, 0,
                                     0, 1, 0, 0,
                                     0, 0, 1, 0,
                                     0, 0, 0, 1)
        self.build_projection()

    # Строит матрицу перспективного искажения
    def build_projection(self):
        scale = 1.0 / np.tan(self.fov * 0.5 * 3.1415 / 180)
        self.projection.m00 = scale * self.aspect  # scale the x coordinates of the projected point
        self.projection.m11 = scale  # scale the y coordinates of the projected point
        self.projection.m22 = -self.zfar / (self.zfar - self.znear)  # used to remap z to [0,1]
        self.projection.m32 = -self.zfar * self.znear / (self.zfar - self.znear)  # used to remap z [0,1]
        self.projection.m23 = -1  # set w = -z
        self.projection.m33 = 0

        # ось Z системы координат камеры

    @property
    def front(self) -> Vec3: return self.lookAtTransform.front

    # ось Y системы координат камеры
    @property
    def up(self) -> Vec3: return self.lookAtTransform.up

    # ось Z системы координат камеры
    @property
    def right(self) -> Vec3: return self.lookAtTransform.right

    # Cтроит матрицу вида
    def look_at(self, target: Vec3, eye: Vec3, up: Vec3 = Vec3(0, 1, 0)): self.lookAtTransform.look_at(target, eye, up)

    # Переводит точку в пространстве в собственную систему координат камеры
    def to_camera_space(self, v: Vec3) -> Vec3: return self.lookAtTransform.inv_transform_vect(v, 1)

    # Переводит точку в пространстве сперва в собственную систему координат камеры,
    # а после в пространство перспективной проекции
    def to_clip_space(self, vect: Vec3) -> Vec3:
        v = self.to_camera_space(vect)
        out = Vec3(
            v.x * self.projection.m00 + v.y * self.projection.m10 + v.z * self.projection.m20 + self.projection.m30,
            v.x * self.projection.m01 + v.y * self.projection.m11 + v.z * self.projection.m21 + self.projection.m31,
            v.x * self.projection.m02 + v.y * self.projection.m12 + v.z * self.projection.m22 + self.projection.m32)
        w = v.x * self.projection.m03 + v.y * self.projection.m13 + v.z * self.projection.m23 + self.projection.m33
        if w != 1:  # normalize if w is different from 1 (convert from homogeneous to Cartesian coordinates)
            out.x /= w
            out.y /= w
            out.z /= w
        return out


def render_camera(fb: FrameBuffer, lookat: Vec3, eye: Vec3) -> Camera:
    cam = Camera()
    cam.aspect = float(fb.height) / fb.width
    cam.look_at(lookat, eye)
    cam.build_projection()
    return cam
