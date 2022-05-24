from transforms.transform import Transform
from vmath.mathUtils import Vec3


class Ray(object):
    def __init__(self, orig=Vec3(0, 0, 0), dir: Vec3 = Vec3(0, 0, 1)):
        self.orig: Vec3 = orig
        self.dir: Vec3 = dir
        self.length = 0


class Light(object):
    """description of class"""

    def __init__(self):
        self.lightTransform: Transform = Transform()
# def emitRay(self):
