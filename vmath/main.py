from computational_geometry.bezier.bezier_curve_3 import BezierCurve3
from computational_geometry.transforms.transform import Transform
from computational_geometry.surface.patch import CubicPatch
from computational_geometry.vectors import Vec3, Vec2
import numpy as np
import time


def vec2_test():
    print("=================vec2_test================")
    v1 = Vec2(1, 3)
    v2 = Vec2(3, 1)
    print(f"v1 + v2:  {v1 + v2}")
    print(f"v1 - v2:  {v1 - v2}")
    print(f"v1 * v2:  {v1 * v2}")
    print(f"v1 / v2:  {v1 / v2}")
    print(f"(v1,v2):  {Vec2.dot(v1, v2)}")
    print(f"[v1;v2]:  {Vec2.cross(v1, v2)}")



def transforms_3_test():
    print("=============transforms_3_test=============")
    t = Transform()
    v = Vec3(1, 2, 3)
    t.x = 1.0
    t.y = 4.0
    t.z = 8.0
    t.sx = 2.2
    t.sy = 4.4
    t.sz = 5.5
    t.ax = 15
    t.ay = 30
    t.ay = 60
    print(t)
    v = t.transform_vect(v)
    print(f"v transformed    : {v}")
    v = t.inv_transform_vect(v)
    print(f"v inv transformed: {v}")


def transforms_2_test():
    from computational_geometry.transforms.transform2 import Transform2
    from computational_geometry.vectors import Vec2
    print("=============transforms_2_test=============")
    t = Transform2()
    v = Vec2(1, 2)
    t.x = 1.0
    t.y = 4.0
    t.sx = 2.2
    t.sy = 4.4
    t.az = 15
    print(t)
    v = t.transform_vect(v)
    print(f"v transformed    : {v}")
    v = t.inv_transform_vect(v)
    print(f"v inv transformed: {v}")


def surface_test():
    print("=============surface_test=============")
    patch = CubicPatch()
    print(patch)
   # print(patch.__mesh)


def bezier_test():
    print("=============bezier_test=============")
    curve = BezierCurve3()
    curve.add_point(Vec3(0, 0, 0), True)
    curve.add_point(Vec3(1, 0, 0), True)
    curve.add_point(Vec3(1, 1, 0), True)
    print(curve)


def time_test():
    t = time.time()
    for i in range(100000):
        arr = [3.0, 3.0, 3.0]
    print(f"list time {time.time() - t}")

    for i in range(100000):
        arr = np.zeros((3,), dtype=float)
    print(f"np time {time.time() - t}")


if __name__ == '__main__':
    vec2_test()
    time_test()
    transforms_2_test()
    transforms_3_test()
    surface_test()
    bezier_test()

