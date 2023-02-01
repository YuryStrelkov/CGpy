from matplotlib import pyplot as plt

from cgeo import rotate_x, rotate_y, rotate_z, rot_m_to_euler_angles, mutils, Quaternion
from cgeo.bezier.bezier_curve_3 import BezierCurve3
from cgeo.transforms.transform import Transform
from cgeo.surface.patch import CubicPatch
from cgeo.vectors import Vec3, Vec2
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
    print(f"v original       : {v}")
    v = t.transform_vect(v)
    print(f"v transformed    : {v}")
    v = t.inv_transform_vect(v)
    print(f"v inv transformed: {v}")


def transforms_2_test():
    from cgeo.transforms.transform2 import Transform2
    from cgeo.vectors import Vec2
    print("=============transforms_2_test=============")
    t = Transform2()
    v = Vec2(1, 2)
    t.x = 1.0
    t.y = 4.0
    t.sx = 2.2
    t.sy = 4.4
    t.az = 15
    print(t)
    print(f"v original       : {v}")
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



_bicubic_poly_coefficients = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              -3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                              -3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0,
                              9.0, -9.0, -9.0, 9.0, 6.0, 3.0, -6.0, -3.0, 6.0, -6.0, 3.0, -3.0, 4.0, 2.0, 2.0, 1.0,
                              -6.0, 6.0, 6.0, -6.0, -4.0, -2.0, 4.0, 2.0, -3.0, 3.0, -3.0, 3.0, -2.0, -1.0, -2.0, -1.0,
                              2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                              -6.0, 6.0, 6.0, -6.0, -3.0, -3.0, 3.0, 3.0, -4.0, 4.0, -2.0, 2.0, -2.0, -2.0, -1.0, -1.0,
                              4.0, -4.0, -4.0, 4.0, 2.0, 2.0, -2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 1.0, 1.0, 1.0, 1.0)


if __name__ == '__main__':

    pnts = [(5.0, 5.0), (10.0, 10.0), (15.0, 5.0), (20.0, -10.0), (15.0, 10.0), (15.0, 20.0), (50, -5)]
    xy = mutils.quad_interpolate_line2(pnts)
    x = [v[0] for v in xy]
    y = [v[1] for v in xy]
    plt.plot(x, y, 'r')
    [plt.plot(px, py, "go") for (px, py) in pnts]
    plt.show()

    ax = 36.0  # in [0:180)
    ay = 47.0  # in [0:90)
    az = 58.0  # in [0:180)

    xr = rotate_x(ax / 180.0 * np.pi)
    yr = rotate_y(ay / 180.0 * np.pi)
    zr = rotate_z(az / 180.0 * np.pi)

    rm = zr * yr * xr

    angles = rot_m_to_euler_angles(rm)

    q = Quaternion(ax / 180.0 * np.pi, ay / 180.0 * np.pi, az / 180.0 * np.pi)

    qm = q.as_rot_mat

    print(qm)

    print(rm)

    print(f"=> ax: {ax:3} | ay: {ay:3} | az: {az:3}")

    print(f"<= ax: {angles.x/np.pi*180:3} | ay: {angles.y/np.pi*180:3} | az: {angles.z/np.pi*180:3}")

    anglesq = rot_m_to_euler_angles(qm)

    print(f"<= ax: {anglesq.x/np.pi*180:3} | ay: {anglesq.y/np.pi*180:3} | az: {anglesq.z/np.pi*180:3}")

    exit()
    str_ = ""
    cntr = 0
    for row in range(16):
        str_ += f"c[{row}] = "
        for col in range(16):
            val = _bicubic_poly_coefficients[row * 16 + col]
            if val == 0.0:
                continue
            if val > 0:
                str_ += f"+{val:1} * b[{col:2}]"
            else:
                str_ += f"{val:1} * b[{col:2}]"
            cntr += 1
        str_ += ";\n"
    print(f"cntr: {cntr}, cntr_percent {cntr/256 * 100:3}")
    print(str_)

    #time_test()
    transforms_2_test()
    transforms_3_test()
    #surface_test()
    #bezier_test()

