from models.tris_mesh import TrisMesh, BoundingBox
from transforms.transform import Transform
from models.triangle import Triangle
from vmath.math_utils import Vec3
from models import tris_mesh
from vmath import vectors


def quadratic_bezier_patch(p1: Vec3, p2: Vec3, p3: Vec3,
                           p4: Vec3, p5: Vec3, p6: Vec3,
                           p7: Vec3, p8: Vec3, p9: Vec3, u: float, v: float) -> [Vec3]:
    phi1: float = (1 - u) * (1 - u)
    phi3: float = u * u
    phi2: float = -2 * phi3 + 2 * u

    psi1: float = (1 - v) * (1 - v)
    psi3: float = v * v
    psi2: float = -2 * psi3 + 2 * v

    p: Vec3 = p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + \
              p4 * phi2 * psi1 + p5 * phi2 * psi2 + p6 * phi2 * psi3 + \
              p7 * phi3 * psi1 + p8 * phi3 * psi2 + p9 * phi3 * psi3

    dp1: float = -2 + u * 2
    dp2: float =  2 * u
    dp3: float = -2 * psi3 + 2

    du: Vec3 = p1 * dp1 * psi1 + p2 * dp1 * psi2 + p3 * dp1 * psi3 + \
               p4 * dp2 * psi1 + p5 * dp2 * psi2 + p6 * dp2 * psi3 + \
               p7 * dp3 * psi1 + p8 * dp3 * psi2 + p9 * dp3 * psi3

    dp1 = -2 + v * 2
    dp2 =  2 * v
    dp3 = -2 * psi3 + 2

    dv: Vec3 = p1 * phi1 * dp1 + p2 * phi1 * dp2 + p3 * phi1 * dp3 + \
               p4 * phi2 * dp1 + p5 * phi2 * dp2 + p6 * phi2 * dp3 + \
               p7 * phi3 * dp1 + p8 * phi3 * dp2 + p9 * phi3 * dp3

    return [p, vectors.cross(dv, du).normalize()]


def cubic_bezier_patch(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3,
                       p5: Vec3, p6: Vec3, p7: Vec3, p8: Vec3,
                       p9: Vec3, p10: Vec3, p11: Vec3, p12: Vec3,
                       p13: Vec3, p14: Vec3, p15: Vec3, p16: Vec3, u: float, v: float) -> [Vec3]:

    phi1: float = (1 - u) * (1 - u) * (1 - u)
    phi4: float =  u * u * u
    phi2: float =  3 * phi4 - 6 * u * u + 3 * u
    phi3: float = -3 * phi4 + 3 * u * u

    psi1: float = (1 - v) * (1 - v) * (1 - v)
    psi4: float =  v * v * v
    psi2: float =  3 * psi4 - 6 * v * v + 3 * v
    psi3: float = -3 * psi4 + 3 * v * v

    p: Vec3 = p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + p4 * phi1 * psi4 + \
              p5 * phi2 * psi1 + p6 * phi2 * psi2 + p7 * phi2 * psi3 + p8 * phi2 * psi4 + \
              p9 * phi3 * psi1 + p10 * phi3 * psi2 + p11 * phi3 * psi3 + p12 * phi3 * psi4 + \
              p13 * phi4 * psi1 + p14 * phi4 * psi2 + p15 * phi4 * psi3 + p16 * phi4 * psi4

    d4: float =  3 * u * u
    d1: float = -3 + 6 * u - d4
    d2: float =  3 * phi4 - 12 * u + 3
    d3: float = -3 * phi4 + 6 * u

    dpu: Vec3 = p1  * d1 * psi1 + p2  * d1 * psi2 + p3  * d1 * psi3 + p4  * d1 * psi4 + \
                p5  * d2 * psi1 + p6  * d2 * psi2 + p7  * d2 * psi3 + p8  * d2 * psi4 + \
                p9  * d3 * psi1 + p10 * d3 * psi2 + p11 * d3 * psi3 + p12 * d3 * psi4 + \
                p13 * d4 * psi1 + p14 * d4 * psi2 + p15 * d4 * psi3 + p16 * d4 * psi4

    d4 =  3 * v * v
    d1 = -3 + 6 * v - d4
    d2 =  3 * phi4 - 12 * v + 3
    d3 = -3 * phi4 + 6 * v

    dpv: Vec3 = p1  * phi1 * d1 + p2  * phi1 * d2 + p3  * phi1 * d3 + p4  * phi1 * d4 + \
                p5  * phi2 * d1 + p6  * phi2 * d2 + p7  * phi2 * d3 + p8  * phi2 * d4 + \
                p9  * phi3 * d1 + p10 * phi3 * d2 + p11 * phi3 * d3 + p12 * phi3 * d4 + \
                p13 * phi4 * d1 + p14 * phi4 * d2 + p15 * phi4 * d3 + p16 * phi4 * d4

    return [p, vectors.cross(dpv, dpu).normalize()]


class CubicPatch(object):
    def __init__(self):
        self.__width_points = 64
        self.__height_points = 64
        self.__transform: Transform = Transform()
        self.__mesh: TrisMesh = tris_mesh.create_plane(1.0, 1.0, self.__height_points, self.__width_points)
        self.__controllers: [Vec3] = \
            [Vec3(-0.5, 0,   -0.5),    Vec3(-0.1666, 0.1, -0.5),    Vec3(0.1666, 0.1, -0.5),    Vec3(0.5, 0,   -0.5),
             Vec3(-0.5, 0.1, -0.1666), Vec3(-0.1666, 1,   -0.1666), Vec3(0.1666, 1,   -0.1666), Vec3(0.5, 0.1, -0.1666),
             Vec3(-0.5, 0.1,  0.1666), Vec3(-0.1666, 1,    0.1666), Vec3(0.1666, 1,    0.1666), Vec3(0.5, 0.1,  0.1666),
             Vec3(-0.5, 0,    0.5),    Vec3(-0.1666, 0.1,  0.5),    Vec3(0.1666, 0.1,  0.5),    Vec3(0.5, 0,    0.5)]
        self.__update_mesh()

    def __update_mesh(self) -> None:
        u: float
        v: float
        for i in range(0, self.__mesh.vertices_count):
            u = (i / self.__width_points) / float(self.__width_points - 1)
            v = (i % self.__width_points) / float(self.__width_points - 1)
            [v_, n_] = cubic_bezier_patch(self.__controllers[0], self.__controllers[1],
                                          self.__controllers[2], self.__controllers[3],
                                          self.__controllers[4], self.__controllers[5],
                                          self.__controllers[6], self.__controllers[7],
                                          self.__controllers[8], self.__controllers[9],
                                          self.__controllers[10], self.__controllers[11],
                                          self.__controllers[12], self.__controllers[13],
                                          self.__controllers[14], self.__controllers[15],
                                          u, v)
            self.__mesh.set_vertex(i, v_)
            self.__mesh.set_normal(i, n_)

    def __update_control_point(self, control_point_id, pos: Vec3) -> None:
        self.__controllers[control_point_id] = pos
        self.__update_mesh()

    @property
    def patch_mesh(self) -> TrisMesh:
        return self.__mesh

    @property
    def control_points(self) -> [Vec3]:
        return self.__controllers

    @property
    def bbox(self) -> BoundingBox:
        return self.__mesh.bbox

    @property
    def center_world_space(self) -> Vec3:
        return self.__transform.transform_vect(self.bbox.center, 1)

    @property
    def min_world_space(self) -> Vec3:
        return self.__transform.transform_vect(self.bbox.min, 1)

    @property
    def max_world_space(self) -> Vec3:
        return self.__transform.transform_vect(self.bbox.max, 1)

    @property
    def size_world_space(self) -> Vec3:
        return self.__transform.transform_vect(self.bbox.size, 1)

    def triangles_local_space(self):
        tris_id: int = 0
        while tris_id < self.__mesh.faces_count:
            yield self.__mesh.get_triangle(tris_id)
            tris_id += 1

    def triangles_world_space(self):
        tris_id: int = 0
        while tris_id < self.__mesh.faces_count:
            tris: Triangle = self.__mesh.get_triangle(tris_id)
            tris.transform(self.__transform)
            yield tris
            tris_id += 1

    @property
    def transform(self):
        return self.__transform

    @property
    def p1(self):
        return self.__controllers[0]

    @property
    def p2(self):
        return self.__controllers[1]

    @property
    def p3(self):
        return self.__controllers[2]

    @property
    def p4(self):
        return self.__controllers[3]

    @property
    def p5(self):
        return self.__controllers[4]

    @property
    def p6(self):
        return self.__controllers[5]

    @property
    def p7(self):
        return self.__controllers[6]

    @property
    def p8(self):
        return self.__controllers[7]

    @property
    def p9(self):
        return self.__controllers[8]

    @property
    def p10(self):
        return self.__controllers[9]

    @property
    def p11(self):
        return self.__controllers[10]

    @property
    def p12(self):
        return self.__controllers[11]

    @property
    def p13(self):
        return self.__controllers[12]

    @property
    def p14(self):
        return self.__controllers[13]

    @property
    def p15(self):
        return self.__controllers[14]

    @property
    def p16(self):
        return self.__controllers[15]

    ###########################################
    @p1.setter
    def p1(self, p: Vec3):
        self.__controllers[0] = p
        self.__update_mesh()

    @p2.setter
    def p2(self, p: Vec3):
        self.__controllers[1] = p
        self.__update_mesh()

    @p3.setter
    def p3(self, p: Vec3):
        self.__controllers[2] = p
        self.__update_mesh()

    @p4.setter
    def p4(self, p: Vec3):
        self.__controllers[3] = p
        self.__update_mesh()

    ###########################################
    @p5.setter
    def p5(self, p: Vec3):
        self.__controllers[4] = p
        self.__update_mesh()

    @p6.setter
    def p6(self, p: Vec3):
        self.__controllers[5] = p
        self.__update_mesh()

    @p7.setter
    def p7(self, p: Vec3):
        self.__controllers[6] = p
        self.__update_mesh()

    @p8.setter
    def p8(self, p: Vec3):
        self.__controllers[7] = p
        self.__update_mesh()

    ###########################################
    @p9.setter
    def p9(self, p: Vec3):
        self.__controllers[8] = p
        self.__update_mesh()

    @p10.setter
    def p10(self, p: Vec3):
        self.__controllers[9] = p
        self.__update_mesh()

    @p11.setter
    def p11(self, p: Vec3):
        self.__controllers[10] = p
        self.__update_mesh()

    @p12.setter
    def p12(self, p: Vec3):
        self.__controllers[11] = p
        self.__update_mesh()

    ###########################################
    @p13.setter
    def p13(self, p: Vec3):
        self.__controllers[12] = p
        self.__update_mesh()

    @p14.setter
    def p14(self, p: Vec3):
        self.__controllers[13] = p
        self.__update_mesh()

    @p15.setter
    def p15(self, p: Vec3):
        self.__controllers[14] = p
        self.__update_mesh()

    @p16.setter
    def p16(self, p: Vec3):
        self.__controllers[15] = p
        self.__update_mesh()
###########################################
