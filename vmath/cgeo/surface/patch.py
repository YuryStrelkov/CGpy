from cgeo.tris_mesh.tris_mesh import create_plane
from cgeo.bounds.bounding_box import BoundingBox
from cgeo.transforms.transform import Transform
from cgeo.tris_mesh.tris_mesh import TrisMesh
from cgeo.tris_mesh.triangle import Triangle
from cgeo.gutils import cubic_bezier_patch
from cgeo.vectors import Vec3
from typing import List


class CubicPatch:

    __slots__ = "__width_points", "__height_points", "__transform", "__mesh", "__controllers"

    def __init__(self):
        self.__width_points: int = 8
        self.__height_points: int = 8
        self.__transform: Transform = Transform()
        self.__mesh: TrisMesh = create_plane(1.0, 1.0, self.__height_points, self.__width_points)
        # print(self.__mesh)
        self.__controllers: List[Vec3] = \
            [Vec3(-0.5, 0,   -0.5),    Vec3(-0.1666, 0.1, -0.5),    Vec3(0.1666, 0.1, -0.5),    Vec3(0.5, 0,   -0.5),
             Vec3(-0.5, 0.1, -0.1666), Vec3(-0.1666, 1,   -0.1666), Vec3(0.1666, 1,   -0.1666), Vec3(0.5, 0.1, -0.1666),
             Vec3(-0.5, 0.1,  0.1666), Vec3(-0.1666, 1,    0.1666), Vec3(0.1666, 1,    0.1666), Vec3(0.5, 0.1,  0.1666),
             Vec3(-0.5, 0,    0.5),    Vec3(-0.1666, 0.1,  0.5),    Vec3(0.1666, 0.1,  0.5),    Vec3(0.5, 0,    0.5)]
        self.__update_mesh()

    def __str__(self):
        nl = ",\n\t\t"
        return f"{{\n\t\"width_points\":  {self.__width_points},\n" \
                   f"\t\"height_points\": {self.__height_points},\n" \
                   f"\t\"transform\":     {self.transform},\n" \
                   f"\t\"controllers\":\n\t[\n\t\t{nl.join(str(v) for v in self.__controllers)}\n\t]\n}}"

    def __update_mesh(self) -> None:
        u: float
        v: float
        for i in range(self.__mesh.vertices_count):
            u = (i / self.__width_points) / float(self.__width_points - 1)
            v = (i % self.__width_points) / float(self.__width_points - 1)
            v_, n_ = cubic_bezier_patch(self.__controllers[0],  self.__controllers[1],
                                        self.__controllers[2],  self.__controllers[3],
                                        self.__controllers[4],  self.__controllers[5],
                                        self.__controllers[6],  self.__controllers[7],
                                        self.__controllers[8],  self.__controllers[9],
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
    def control_points(self) -> List[Vec3]:
        return self.__controllers

    @property
    def bbox(self) -> BoundingBox:
        return self.__mesh.bbox

    @property
    def center_world_space(self) -> Vec3:
        return self.__transform.transform_vect(self.bbox.center, 1.0)

    @property
    def min_world_space(self) -> Vec3:
        return self.__transform.transform_vect(self.bbox.min, 1.0)

    @property
    def max_world_space(self) -> Vec3:
        return self.__transform.transform_vect(self.bbox.max, 1.0)

    @property
    def size_world_space(self) -> Vec3:
        return self.__transform.transform_vect(self.bbox.size, 1.0)

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
    def transform(self) -> Transform:
        return self.__transform

    ###########################################
    @property
    def p1(self) -> Vec3:
        return self.__controllers[0]

    @property
    def p2(self) -> Vec3:
        return self.__controllers[1]

    @property
    def p3(self) -> Vec3:
        return self.__controllers[2]

    @property
    def p4(self) -> Vec3:
        return self.__controllers[3]

    ###########################################
    @property
    def p5(self) -> Vec3:
        return self.__controllers[4]

    @property
    def p6(self) -> Vec3:
        return self.__controllers[5]

    @property
    def p7(self) -> Vec3:
        return self.__controllers[6]

    @property
    def p8(self) -> Vec3:
        return self.__controllers[7]

    ###########################################
    @property
    def p9(self) -> Vec3:
        return self.__controllers[8]

    @property
    def p10(self) -> Vec3:
        return self.__controllers[9]

    @property
    def p11(self) -> Vec3:
        return self.__controllers[10]

    @property
    def p12(self) -> Vec3:
        return self.__controllers[11]

    ###########################################
    @property
    def p13(self) -> Vec3:
        return self.__controllers[12]

    @property
    def p14(self) -> Vec3:
        return self.__controllers[13]

    @property
    def p15(self) -> Vec3:
        return self.__controllers[14]

    @property
    def p16(self) -> Vec3:
        return self.__controllers[15]

    ###########################################
    @p1.setter
    def p1(self, p: Vec3) -> None:
        self.__controllers[0] = p
        self.__update_mesh()

    @p2.setter
    def p2(self, p: Vec3) -> None:
        self.__controllers[1] = p
        self.__update_mesh()

    @p3.setter
    def p3(self, p: Vec3) -> None:
        self.__controllers[2] = p
        self.__update_mesh()

    @p4.setter
    def p4(self, p: Vec3) -> None:
        self.__controllers[3] = p
        self.__update_mesh()

    ###########################################
    @p5.setter
    def p5(self, p: Vec3) -> None:
        self.__controllers[4] = p
        self.__update_mesh()

    @p6.setter
    def p6(self, p: Vec3) -> None:
        self.__controllers[5] = p
        self.__update_mesh()

    @p7.setter
    def p7(self, p: Vec3) -> None:
        self.__controllers[6] = p
        self.__update_mesh()

    @p8.setter
    def p8(self, p: Vec3) -> None:
        self.__controllers[7] = p
        self.__update_mesh()

    ###########################################
    @p9.setter
    def p9(self, p: Vec3) -> None:
        self.__controllers[8] = p
        self.__update_mesh()

    @p10.setter
    def p10(self, p: Vec3) -> None:
        self.__controllers[9] = p
        self.__update_mesh()

    @p11.setter
    def p11(self, p: Vec3) -> None:
        self.__controllers[10] = p
        self.__update_mesh()

    @p12.setter
    def p12(self, p: Vec3) -> None:
        self.__controllers[11] = p
        self.__update_mesh()

    ###########################################
    @p13.setter
    def p13(self, p: Vec3) -> None:
        self.__controllers[12] = p
        self.__update_mesh()

    @p14.setter
    def p14(self, p: Vec3) -> None:
        self.__controllers[13] = p
        self.__update_mesh()

    @p15.setter
    def p15(self, p: Vec3) -> None:
        self.__controllers[14] = p
        self.__update_mesh()

    @p16.setter
    def p16(self, p: Vec3) -> None:
        self.__controllers[15] = p
        self.__update_mesh()
    ###########################################
