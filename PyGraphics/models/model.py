from models.tris_mesh import TrisMesh, BoundingBox
from transforms.transform import Transform
from materials.material import Material
from models.triangle import Triangle
from vmath.vectors import Vec3
import materials.material
import models.tris_mesh


class Model(object):
    def __init__(self, geometry_origin: str = "", material_origin: str = ""):
        self.__meshes: [TrisMesh] = models.tris_mesh.read_obj_mesh(geometry_origin)
        self.__materials: [Material] = materials.material.read_material(material_origin)
        self.__transform: Transform = Transform()
        self.__mesh_origin: str = ""
        self.__bbox: BoundingBox = BoundingBox()
        for m in self.__meshes:
            self.__bbox.update_bounds(m.bbox.min)
            self.__bbox.update_bounds(m.bbox.max)

    def __mesh_id_in_range(self, _id: int) -> bool:
        if _id < 0:
            return False

        if _id >= len(self.__meshes):
            return False

        return True

    def __material_id_in_range(self, _id: int) -> bool:
        if _id < 0:
            return False

        if _id >= len(self.__materials):
            return False

        return True

    @property
    def transform(self) -> Transform:
        return self.__transform

    @property
    def meshes(self) -> [TrisMesh]:
        return self.__meshes

    @property
    def materials(self):
        return self.__materials

    @property
    def bbox(self) -> BoundingBox:
        return self.__bbox

    @property
    def meshes_count(self) -> int:
        return len(self.__meshes)

    @property
    def materials_count(self) -> int:
        return len(self.__materials)

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

    def get_mesh(self, _id: int) -> TrisMesh:
        if not self.__mesh_id_in_range(_id):
            raise IndexError(f"no mesh with index: {_id}")
        return self.__meshes[_id]

    def get_material(self, _id: int) -> Material:
        if not self.__material_id_in_range(_id):
            raise IndexError(f"no material with index: {_id}")
        return self.__materials[_id]

    def add_mesh(self, mesh: TrisMesh) -> None:
        self.__bbox.update_bounds(mesh.bbox.max)
        self.__bbox.update_bounds(mesh.bbox.min)
        self.__meshes.append(mesh)

    def add_material(self, mat: Material) -> None:
        self.__materials.append(mat)

    def get_vert_local_space(self, vert_id, mesh_id: int = 0):
        if not self.__mesh_id_in_range(mesh_id):
            raise IndexError(f"no mesh with index: {mesh_id}")
        return self.__meshes[mesh_id].vertices[vert_id]

    def get_normal_local_space(self, normal_id, mesh_id: int = 0):
        if not self.__mesh_id_in_range(mesh_id):
            raise IndexError(f"no mesh with index: {mesh_id}")
        return self.__meshes[mesh_id].normals[normal_id]

    def get_vert_world_space(self, vert_id, mesh_id: int = 0):
        return self.__transform.transform_vect(self.get_vert_local_space(vert_id, mesh_id), 1)

    def get_normal_world_space(self, normal_id, mesh_id: int = 0):
        n: Vec3 = self.__transform.transform_vect(self.get_normal_local_space(normal_id, mesh_id), 0)
        n.normalize()
        return n

    def tris_local_space(self, mesh_id: int, tris_id: int) -> Triangle:
        if not self.__mesh_id_in_range(mesh_id):
            raise IndexError(f"no mesh with index {mesh_id}")

        return self.__meshes[mesh_id].get_triangle(tris_id)

    def tris_world_space(self, mesh_id: int, tris_id: int) -> Triangle:
        if not self.__mesh_id_in_range(mesh_id):
            raise IndexError(f"no mesh with index: {mesh_id}")

        tris: Triangle = self.__meshes[mesh_id].get_triangle(tris_id)
        tris.transform(self.__transform)
        return tris

    def triangles_local_space(self, mesh_id: int):
        if not self.__mesh_id_in_range(mesh_id):
            raise IndexError(f"no mesh with index: {mesh_id}")
        tris_id: int = 0
        while tris_id < self.__meshes[mesh_id].faces_count:
            yield self.__meshes[mesh_id].get_triangle(tris_id)
            tris_id += 1

    def triangles_world_space(self, mesh_id: int):
        if not self.__mesh_id_in_range(mesh_id):
            raise IndexError(f"no mesh with index: {mesh_id}")
        tris_id: int = 0
        while tris_id < self.__meshes[mesh_id].faces_count:
            tris: Triangle = self.__meshes[mesh_id].get_triangle(tris_id)
            tris.transform(self.__transform)
            yield tris
            tris_id += 1