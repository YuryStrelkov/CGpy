from typing import Tuple, List

from core.transforms.transform import Transform
from core.tris_mesh.triangle import Triangle
from core.bounding_box import BoundingBox
from core.vectors import Vec3, Vec2
import numpy as np
import re


class Face:

    __slots__ = "__p_1", "__p_2", "__p_3", "__uv1", "__uv2", "__uv3", "__n_1", "__n_2", "__n_3"

    def __init__(self):
        self.__p_1: int = -1
        self.__uv1: int = -1
        self.__n_1: int = -1
        self.__p_2: int = -1
        self.__uv2: int = -1
        self.__n_2: int = -1
        self.__p_3: int = -1
        self.__uv3: int = -1
        self.__n_3: int = -1

    def __str__(self):
        return f"{{" \
               f"\t\"p_1\": {self.p_1:4}, \"uv1\": {self.uv1:4}, \"n_1\": {self.n_1:4}," \
               f"\t\"p_2\": {self.p_2:4}, \"uv2\": {self.uv2:4}, \"n_2\": {self.n_2:4}," \
               f"\t\"p_3\": {self.p_3:4}, \"uv3\": {self.uv3:4}, \"n_3\": {self.n_3:4}" \
               f"}}"

    @property
    def points(self):
        yield self.pt_1
        yield self.pt_2
        yield self.pt_3

    @property
    def pt_1(self) -> Tuple[int, int, int]:
        return self.__p_1, self.__n_1, self.__uv1

    @property
    def pt_2(self) -> Tuple[int, int, int]:
        return self.__p_2, self.__n_2, self.__uv2

    @property
    def pt_3(self) -> Tuple[int, int, int]:
        return self.__p_3, self.__n_3, self.__uv3

    @property
    def p_1(self) -> int:
        return self.__p_1

    @p_1.setter
    def p_1(self, val: int) -> None:
        self.__p_1 = max(0, val)

    @property
    def n_1(self) -> int:
        return self.__n_1

    @n_1.setter
    def n_1(self, val: int) -> None:
        self.__n_1 = max(0, val)

    @property
    def uv1(self) -> int:
        return self.__uv1

    @uv1.setter
    def uv1(self, val: int) -> None:
        self.__uv1 = max(0, val)

    @property
    def p_2(self) -> int:
        return self.__p_2

    @p_2.setter
    def p_2(self, val: int) -> None:
        self.__p_2 = max(0, val)

    @property
    def n_2(self) -> int:
        return self.__n_2

    @n_2.setter
    def n_2(self, val: int) -> None:
        self.__n_2 = max(0, val)

    @property
    def uv2(self) -> int:
        return self.__uv2

    @uv2.setter
    def uv2(self, val: int) -> None:
        self.__uv2 = max(0, val)

    @property
    def p_3(self) -> int:
        return self.__p_3

    @p_3.setter
    def p_3(self, val: int) -> None:
        self.__p_3 = max(0, val)

    @property
    def n_3(self) -> int:
        return self.__n_3

    @n_3.setter
    def n_3(self, val: int) -> None:
        self.__n_3 = max(0, val)

    @property
    def uv3(self) -> int:
        return self.__uv3

    @uv3.setter
    def uv3(self, val: int) -> None:
        self.__uv3 = max(0, val)

    def index1(self, index) -> None:
        self.__p_1: int = index
        self.__uv1: int = index
        self.__n_1: int = index

    def index2(self, index) -> None:
        self.__p_2: int = index
        self.__uv2: int = index
        self.__n_2: int = index

    def index3(self, index) -> None:
        self.__p_3: int = index
        self.__uv3: int = index
        self.__n_3: int = index


class TrisMesh:

    __slots__ = "name", "_vertices", "_normals", "_uvs", "_faces", "_bbox"

    def __init__(self):
        self.name: str = "no name"
        self._vertices: List[Vec3] = []
        self._normals: List[Vec3] = []
        self._uvs: List[Vec2] = []
        self._faces: List[Face] = []
        self._bbox: BoundingBox = BoundingBox()

    def __str__(self):
        new_l = ",\n\t\t"
        return f"{{\n" \
               f"\t\"name\"     : \"{self.name}\",\n" \
               f"\t\"unique_id\": {self.unique_id},\n" \
               f"\t\"bounds\"   :\n{self._bbox},\n" \
               f"\t\"vertices\" :\n\t[\n\t\t{new_l.join(str(v) for v in self._vertices)}\n\t],\n" \
               f"\t\"normals\"  :\n\t[\n\t\t{new_l.join(str(v) for v in self._normals)}\n\t],\n" \
               f"\t\"uvs\"      :\n\t[\n\t\t{new_l.join(str(v) for v in self._uvs)}\n\t],\n" \
               f"\t\"faces\"    :\n\t[\n\t\t{new_l.join(str(v) for v in self._faces)}\n\t]\n" \
               f"}}"

    @property
    def unique_id(self) -> int:
        return id(self)

    @property
    def vertex_array_data(self) -> np.ndarray:
        size_ = self.vertices_count * 8
        v_data = np.zeros(size_, dtype=np.float32)
        unique_vert_id = {}
        for f in self.faces:
            for pt in f.points:
                if not(pt[0] in unique_vert_id):
                    if pt[0] != -1:
                        v = self.vertices[pt[0]]
                        idx = pt[0] * 3
                        v_data[idx + 0] = v.x
                        v_data[idx + 1] = v.y
                        v_data[idx + 2] = v.z

                    if pt[1] != -1:
                        n = self.normals[pt[1]]
                        idx = self.vertices_count * 3 + pt[0] * 3
                        v_data[idx + 0] = n.x
                        v_data[idx + 1] = n.y
                        v_data[idx + 2] = n.z

                    if pt[2] != -1:
                        uv = self.uvs[pt[2]]
                        idx = self.vertices_count * 6 + pt[0] * 2
                        v_data[idx + 0] = uv.x
                        v_data[idx + 1] = uv.y
                    unique_vert_id[pt[0]] = pt[0]

        return v_data

    @property
    def index_array_data(self) -> np.ndarray:
        i_data = np.zeros(self.faces_count * 3, dtype=np.uint32)
        idx: int = 0
        for f in self.faces:
            i_data[idx + 0] = f.p_1
            i_data[idx + 1] = f.p_2
            i_data[idx + 2] = f.p_3
            idx += 3
        return i_data

    @property
    def vertices(self) -> List[Vec3]:
        return self._vertices

    @property
    def normals(self) -> List[Vec3]:
        return self._normals

    @property
    def uvs(self) -> List[Vec2]:
        return self._uvs

    @property
    def faces(self) -> List[Face]:
        return self._faces

    def set_vertex(self, i_id: int, v: Vec3) -> None:
        if i_id < 0:
            return
        if i_id >= self.vertices_count:
            return
        self._bbox.encapsulate(v)
        self._vertices[i_id] = v

    def set_normal(self, i_id: int, v: Vec3) -> None:
        if i_id < 0:
            return
        if i_id >= self.normals_count:
            return
        self._normals[i_id] = v

    def set_uv(self, i_id: int, v: Vec2) -> None:
        if i_id < 0:
            return
        if i_id >= self.uvs_count:
            return
        self._uvs[i_id] = v

    @property
    def faces_count(self) -> int:
        return len(self._faces)

    @property
    def vertices_count(self) -> int:
        return len(self._vertices)

    @property
    def uvs_count(self) -> int:
        return len(self._uvs)

    @property
    def normals_count(self) -> int:
        return len(self._normals)

    @property
    def bbox(self) -> BoundingBox:
        return self._bbox

    def append_vertex(self, v: Vec3) -> None:
        self._bbox.encapsulate(v)
        self._vertices.append(v)

    def append_normal(self, v: Vec3) -> None:
        self._normals.append(v)

    def append_uv(self, v: Vec2) -> None:
        self._uvs.append(v)

    def append_face(self, f: Face) -> None:
        self._faces.append(f)

    def clean_up(self) -> None:
        if len(self._vertices) == 0:
            return
        del self._uvs
        del self._vertices
        del self._normals
        del self._faces

        self._uvs: [Vec2] = []
        self._vertices: [Vec3] = []
        self._normals: [Vec3] = []
        self._faces: [Face] = []

    def transform_mesh(self, transform: Transform = None) -> None:
        for i in range(len(self._vertices)):
            self._vertices[i] = transform.transform_vect(self._vertices[i], 1.0)

        for i in range(len(self._normals)):
            self._normals[i] = transform.transform_vect(self._normals[i], 0.0)

    def get_triangle(self, tris_id: int) -> Triangle:

        if len(self._faces) <= tris_id or tris_id < 0:
            raise IndexError(f"no face with index: {tris_id} in mesh: {self.name}")

        f: Face = self._faces[tris_id]

        return Triangle(self._vertices[f.p_1], self._vertices[f.p_2], self._vertices[f.p_3],
                        self._normals[f.n_1], self._normals[f.n_2], self._normals[f.n_3],
                        self._uvs[f.uv1], self._uvs[f.uv2], self._uvs[f.uv3])


def read_obj_mesh(path: str) -> List[TrisMesh]:
    try:
        with open(path, mode='r') as file:

            tmp: List[str]
            tmp2: List[str]
            id_: int

            meshes: [TrisMesh] = []
            uv_shift: int = 0
            v__shift: int = 0
            n__shift: int = 0

            for str_ in file:

                line = (re.sub(r"[\n\t]*", "", str_))

                if len(line) == 0:
                    continue

                tmp = line.strip().split()

                id_ = len(tmp) - 1

                if id_ == -1:
                    continue

                if tmp[0] == "#":
                    if id_ == 0:
                        continue

                    if not (tmp[1] == "object"):
                        continue

                    mesh: TrisMesh = TrisMesh()
                    mesh.name = tmp[2]
                    meshes.append(mesh)
                    if len(meshes) == 1:
                        continue
                    uv_shift += meshes[len(meshes) - 2].uvs_count
                    v__shift += meshes[len(meshes) - 2].vertices_count
                    n__shift += meshes[len(meshes) - 2].normals_count
                    continue

                if tmp[0] == "o":
                    mesh: TrisMesh = TrisMesh()
                    mesh.name = tmp[1]
                    meshes.append(mesh)
                    if len(meshes) == 1:
                        continue
                    uv_shift += meshes[len(meshes) - 2].uvs_count
                    v__shift += meshes[len(meshes) - 2].vertices_count
                    n__shift += meshes[len(meshes) - 2].normals_count
                    continue

                if tmp[0] == "vn":
                    meshes[len(meshes) - 1].append_normal(
                        Vec3(float(tmp[id_ - 2]), float(tmp[id_ - 1]), float(tmp[id_])))
                    continue

                if tmp[0] == "v":
                    meshes[len(meshes) - 1].append_vertex(
                        Vec3(float(tmp[id_ - 2]), float(tmp[id_ - 1]), float(tmp[id_])))
                    continue

                if tmp[0] == "vt":
                    meshes[len(meshes) - 1].append_uv(Vec2(float(tmp[id_ - 1]), float(tmp[id_])))
                    continue

                if tmp[0] == "f":
                    tmp2 = tmp[1].strip().split("/")
                    face_ = Face()
                    face_.p_1 = int(tmp2[0]) - 1 - v__shift
                    face_.uv1 = int(tmp2[1]) - 1 - uv_shift
                    face_.n_1 = int(tmp2[2]) - 1 - n__shift

                    tmp2 = tmp[2].split("/")
                    face_.p_2 = int(tmp2[0]) - 1 - v__shift
                    face_.uv2 = int(tmp2[1]) - 1 - uv_shift
                    face_.n_2 = int(tmp2[2]) - 1 - n__shift

                    tmp2 = tmp[3].split("/")
                    face_.p_3 = int(tmp2[0]) - 1 - v__shift
                    face_.uv3 = int(tmp2[1]) - 1 - uv_shift
                    face_.n_3 = int(tmp2[2]) - 1 - n__shift
                    meshes[len(meshes) - 1].append_face(face_)
                    continue
            return meshes
    except IOError:
        print(f"file: \"{path}\" not found")
        return []


def create_plane(height: float = 1.0, width: float = 1.0, rows: int = 10,
                 cols: int = 10, transform: Transform = None) -> TrisMesh:
    if rows < 2:
        rows = 2
    if cols < 2:
        cols = 2
    points_n: int = cols * rows
    x: float
    z: float
    mesh: TrisMesh = TrisMesh()
    normal: Vec3 = Vec3(0, 1, 0)
    for index in range(0, points_n):
        col = index % cols
        row = int(index / cols)
        x = width * ((cols - 1) * 0.5 - col) / (cols - 1.0)
        z = height * ((cols - 1) * 0.5 - row) / (cols - 1.0)
        mesh.append_vertex(Vec3(x, 0, z))
        mesh.append_uv(Vec2(1.0 - col * 1.0 / (cols - 1), row * 1.0 / (cols - 1)))
        mesh.append_normal(normal)
        if (index + 1) % cols == 0:
            continue  # пропускаем последю
        if rows - 1 == row:
            continue
        f = Face()
        f.index1(index)
        f.index2(index + 1)
        f.index3(index + cols)
        mesh.append_face(f)
        f = Face()
        f.index1(index + cols)
        f.index2(index + 1)
        f.index3(index + cols + 1)
        mesh.append_face(f)
    if transform is not None:
        mesh.transform_mesh(transform)
    return mesh
