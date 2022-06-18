import re

from models.triangle import Triangle
from vmath.mathUtils import Vec3, Vec2


class Face:
    def index1(self, index):
        self.p_1: int = index
        self.uv1: int = index
        self.n_1: int = index

    def index2(self, index):
        self.p_2: int = index
        self.uv2: int = index
        self.n_2: int = index

    def index3(self, index):
        self.p_3: int = index
        self.uv3: int = index
        self.n_3: int = index

    def __init__(self):
        self.p_1: int = -1
        self.uv1: int = -1
        self.n_1: int = -1
        self.p_2: int = -1
        self.uv2: int = -1
        self.n_2: int = -1
        self.p_3: int = -1
        self.uv3: int = -1
        self.n_3: int = -1

    def __repr__(self):
        res: str = "<face "
        res += "%s/%s/%s" % (self.p_1, self.uv1, self.n_1)
        res += "%s/%s/%s" % (self.p_2, self.uv2, self.n_2)
        res += "%s/%s/%s" % (self.p_3, self.uv3, self.n_3)
        res += ">"
        return res

    def __str__(self):
        res: str = "f ["
        res += "%s/%s/%s " % (self.p_1, self.uv1, self.n_1)
        res += "%s/%s/%s " % (self.p_2, self.uv2, self.n_2)
        res += "%s/%s/%s]" % (self.p_3, self.uv3, self.n_3)
        return res


class BoundingBox(object):
    def __init__(self):
        self.__max: Vec3 = Vec3(-1e12, -1e12, -1e12)
        self.__min: Vec3 = Vec3(1e12, 1e12, 1e12)

    def update_bounds(self, v: Vec3) -> None:
        if v.x > self.__max.x:
            self.__max.x = v.x
        if v.y > self.__max.y:
            self.__max.y = v.y
        if v.z > self.__max.z:
            self.__max.z = v.z
        # update min bound
        if v.x < self.__min.x:
            self.__min.x = v.x
        if v.y < self.__min.y:
            self.__min.y = v.y
        if v.z < self.__min.z:
            self.__min.z = v.z

    @property
    def min(self) -> Vec3:
        return self.__min

    @property
    def max(self) -> Vec3:
        return self.__max

    @property
    def size(self) -> Vec3:
        return self.__max - self.__min

    @property
    def center(self) -> Vec3:
        return (self.__max + self.__min) * 0.5


class TrisMesh(object):

    def __init__(self):
        self.name: str = ""
        self.__vertices: [Vec3] = []
        self.__normals: [Vec3] = []
        self.__uvs: [Vec2] = []
        self.__faces: [Face] = []
        self.__bbox: BoundingBox = BoundingBox()

    def __str__(self):
        res: str = f"mesh: {self.name}\n"
        res += f"max: {self.__bbox.max}\n"
        res += f"min: {self.__bbox.min}\n"
        res += "vertices: \n"
        for v in self.__vertices:
            res += f"{v}\n"
        res += "normals: \n"
        for v in self.__normals:
            res += f"{v}\n"
        res += "uvs: \n"
        for v in self.__uvs:
            res += f"{v}\n"
        for f in self.__faces:
            res += f"{f}\n"
        return res

    def __repr__(self):
        res: str = f"<\nmesh: {self.name}\n"
        res += f"max: {self.__bbox.max}\n"
        res += f"min: {self.__bbox.min}\n"
        res += "vertices: \n"
        for v in self.__vertices:
            res += f"{v}\n"
        res += "normals: \n"
        for v in self.__normals:
            res += f"{v}\n"
        res += "uvs: \n"
        for v in self.__uvs:
            res += f"{v}\n"
        for f in self.__faces:
            res += f"{f}\n"
        res += "\n>"
        return res

    def __iter__(self):
        if len(self.__faces) == 0:
            raise StopIteration
        self.__iter_face_i: int = -1
        return self

    def __next__(self) -> Triangle:

        self.__iter_face_i += 1

        if self.__iter_face_i >= len(self.__faces):
            self.__iter_face_i = -1
            raise StopIteration

        f: Face = self.__faces[self.__iter_face_i]
        try:
            return Triangle(self.__vertices[f.p_1], self.__vertices[f.p_2], self.__vertices[f.p_3],
                            self.__normals[f.n_1], self.__normals[f.n_2], self.__normals[f.n_3],
                            self.__uvs[f.uv1], self.__uvs[f.uv2], self.__uvs[f.uv3])
        except IndexError:
            print("bad  triangle info \n")
            print("n_points = %s, n_normals = %s, n_uvs = %s " % (str(self.vertices_count), str(self.normals_count),
                                                                  str(self.uvs_count)))
            print(f)

    def get_triangle(self, tris_id: int):

        if len(self.__faces) <= tris_id or tris_id < 0:
            raise IndexError(f"no face with index %s in mesh %s " % (str(tris_id), self.name))

        f: Face = self.__faces[tris_id]

        return Triangle(self.__vertices[f.p_1], self.__vertices[f.p_2], self.__vertices[f.p_3],
                        self.__normals[f.n_1], self.__normals[f.n_2], self.__normals[f.n_3],
                        self.__uvs[f.uv1], self.__uvs[f.uv2], self.__uvs[f.uv3])

    def set_vertex(self, i_id: int, v: Vec3) -> None:
        if i_id < 0:
            return
        if i_id >= self.vertices_count:
            return
        self.__bbox.update_bounds(v)
        self.__vertices[i_id] = v

    def set_normal(self, i_id: int, v: Vec3) -> None:
        if i_id < 0:
            return
        if i_id >= self.normals_count:
            return
        self.__normals[i_id] = v

    def set_uv(self, i_id: int, v: Vec2) -> None:
        if i_id < 0:
            return
        if i_id >= self.uvs_count:
            return
        self.__uvs[i_id] = v

    @property
    def faces_count(self) -> int:
        return len(self.__faces)

    @property
    def vertices_count(self) -> int:
        return len(self.__vertices)

    @property
    def uvs_count(self) -> int:
        return len(self.__uvs)

    @property
    def normals_count(self) -> int:
        return len(self.__normals)

    @property
    def bbox(self) -> BoundingBox:
        return self.__bbox

    def append_vertex(self, v: Vec3) -> None:
        self.__bbox.update_bounds(v)
        self.__vertices.append(v)

    def append_normal(self, v: Vec3) -> None:
        self.__normals.append(v)

    def append_uv(self, v: Vec2) -> None:
        self.__uvs.append(v)

    def append_face(self, f: Face) -> None:
        self.__faces.append(f)

    def clean_up(self):
        if len(self.__vertices) == 0:
            return
        del self.__uvs
        del self.__vertices
        del self.__normals
        del self.__faces

        self.__uvs: [Vec2] = []
        self.__vertices: [Vec3] = []
        self.__normals: [Vec3] = []
        self.__faces: [Face] = []


def read_obj_mesh(path: str) -> [TrisMesh]:
    try:
        with open(path, mode='r') as file:

            tmp: [str]
            tmp2: [str]
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
        print("file \"%s\" not found" % path)
        return []


def create_plane(height: float = 1.0, width: float = 1.0, rows: int = 10, cols: int = 10) -> TrisMesh:
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
        x = width * ((cols - 1) / 2.0 - col) / (cols - 1.0)
        z = height * ((cols - 1) / 2.0 - row) / (cols - 1.0)
        mesh.append_vertex(Vec3(x, 0, z))
        mesh.append_uv(Vec2(col * 1.0 / cols, row * 1.0 / cols))
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
    return mesh
