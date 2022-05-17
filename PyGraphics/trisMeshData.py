from transform import Transform
from mathUtils import Vec3, Vec2


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


class TrisMeshData(object):
    def __init__(self):
        self.transform_: Transform = Transform()
        self.uvs: [Vec2] = []
        self.vertices: [Vec3] = []
        self.normals: [Vec3] = []
        self.faces: [Face] = []
        self.max_: Vec3 = Vec3(-1e12, -1e12, -1e12)
        self.min_: Vec3 = Vec3(1e12, 1e12, 1e12)

    @property
    def transformation(self) -> Transform:
        return self.transform_

    @property
    def max_local_space(self) -> Vec3:
        return self.max_

    @property
    def min_local_space(self) -> Vec3:
        return self.min_

    @property
    def size_local_space(self) -> Vec3:
        return self.max_ - self.min_

    @property
    def center_local_space(self) -> Vec3:
        return (self.max_ + self.min_) * 0.5

    @property
    def center_world_space(self) -> Vec3:
        return self.transformation.transform_vect(self.center_local_space, 1)

    @property
    def min_world_space(self) -> Vec3:
        return self.transformation.transform_vect(self.min_, 1)

    @property
    def max_world_space(self) -> Vec3:
        return self.transformation.transform_vect(self.max_, 1)

    @property
    def size_world_space(self) -> Vec3:
        return self.transformation.transform_vect(self.size_local_space, 0)

    def clean_up(self):
        if len(self.vertices) == 0:
            return
        del self.uvs
        del self.vertices
        del self.normals
        del self.faces

        self.uvs: [Vec2] = []
        self.vertices: [Vec3] = []
        self.normals: [Vec3] = []
        self.faces: [Face] = []

    def write(self):
        if len(self.vertices) == 0:
            return
        print("Obj model")
        print("vertices number: ", len(self.vertices))
        for v in self.vertices:
            print("v  ", v)
        print("normals number: ", len(self.normals))
        for n in self.normals:
            print("n ", n)
        print("uvs number: ", len(self.uvs))
        for vt in self.uvs:
            print("vt ", vt)
        print("faces number: ", len(self.faces))
        for f in self.faces:
            print(f)

    def read(self, path: str):
        self.clean_up()
        file = open(path)
        tmp: [str]
        tmp2: [str]
        lines: [str] = []
        for str_ in file:
            lines.append(str_)
        file.close()

        for i in range(len(lines)):
            if len(lines[i]) == 0:
                continue

            tmp = lines[i].split(" ")

            if len(tmp) == 0:
                continue

            if tmp[0] == "vn":
                self.normals.append(Vec3(float(tmp[1]), float(tmp[2]), float(tmp[3])))
            if tmp[0] == "v":
                v = Vec3(float(tmp[1]), float(tmp[2]), float(tmp[3]))
                # update max bound
                if v.x > self.max_.x:
                    self.max_.x = v.x
                if v.y > self.max_.y:
                    self.max_.y = v.y
                if v.z > self.max_.z:
                    self.max_.z = v.z
                # update min bound
                if v.x < self.min_.x:
                    self.min_.x = v.x
                if v.y < self.min_.y:
                    self.min_.y = v.y
                if v.z < self.min_.z:
                    self.min_.z = v.z

                self.vertices.append(v)
            if tmp[0] == "vt":
                self.uvs.append(Vec2(float(tmp[1]), float(tmp[2])))
            if tmp[0] == "f":
                tmp2 = tmp[1].split("/")
                face_ = Face()
                face_.p_1 = int(tmp2[0]) - 1
                face_.uv1 = int(tmp2[1]) - 1
                face_.n_1 = int(tmp2[2]) - 1

                tmp2 = tmp[2].split("/")
                face_.p_2 = int(tmp2[0]) - 1
                face_.uv2 = int(tmp2[1]) - 1
                face_.n_2 = int(tmp2[2]) - 1

                tmp2 = tmp[3].split("/")
                face_.p_3 = int(tmp2[0]) - 1
                face_.uv3 = int(tmp2[1]) - 1
                face_.n_3 = int(tmp2[2]) - 1
                self.faces.append(face_)

    def get_vert_local_space(self, vert_id):
        return self.vertices[vert_id]

    def get_normal_local_space(self, normal_id):
        return self.normals[normal_id]

    def get_vert_world_space(self, vert_id):
        return self.transform_.transform_vect(self.get_vert_local_space(vert_id), 1)

    def get_normal_world_space(self, normal_id):
        n: Vec3 = self.transform_.transform_vect(self.get_normal_local_space(normal_id), 0)
        n.normalize()
        return n


def create_plane(height: float = 1.0, width: float = 1.0, rows: int = 10, cols: int = 10) -> TrisMeshData:
    if rows < 2:
        rows = 2
    if cols < 2:
        cols = 2
    points_n: int = cols * rows
    x: float
    z: float
    mesh: TrisMeshData = TrisMeshData()
    normal: Vec3 = Vec3(0, 1, 0)
    for index in range(0, points_n):
        col = index % cols
        row = int(index / cols)
        x = width * ((cols - 1) / 2.0 - col) / (cols - 1.0)
        z = height * ((cols - 1) / 2.0 - row) / (cols - 1.0)
        mesh.vertices.append(Vec3(x, 0, z))
        mesh.uvs.append(Vec2(col * 1.0 / cols, row * 1.0 / cols))
        mesh.normals.append(normal)
        if (index + 1) % cols == 0:
            continue  # пропускаем последю
        if rows - 1 == row:
            continue
        f = Face()
        f.index1(index)
        f.index2(index + 1)
        f.index3(index + cols)
        mesh.faces.append(f)
        f = Face()
        f.index1(index + cols)
        f.index2(index + 1)
        f.index3(index + cols + 1)
        mesh.faces.append(f)
    return mesh
