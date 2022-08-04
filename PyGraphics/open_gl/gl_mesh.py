from open_gl.gpu_buffer import GPUBuffer
from models.tris_mesh import TrisMesh
from utils.bit_set import BitSet
from models import tris_mesh
from OpenGL.GL import *
import numpy as np


class MeshGL(object):

    VerticesAttribute = 0
    NormalsAttribute = 1
    TangentsAttribute = 2
    UVsAttribute = 3
    TrianglesAttribute = 4

    __vao_instances = {}

    @staticmethod
    def create_plane_gl(height: float = 1.0, width: float = 1.0, rows: int = 10, cols: int = 10):
        return MeshGL(tris_mesh.create_plane(height, width, rows, cols))

    @staticmethod
    def vao_enumerate():
        print(MeshGL.__vao_instances.items())
        for buffer in MeshGL.__vao_instances.items():
            yield buffer[1]

    @staticmethod
    def vao_delete_all():
        while len(MeshGL.__vao_instances) != 0:
            item = MeshGL.__vao_instances.popitem()
            item[1].delete_mesh()

    def __repr__(self):
        return f"vao  : {self.__vao}\n" \
               f"vbo  :\n{self.__vbo}" \
               f"ibo  :\n{self.__ibo}" \


    def __init__(self, mesh: TrisMesh = None):
        self.__vao: int = 0
        self.__vbo: GPUBuffer = None
        self.__ibo: GPUBuffer = None
        self.__vertex_attributes: BitSet = BitSet()
        self.__vertex_byte_size = 0
        if not(mesh is None):
            self.__create_gpu_buffers(mesh)

    @property
    def has_vertices(self):
        return self.__vertex_attributes.is_bit_set(MeshGL.VerticesAttribute)

    @property
    def has_normals(self):
        return self.__vertex_attributes.is_bit_set(MeshGL.NormalsAttribute)

    @property
    def has_uvs(self):
        return self.__vertex_attributes.is_bit_set(MeshGL.UVsAttribute)

    @property
    def has_tangents(self):
        return self.__vertex_attributes.is_bit_set(MeshGL.TangentsAttribute)

    @property
    def has_triangles(self):
        return self.__vertex_attributes.is_bit_set(MeshGL.TrianglesAttribute)

    def set_mesh(self, m: TrisMesh) -> None:
        if self.__vao != 0:
            self.delete_mesh()
        self.__create_gpu_buffers(m)

    def __del__(self):
        self.delete_mesh()

    def __gen_vao(self):
        if self.__vao == 0:
            self.__vao = glGenVertexArrays(1)
            MeshGL.__vao_instances[self.__vao] = self
            self.__vertex_byte_size = 0

        self.bind()

    def __create_gpu_buffers(self, mesh: TrisMesh) -> bool:

        if mesh.vertices_count == 0:
            return False
        if mesh.faces_count == 0:
            return False

        self.__vertex_attributes.set_bit(MeshGL.VerticesAttribute)
        self.__vertex_attributes.set_bit(MeshGL.NormalsAttribute)
        self.__vertex_attributes.set_bit(MeshGL.UVsAttribute)
        self.__vertex_attributes.set_bit(MeshGL.TrianglesAttribute)

        self.__gen_vao()

        self.vertices_array = mesh.vertex_array_data

        self.indices_array = mesh.index_array_data

        self.set_attributes(self.__vertex_attributes)

        return True

    def delete_mesh(self):
        glDeleteVertexArrays(1, np.ndarray([self.__vao]))
        self.__vbo.delete_buffer()
        self.__ibo.delete_buffer()
        if self.__vao in MeshGL.__vao_instances:
            del MeshGL.__vao_instances[self.__vao]
        self.__vao = 0

    @property
    def indices_array(self) -> np.ndarray:
        return self.__ibo.read_back_data()

    @indices_array.setter
    def indices_array(self, indices: np.ndarray) -> None:
        self.__gen_vao()
        if self.__ibo is None:
            self.__ibo = GPUBuffer(len(indices), int(indices.nbytes / len(indices)), GL_ELEMENT_ARRAY_BUFFER)
        self.__ibo.load_buffer_data(indices)

    @property
    def vertices_array(self) -> np.ndarray:
        return self.__vbo.read_back_data()

    @vertices_array.setter
    def vertices_array(self, vertices: np.ndarray) -> None:
        self.__gen_vao()
        if self.__vbo is None:
            self.__vbo = GPUBuffer(len(vertices), int(vertices.nbytes / len(vertices)))
        self.__vbo.load_buffer_data(vertices)

    def set_attributes(self, attributes: BitSet):

        if self.__vao == 0:
            return

        if self.__vbo is None:
            return

        self.__gen_vao()

        self.__vertex_byte_size = 0

        self.__vertex_attributes = attributes

        if attributes.is_bit_set(MeshGL.VerticesAttribute):
            self.__vertex_byte_size += 3

        if attributes.is_bit_set(MeshGL.NormalsAttribute):
            self.__vertex_byte_size += 3

        if attributes.is_bit_set(MeshGL.TangentsAttribute):
            self.__vertex_byte_size += 3

        if attributes.is_bit_set(MeshGL.UVsAttribute):
            self.__vertex_byte_size += 2

        ptr = 0

        attr_i = 0

        d_ptr = int(self.__vbo.filling / self.__vertex_byte_size)

        self.__vbo.bind()

        if self.has_vertices:
            glEnableVertexAttribArray(attr_i)
            glVertexAttribPointer(attr_i, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(ptr))
            ptr += d_ptr * 12
            attr_i += 1

        if self.has_normals:
            glEnableVertexAttribArray(attr_i)
            glVertexAttribPointer(attr_i, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(ptr))
            ptr += d_ptr * 12
            attr_i += 1

        if self.has_tangents:
            glEnableVertexAttribArray(attr_i)
            glVertexAttribPointer(attr_i, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(ptr))
            ptr += d_ptr * 12
            attr_i += 1

        if self.has_uvs:
            glEnableVertexAttribArray(attr_i)
            glVertexAttribPointer(attr_i, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(ptr))

    def clean_up(self):
        self.__vbo.delete_buffer()
        self.__ibo.delete_buffer()

    def bind(self):
        glBindVertexArray(self.__vao)

    def unbind(self):
        glBindVertexArray(0)

    def draw(self):
        if self.__vertex_attributes.is_empty:
            return
        self.bind()
        glDrawElements(GL_TRIANGLES, self.__ibo.filling, GL_UNSIGNED_INT, None)
