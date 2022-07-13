import numpy as np
from OpenGL.GL import *

from models.tris_mesh import TrisMesh
from open_gl.gpu_buffer import GPUBuffer


class MeshGL(object):
    __vao_instances = {}

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
               f"vbo  :\n{self.__vbo}"\
               f"ibo  :\n{self.__ibo}"\
               f"mesh :\n{self.__mesh}"

    def __init__(self, mesh: TrisMesh = None):
        self.__mesh: TrisMesh = mesh
        self.__vao: int = 0
        self.__vbo: GPUBuffer = None
        self.__ibo: GPUBuffer = None
        if not(self.__mesh is None):
            self.__create_gpu_buffers()

    @property
    def mesh(self) -> TrisMesh:
        return self.__mesh

    @mesh.setter
    def mesh(self, m: TrisMesh) -> None:
        if not(self.__mesh is None):
            self.delete_mesh()
        self.__mesh = m
        self.__create_gpu_buffers()

    def __del__(self):
        # super(Mesh, self).__del__()
        self.delete_mesh()

    def __create_gpu_buffers(self) -> bool:

        if self.__mesh.vertices_count == 0:
            return False
        if self.__mesh.faces_count == 0:
            return False
        if self.__vao == 0:
            self.__vao = glGenVertexArrays(1)

        self.bind()

        vertices = self.__vertex_array_data()

        indices = self.__index_array_data()

        if self.__ibo is None:
            self.__ibo = GPUBuffer(len(indices), int(indices.nbytes / len(indices)), GL_ELEMENT_ARRAY_BUFFER)
        self.__ibo.load_buffer_data(indices)

        if self.__vbo is None:
            self.__vbo = GPUBuffer(len(vertices), int(vertices.nbytes / len(vertices)))
        self.__vbo.load_buffer_data(vertices)
        ptr = 0
        if self.__mesh.vertices_count != 0:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(ptr))
            ptr += self.__mesh.vertices_count * 3 * 4

        if self.__mesh.normals_count != 0:
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(ptr))
            ptr += self.__mesh.normals_count * 3 * 4

        if self.__mesh.uvs_count != 0:
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(ptr))

        return True

    def delete_mesh(self):
        glDeleteVertexArrays(1, np.ndarray([self.__vao]))
        self.__vbo.delete_buffer()
        self.__ibo.delete_buffer()
        if self.__vao in MeshGL.__vao_instances:
            del MeshGL.__vao_instances[self.__vao]
        self.__vao = 0

    def clean_up(self):
        self.__mesh.clean_up()
        self.__vbo.delete_buffer()
        self.__ibo.delete_buffer()

    def __vertex_array_data(self) -> np.ndarray:
        size_ = self.__mesh.vertices_count * 8
        v_data = np.zeros(size_, dtype=np.float32)
        unique_vert_id = {}
        for f in self.__mesh.faces:
            for pt in f.points:
                if not(pt[0] in unique_vert_id):
                    if pt[0] != -1:
                        v = self.__mesh.vertices[pt[0]]
                        idx = pt[0] * 3
                        v_data[idx + 0] = v.x
                        v_data[idx + 1] = v.y
                        v_data[idx + 2] = v.z

                    if pt[1] != -1:
                        n = self.__mesh.normals[pt[1]]
                        idx = self.__mesh.vertices_count * 3 + pt[0] * 3
                        v_data[idx + 0] = n.x
                        v_data[idx + 1] = n.y
                        v_data[idx + 2] = n.z

                    if pt[2] != -1:
                        uv = self.__mesh.uvs[pt[2]]
                        idx = self.__mesh.vertices_count * 6 + pt[0] * 2
                        v_data[idx + 0] = uv.x
                        v_data[idx + 1] = uv.y

                    unique_vert_id[pt[0]] = pt[0]

        return v_data

    def __index_array_data(self) -> np.ndarray:
        i_data = np.zeros(self.__mesh.faces_count * 3, dtype=np.uint32)
        idx: int = 0
        for f in self.__mesh.faces:
            i_data[idx + 0] = f.p_1
            i_data[idx + 1] = f.p_2
            i_data[idx + 2] = f.p_3
            idx += 3
        return i_data

    def bind(self):
        glBindVertexArray(self.__vao)

    def unbind(self):
        glBindVertexArray(0)

    def load_to_gpu(self) -> bool:
        return self.__create_gpu_buffers()

    def draw(self):
        self.bind()
        glDrawElements(GL_TRIANGLE_STRIP, self.__mesh.faces_count * 9, GL_UNSIGNED_INT, None)
