from OpenGL.GL import *

from models.trisMesh import TrisMesh


class Mesh(TrisMesh):
    def __init__(self):
        super().__init__()
        self.__vao: int = glGenVertexArrays(1)
        glBindVertexArray(self.__vao)
        self.__vbo: int = glGenBuffers(1)
        self.__ibo: int = glGenBuffers(1)
        self.__attribytes: int = -1

    def bind(self):
        pass

    def load_to_gpu(self):
        glBindVertexArray(self.__vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.__vbo)

        if self.vertices_count != 0:
            glBufferSubData(GL_ARRAY_BUFFER, self.vertices_count * 3 * 4, self._vertices, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))

        if self.normals_count != 0:
            glBufferSubData(GL_ARRAY_BUFFER, self.vertices_count * 3 * 4, self._vertices, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))


