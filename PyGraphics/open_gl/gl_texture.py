from materials.material import Texture
from materials.rgb import RGB
from OpenGL.GL import *
import numpy as np


# TODO texture_cube
class TextureGL(Texture):

    __textures_instances = {}

    @staticmethod
    def textures_enumerate():
        for texture in TextureGL.__textures_instances.items():
            yield texture[1]

    @staticmethod
    def delete_all_textures():
        while len(TextureGL.__textures_instances) != 0:
            item = TextureGL.__textures_instances.popitem()
            item[1].delete_texture()

    def __init__(self, w: int = 100, h: int = 100, col: RGB = RGB(np.uint8(255), np.uint8(0), np.uint8(0))):
        super().__init__(w, h, 3, col)
        self.__id = 0
        self.__bind_target: GLenum = GL_TEXTURE_2D
        self.__load_data()

    @property
    def bind_target(self) -> GLenum:
        return self.__bind_target

    def repeat(self):
        self.bind()
        glTexParameteri(self.bind_target, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(self.bind_target, GL_TEXTURE_WRAP_T, GL_REPEAT)

    def mirrored_repeat(self):
        self.bind()
        glTexParameteri(self.bind_target, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
        glTexParameteri(self.bind_target, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT)

    def clamp_to_edge(self):
        self.bind()
        glTexParameteri(self.bind_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(self.bind_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def clamp_to_border(self):
        self.bind()
        glTexParameteri(self.bind_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(self.bind_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

    def nearest(self):
        self.bind()
        glTexParameteri(self.bind_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(self.bind_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def bi_linear(self):
        self.bind()
        glTexParameteri(self.bind_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(self.bind_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def tri_linear(self):
        self.bind()
        glTexParameteri(self.bind_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(self.bind_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR)

    def bind(self):
        glBindTexture(self.bind_target, self.__id)

    def delete_texture(self):
        glDeleteTextures(1, np.ndarray([self.__id]))
        if self.__id in TextureGL.__textures_instances:
            del TextureGL.__textures_instances[self.__id]
        self.__id = 0

    def __create(self):
        self.__id = glGenTextures(1)

    def __load_data(self):

        if self.__id == 0:
            self.__create()
        else:
            self.delete_texture()
            self.__create()

        if self.texture_byte_size == 0:
            self.delete_texture()
            return

        self.bind()
        glTexImage2D(self.bind_target, 0, GL_RGB, self.width, self.height, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, self.image_data)
        self.repeat()
        self.bi_linear()
        glGenerateMipmap(self.bind_target)

    def load(self, origin: str):

        super().load(origin)

        self.__load_data()



