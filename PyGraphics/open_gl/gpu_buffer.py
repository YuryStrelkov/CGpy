import numpy as np
from OpenGL.GL import *

from vmath.vectors import Vec3


class GPUBuffer(object):
    def __init__(self, cap: int,
                 item_b_size: int = 4,
                 bind_target: GLenum = GL_ARRAY_BUFFER,
                 usage_target: GLenum = GL_STATIC_DRAW):
        self.__id: int = 0
        self.__bind_target: GLenum = bind_target
        self.__usage_target: GLenum = usage_target
        self.__filling: int = 0
        self.__capacity: int = cap
        self.__item_byte_size: int = item_b_size
        self._create()

    def __str__(self):
        return f"id            :{self.buffer_id},\n" \
               f"capacity      :{self.capacity},\n" \
               f"filling       :{self.filling},\n" \
               f"item_byte_size:{self.item_byte_size},\n" \
               f"bind_target   :{self.bind_target},\n" \
               f"usage_target  :{self.usage_target},\n"\
               f"data          :{self.read_back_data()}"

    @property
    def bind_target(self) -> GLenum:
        return self.__bind_target

    @property
    def usage_target(self) -> GLenum:
        return self.__usage_target

    @property
    def filling(self) -> int:
        return self.__filling

    @property
    def capacity(self) -> int:
        return self.__capacity

    @property
    def item_byte_size(self) -> int:
        return self.__item_byte_size

    @property
    def buffer_id(self) -> int:
        return self.__id

    def _create(self) -> None:
        try:
            self.__id = glGenBuffers(1)
        except Exception as ex:
            print(f"GPUBuffer creation error:\n{ex.args}")
        self.bind()
        try:
            glBufferData(self.__bind_target, self.__capacity * self.__item_byte_size,
                         ctypes.c_int(0), self.__usage_target)
        except Exception as ex:
            print(f"GPUBuffer allocation error:\n{ex.args}")

    def bind(self) -> None:
        glBindBuffer(self.__bind_target, self.__id)

    def unbind(self) -> None:
        glBindBuffer(self.__bind_target, 0)

    def delete_buffer(self) -> None:
        if self.__id == -1:
            return
        glDeleteBuffers(1, self.__id)
        self.__id = 0
        self.__filling = 0
        self.__capacity = 0
        # self.__item_byte_size = 0

    def read_back_data(self, start_element: int = None, elements_number: int = None) -> list:
        if start_element is None:
            start_element = 0

        if elements_number is None:
            elements_number = self.capacity

        if start_element + elements_number > self.__capacity:
            return [0]

        self.bind()
        # b_data = np.zeros((1, elements_number), dtype=np.float32)
        b_data = glGetBufferSubData(self.__bind_target,
                                    self.__item_byte_size * start_element,
                                    self.__item_byte_size * elements_number)
        # print(b_data.astype('<f4'))
        return (b_data.view('<f4')).tolist()

    def resize(self, new_size: int) -> None:
        if self.__capacity == new_size:
            return
        if new_size <= 0:
            return
        self.bind()
        if self.__filling == 0:
            self.delete_buffer()
            self.__capacity = new_size
            self._create()
            return
        bid = glGenBuffers(1)
        glBindBuffer(self.__bind_target, new_size)
        glBufferData(self.__bind_target, new_size * self.__item_byte_size, 0, self.__usage_target)
        glBindBuffer(GL_COPY_READ_BUFFER, self.__id)
        glBindBuffer(GL_COPY_WRITE_BUFFER, bid)
        if self.__filling > new_size:
            self.__filling = new_size
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                            0, 0, self.__filling * self.__item_byte_size)
        glDeleteBuffers(1, self.__id)
        self.__id = bid
        self.__capacity = new_size

    def load_buffer_data(self, data: list) -> None:
        self.load_buffer_sub_data(0, data)

    def load_buffer_sub_data(self, start_offset, data: list) -> None:
        self.bind()
        if len(data) > self.__capacity - start_offset:
            self.resize(len(data) + start_offset)
        err_code = glGetError()
        data_array = (GLfloat * len(data))(*data)
        glBufferSubData(self.__bind_target,
                        start_offset * self.__item_byte_size,
                        len(data_array) * self.__item_byte_size, data_array)
        err_code = glGetError()

        if err_code != GL_NO_ERROR:
            print(f"Error buffer data loading: {err_code}")
        self.__filling += len(data)
        if self.__filling > self.__capacity:
            self.__filling = self.__capacity
        self.unbind()

