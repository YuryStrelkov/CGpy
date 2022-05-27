import numpy as np
from PIL import Image
from materials.material import Texture, RGB
from vmath.mathUtils import Vec2


class FrameBuffer(object):
    def __init__(self, w: int, h: int, color: RGB = RGB(125, 135, 145)):
        self.__frame_texture = Texture(w, h, 3, color)
        self.__z_buffer = np.full((self.height * self.width), -np.inf)

    @property
    def width(self):
        return self.__frame_texture.width

    @property
    def height(self):
        return self.__frame_texture.height

    # инициализация z буфера
    def clear_depth(self):
        for i in range(0, len(self.__z_buffer)):
            self.__z_buffer[i] = -np.inf

    def clear_color(self, color: RGB = RGB(np.uint8(255), np.uint8(255), np.uint8(255))):
        self.__frame_texture.clear_color(color)

    def set_pixel_uv(self, uv: Vec2, color: RGB = RGB(np.uint8(255), np.uint8(0), np.uint8(0))):
        self.__frame_texture.set_color_uv(uv, color)

    def set_pixel(self, x: int, y: int, color: RGB = RGB(np.uint8(255), np.uint8(0), np.uint8(0))):
        self.__frame_texture.set_color(x, y, color)

    def set_depth(self, x: int, y: int, depth: float) -> bool:
        pix: int = self.width * y + x
        if pix < 0:
            return False
        if pix >= self.__frame_texture.texture_pixel_size:
            return False
        if self.__z_buffer[pix] > depth:
            return False
        self.__z_buffer[pix] = depth
        return True

    # конвертация массива в объект класса Image библиотеки Pillow и сохранение его
    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    def save(self, path: str):
        im = Image.fromarray(self.img_arr, mode="RGB")
        im.save(path, mode="RGB")

    @property
    def frame_buffer_image(self) -> Image:
        return self.__frame_texture.image_data

    # конвертация массива в объект класса Image библиотеки Pillow и вывод на экран
    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show
    def imshow(self):
        self.__frame_texture.show()
