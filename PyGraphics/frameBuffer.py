import numpy as np
from PIL import Image
from material import Texture, RGB
from mathUtils import Vec2


class FrameBuffer(object):
    def __init__(self, w: int, h: int):
        self.colorTexture = Texture(w, h, 3)
        self.clear_color()
        self.clear_depth()
        self.zBuffer = np.full((self.height * self.width), -np.inf)

    @property
    def width(self):
        return self.colorTexture.width

    @property
    def height(self):
        return self.colorTexture.height

    # инициализация z буфера
    def clear_depth(self):
        self.zBuffer = np.full((self.height * self.width), -np.inf)

    def clear_color(self, color: RGB = RGB(np.uint8(255), np.uint8(255), np.uint8(255))):
        self.colorTexture.clear_color(color)

    def set_pixel_uv(self, uv: Vec2, color: RGB = RGB(np.uint8(255), np.uint8(0), np.uint8(0))):
        self.colorTexture.set_color_uv(uv, color)

    def set_pixel(self, x: int, y: int, color: RGB = RGB(np.uint8(255), np.uint8(0), np.uint8(0))):
        self.colorTexture.set_color(x, y, color)

    def set_depth(self, x: int, y: int, depth: float) -> bool:
        pix: int = self.width * y + x
        if pix < 0:
            return False
        if pix >= self.colorTexture.texture_pixel_size:
            return False
        if self.zBuffer[pix] > depth:
            return False
        self.zBuffer[pix] = depth
        return True

    # конвертация массива в объект класса Image библиотеки Pillow и сохранение его
    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    def save(self, path: str):
        im = Image.fromarray(self.img_arr, mode="RGB")
        im.save(path, mode="RGB")

    @property
    def frame_buffer_image(self) -> Image:
        return self.colorTexture.image_data;

    # конвертация массива в объект класса Image библиотеки Pillow и вывод на экран
    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show
    def imshow(self):
        self.colorTexture.show();
