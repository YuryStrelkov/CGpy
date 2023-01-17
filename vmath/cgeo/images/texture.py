import math

from cgeo.transforms.transform2 import Transform2
from cgeo.images.rgba import RGBA
import matplotlib.pyplot as plt
from cgeo.vectors import Vec2
from typing import Tuple
from cgeo import gutils
from PIL import Image
import numpy as np
import numba

_bicubic_poly_coefficients = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              -3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                              -3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0,
                              9.0, -9.0, -9.0, 9.0, 6.0, 3.0, -6.0, -3.0, 6.0, -6.0, 3.0, -3.0, 4.0, 2.0, 2.0, 1.0,
                              -6.0, 6.0, 6.0, -6.0, -4.0, -2.0, 4.0, 2.0, -3.0, 3.0, -3.0, 3.0, -2.0, -1.0, -2.0, -1.0,
                              2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                              -6.0, 6.0, 6.0, -6.0, -3.0, -3.0, 3.0, 3.0, -4.0, 4.0, -2.0, 2.0, -2.0, -2.0, -1.0, -1.0,
                              4.0, -4.0, -4.0, 4.0, 2.0, 2.0, -2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 1.0, 1.0, 1.0, 1.0)


_zero = np.uint8(0)

_red = (np.uint8(100), _zero, _zero, _zero)


@numba.njit(fastmath=True)
def _nearest_interp_pt(x: float, y: float, points: np.ndarray, rows: int, cols: int, bpp: int) -> \
        Tuple[np.uint8, np.uint8, np.uint8, np.uint8]:
    """
    Билинейная иетерполяция точки (x,y)
    :param x: x - координата точки
    :param y: y - координата точки
    :param points: одномерный список узловых точек
    :param rows: rows of points array
    :param cols: cols of points array
    :param bpp: bpp of points array
    :return:
    """
    if x < 0:
        return _red  # 100, 0, 0, 0
    if x > 1.0:
        return _red  # 100, 0, 0, 0

    if y < 0:
        return _red  # 100, 0, 0, 0
    if y > 1.0:
        return _red  # 100, 0, 0, 0

    col_ = int(x * (cols - 1))
    row_ = int(y * (rows - 1))

    dx_ = 1.0 / (cols - 1.0)
    dy_ = 1.0 / (rows - 1.0)

    tx = (x - dx_ * col_) / dx_
    ty = (y - dy_ * row_) / dy_

    row_ = row_ if ty < 0.5 else min(row_ + 1, rows - 1)
    col_ = col_ if tx < 0.5 else min(col_ + 1, cols - 1)

    col_ *= bpp
    row_ *= bpp

    if bpp == 1:
        return points[col_ + row_ * cols], _zero, _zero, _zero

    if bpp == 3:
        return points[col_ + row_ * cols    ],\
               points[col_ + row_ * cols + 1],\
               points[col_ + row_ * cols + 2], _zero

    if bpp == 4:
        return points[col_ + row_ * cols    ],\
               points[col_ + row_ * cols + 1],\
               points[col_ + row_ * cols + 2],\
               points[col_ + row_ * cols + 3]


@numba.njit(fastmath=True)
def _bi_lin_interp_pt(x: float, y: float, points: np.ndarray, rows: int, cols: int, bpp: int) ->\
        Tuple[np.uint8, np.uint8, np.uint8, np.uint8]:
    """
    Билинейная иетерполяция точки (x,y)
    :param x: x - координата точки
    :param y: y - координата точки
    :param points: одномерный список узловых точек
    :param rows: rows of points array
    :param cols: cols of points array
    :param bpp: bpp of points array
    :return:
    """
    if x < 0:
        return _red
    if x > 1.0:
        return _red

    if y < 0:
        return _red
    if y > 1.0:
        return _red

    col_ = int(x * (cols - 1))
    row_ = int(y * (rows - 1))

    col_1 = min(col_ + 1, cols - 1) * bpp
    row_1 = min(row_ + 1, rows - 1) * bpp

    dx_ = 1.0 / (cols - 1.0)
    dy_ = 1.0 / (rows - 1.0)

    tx = (x - dx_ * col_) / dx_
    ty = (y - dy_ * row_) / dy_

    col_ *= bpp
    row_ *= bpp

    # q11 = nodes[row_, col_]

    # q00____q01
    # |       |
    # |       |
    # q10____q11

    q00: float = float(points[col_  + row_  * cols])
    q01: float = float(points[col_1 + row_  * cols])
    q10: float = float(points[col_  + row_1 * cols])
    q11: float = float(points[col_1 + row_1 * cols])

    r = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)
    if bpp == 1:
        return np.uint8(r), _zero, _zero, _zero

    q00 = float(points[col_  + row_  * cols + 1])
    q01 = float(points[col_1 + row_  * cols + 1])
    q10 = float(points[col_  + row_1 * cols + 1])
    q11 = float(points[col_1 + row_1 * cols + 1])

    g = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)

    q00 = float(points[col_  + row_  * cols + 2])
    q01 = float(points[col_1 + row_  * cols + 2])
    q10 = float(points[col_  + row_1 * cols + 2])
    q11 = float(points[col_1 + row_1 * cols + 2])

    b = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)

    if bpp == 3:
        return np.uint8(r), np.uint8(g), np.uint8(b), _zero

    q00 = float(points[col_  + row_  * cols + 3])
    q01 = float(points[col_1 + row_  * cols + 3])
    q10 = float(points[col_  + row_1 * cols + 3])
    q11 = float(points[col_1 + row_1 * cols + 3])

    a = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)

    return np.uint8(r), np.uint8(g), np.uint8(b), np.uint8(a)


def _cubic_poly(x: float, y: float, m: np.ndarray) -> float:
    """
    Вспомогательная  функция для вычисления кубического полинома би кубической интерполяции0
    :param x: x - координата
    :param y: y - координата
    :param m: матрица коэффициентов
    :return:
    """
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    return (m[0]  + m[1] *  y + m[2]  * y2 + m[3]  * y3) + \
           (m[4]  + m[5] *  y + m[6]  * y2 + m[7]  * y3) * x + \
           (m[8]  + m[9] *  y + m[10] * y2 + m[11] * y3) * x2 + \
           (m[12] + m[13] * y + m[14] * y2 + m[15] * y3) * x3


class Texture:
    """
    interp_mode = 0 <- nearest
    interp_mode = 1 <- bi-linear
    interp_mode = 2 <- bi-cubic
    """
    def __init__(self, _w: int = 3, _h: int = 3, _bpp: int = 4,
                 color: RGBA = RGBA(125, 135, 145)):
        self.__interp_mode: int = 0
        self.__source_file: str = ""
        self.__transform: Transform2 = Transform2()
        self.__colors = np.zeros(_w * _h * _bpp, dtype=np.uint8)
        self.__width = _w
        self.__bpp   = _bpp
        self.__reset_transform()
        self.clear_color(color)

    def __reset_transform(self) -> None:
        self.__transform.x  = 0.5
        self.__transform.y  = 0.5
        self.__transform.sx = 0.5
        self.__transform.sy = 0.5
        self.__transform.az = 0.0

    def __getitem__(self, index: int):
        if index < 0 or index >= self.pixels_count:
            raise IndexError(f"Texture :: GetItem :: Trying to access index: {index}")
        index *= self.bpp
        if self.bpp == 1:
            return RGBA(self.__colors[index], 0, 0, 0)
        if self.bpp == 3:
            return RGBA(self.__colors[index    ],
                        self.__colors[index + 1],
                        self.__colors[index + 2], 0)
        return RGBA(self.__colors[index],
                    self.__colors[index + 1],
                    self.__colors[index + 2],
                    self.__colors[index + 3])

    def __setitem__(self, index: int, color: RGBA):
        if index < 0 or index >= self.pixels_count:
            raise IndexError(f"Texture :: SetItem ::  Trying to access index: {index}")
        index *= self.bpp
        if self.bpp == 1:
            self.__colors[index] = color.red
            return
        if self.bpp == 3:
            self.__colors[index]     = color.red
            self.__colors[index + 1] = color.green
            self.__colors[index + 2] = color.blue
            return
        self.__colors[index]     = color.red
        self.__colors[index + 1] = color.green
        self.__colors[index + 2] = color.blue
        self.__colors[index + 3] = color.alpha

    def __str__(self) -> str:
        return f"{{\n" \
               f"\"unique_id\"      : {self.unique_id},\n" \
               f"\"source_file\"    : {self.source_file_path},\n" \
               f"\"width\"          : {self.width},\n" \
               f"\"height\"         : {self.height},\n" \
               f"\"bpp\"            : {self.bpp},\n" \
               f"\"interp_mode\"    : {self.interp_mode},\n" \
               f"\"transform\":\n" \
               f"{self.__transform}\n" \
               f"}}"

    @property
    def unique_id(self) -> int:
        return id(self)

    @property
    def transform(self) -> Transform2:
        return self.__transform

    @property
    def pixel_data(self) -> np.ndarray:
        return self.__colors

    @pixel_data.setter
    def pixel_data(self, data: np.ndarray) -> None:
        if data.ndim <= 2:
            if data.dtype == np.uint8:
                for pix_id, pix in enumerate(data.flat):
                    if pix_id == self.bytes_count:
                        break
                    self.__colors[pix_id] = pix
                return

            if data.dtype == float:
                for pix_id, pix in enumerate(data.flat):
                    if pix_id == self.bytes_count:
                        break
                    self.__colors[pix_id] = np.uint8(255 * pix)
                return

            if data.dtype == np.int:
                for pix_id, pix in enumerate(data.flat):
                    if pix_id == self.bytes_count:
                        break
                    self.__colors[pix_id * 3]     = (pix & 0x000F)
                    self.__colors[pix_id * 3 + 1] = (pix & 0x00F0) >> 8
                    self.__colors[pix_id * 3 + 2] = (pix & 0x0F00) >> 16
                    self.__colors[pix_id * 4 + 3] = (pix & 0xF000) >> 24
                return

        if data.ndim == 3:

            rows, cols, depth = data.shape

            if depth != 1 and depth != 3 and depth != 4:
                raise Exception()

            n_pix = min(rows * cols, self.pixels_count)
            bpp = min(self.bpp, depth)
            row: int
            col: int

            if data.dtype == np.uint8:
                for pix_id in range(n_pix):
                    row = pix_id // self.width
                    col = pix_id % self.width
                    for i in range(bpp):
                        self.__colors[pix_id * self.bpp + i] = data[row, col, i]
                return

            if data.dtype == float:
                for pix_id in range(n_pix):
                    row = pix_id // self.width
                    col = pix_id % self.width
                    for i in range(bpp):
                        self.__colors[pix_id * self.bpp + i] =  np.uint8(data[row, col, i])
                return

            if data.dtype == np.int:
                for pix_id in range(n_pix):
                    row = pix_id // self.width
                    col = pix_id % self.width
                    for i in range(bpp):
                        self.__colors[pix_id * self.bpp + i] = max(min(data[row, col, i], 255), 0)
                return

        raise Exception()

    @property
    def name(self) -> str:
        if len(self.__source_file) == 0:
            return ""
        name: [str] = self.__source_file.split("\\")

        if len(name) == 0:
            return ""

        name = name[len(name) - 1].split(".")

        if len(name) == 0:
            return ""

        if len(name) < 2:
            return name[0]
        return name[len(name) - 2]

    @property
    def source_file_path(self) -> str:
        return self.__source_file

    @source_file_path.setter
    def source_file_path(self, path: str) -> None:
        if path == self.__source_file:
            return
        self.load(path)

    @property
    def bpp(self) -> int:
        return self.__bpp

    @property
    def pixels_count(self) -> int:
        return self.bytes_count // self.bpp

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.pixels_count // self.width

    @property
    def aspect(self) -> float:
        return self.width * 1.0 / self.height

    @property
    def tile(self) -> Vec2:
        return self.transform.scale

    @property
    def offset(self) -> Vec2:
        return self.transform.origin

    @property
    def bytes_count(self) -> int:
        return self.__colors.size

    @property
    def rotation(self) -> float:
        return self.transform.az

    @tile.setter
    def tile(self, xy: Vec2):
        self.transform.scale = xy

    @offset.setter
    def offset(self, xy: Vec2):
        self.transform.origin = xy

    @rotation.setter
    def rotation(self, angle: float):
        self.transform.az = gutils.deg_to_rad(angle)

    @property
    def image_data(self) -> Image:
        if self.bpp == 1:
            return Image.frombytes('1', (self.width, self.height), self.pixel_data)
        if self.bpp == 3:
            return Image.frombytes('RGB', (self.width, self.height), self.pixel_data)
        if self.bpp == 4:
            return Image.frombytes('RGBA', (self.width, self.height), self.pixel_data)
        raise RuntimeError(f"Texture::image_data::Incorrect bbp: {self.bpp}")

    @property
    def interp_mode(self) -> int:
        return self.__interp_mode

    @interp_mode.setter
    def interp_mode(self, mode: int) -> None:
        if mode < 0:
            return
        if mode > 2:
            return
        self.__interp_mode = mode

    @property
    def duv(self) -> Vec2:
        return Vec2(1.0 / (self.width  - 1),
                    1.0 / (self.height - 1))

    def load(self, origin: str) -> None:
        if not (self.__colors is None):
            del self.__colors
            self.__width = -1
            self.__bpp = 0
        self.__source_file = origin
        image         = Image.open(self.__source_file)
        self.__width  = image.size[0]
        self.__colors = np.asarray(image, dtype=np.uint8).ravel()
        self.__bpp    = self.pixel_data.size // image.height // image.width
        self.__reset_transform()

    def set_color_raw(self, row: int, col: int, color: Tuple[np.uint8, np.uint8, np.uint8, np.uint8]) -> None:
        try:
            index = (row * self.width + col) * self.bpp
            if self.bpp == 1:
                self.__colors[index] = color[0]
                return
            if self.bpp == 3:
                self.__colors[index] = color[0]
                self.__colors[index + 1] = color[1]
                self.__colors[index + 2] = color[2]
                return
            self.__colors[index] = color[0]
            self.__colors[index + 1] = color[1]
            self.__colors[index + 2] = color[2]
            self.__colors[index + 3] = color[3]
        except IndexError as _ex:
            print(_ex.args)

    def set_color(self, row: int, col: int, color: RGBA) -> None:
        try:
            self[row * self.width + col] = color
        except IndexError as _ex:
            print(_ex.args)

    def get_color(self, row: int, col: int) -> RGBA:
        try:
            return self[row * self.width + col]
        except IndexError as _ex:
            print(_ex.args)
            return RGBA()

    def set_color_uv(self, uv: Vec2, color: RGBA) -> None:
        """
        uv:: uv.x in range[0,1], uv.y in range[0,1]
        """
        uv_ = self.transform.transform_vect(uv)
        self.set_color(int(uv_.y * self.height),
                       int(uv_.x * self.width), color)

    # uv:: uv.x in range[0,1], uv.y in range[0,1]
    def get_color_uv_raw(self, uv: Vec2) -> Tuple[np.uint8, np.uint8, np.uint8, np.uint8]:
        """
        uv:: uv.x in range[0,1], uv.y in range[0,1]
        """
        aspect = self.aspect
        uv.v *= aspect
        uv = self.transform.transform_vect(uv)
        uv.v = (uv.v - self.transform.x) / aspect + self.transform.x
        if self.interp_mode == 1:
            return _bi_lin_interp_pt(uv.x, uv.y, self.pixel_data, self.height, self.width, self.bpp)
        if self.interp_mode == 2:
            return _bi_lin_interp_pt(uv.x, uv.y, self.pixel_data, self.height, self.width, self.bpp)
        return _nearest_interp_pt(uv.x, uv.y, self.pixel_data, self.height, self.width, self.bpp)

    def get_color_uv(self, uv: Vec2) -> RGBA:
        """
        uv:: uv.x in range[0,1], uv.y in range[0,1]
        """
        return RGBA(*self.get_color_uv_raw(uv))

    def show(self) -> None:
        self.image_data.show()

    def clear_color(self, color: RGBA = RGBA(125, 135, 145)) -> None:
        if self.bytes_count == 0:
            return

        rgb = (0,)

        if self.bpp == 1:
            rgb = (color.red,)

        if self.bpp == 3:
            rgb = (color.red, color.green, color.blue)

        if self.bpp == 4:
            rgb = (color.red, color.green, color.blue, color.alpha)

        for i in range(self.bytes_count):
            self.__colors[i] = rgb[i % self.bpp]

    @staticmethod
    def rotate(image, angele: float):
        if not isinstance(image, Texture):
            raise RuntimeError("")
        t_rot = Texture(image.width, image.height, image.bpp)
        old_ang = image.rotation
        image.rotation = angele
        row: int
        col: int
        uv  = Vec2(0.0, 0.0)
        duv = image.duv
        for pix in range(image.pixels_count):
            row  = pix // image.width
            col  = pix  % image.width
            uv.u, uv.v = 2.0 * row * duv.u - 1.0, (2.0 * col * duv.v - 1.0)
            t_rot.set_color_raw(row, col, image.get_color_uv_raw(uv))
        image.rotation = old_ang
        return t_rot

    @staticmethod
    def scale(image, sx: float, sy: float):
        pass

    @staticmethod
    def crop(image, bound_min: Tuple[int, int], bound_max: Tuple[int, int]):
        pass


def transform_test():
    rows = 10
    cols = 25
    d_col = 2.0 / (cols - 1)
    d_row = 2.0 / (rows - 1)
    n_points = rows * cols
    transform = Transform2()
    # transform.sy = 2
    transform.x = 0.5
    transform.y = 0.5
    transform.sx = 0.5
    transform.sy = 0.5
    xy_ = [Vec2(-1.0 + (i % cols) * d_col, -1.0 + (i // cols) * d_row) for i in range(n_points)]
    xy = [transform.transform_vect(v) for v in xy_]
    transform.az = np.pi / 4.0
    # transform.sy = 1.5
    xyt = [transform.transform_vect(v) for v in xy_]
    fig, axs = plt.subplots(1, 1)
    [axs.plot(v.x, v.y, '.r') for v in xy]
    [axs.plot(v.x, v.y, '.g') for v in xyt]
    axs.set_aspect('equal', 'box')
    plt.show()


if __name__ == "__main__":
    # transform_test()
    # exit()
    texture = Texture()
    texture.load("iceland.png")
    texture.interp_mode = 1
    print(texture)
    texture_r = Texture.rotate(texture, 45)
    texture_r.show()
