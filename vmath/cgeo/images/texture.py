from cgeo.transforms.transform2 import Transform2
from cgeo.images.rgba import RGBA
from cgeo.vectors import Vec2
from typing import Tuple
from cgeo import gutils
from PIL import Image
import numpy as np


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


def _nearest_interp_pt(x: float, y: float, points: np.ndarray, rows: int, cols: int, bpp: int) -> Tuple[int, ...]:
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
        return 100, 0, 0, 0
    if x > 1.0:
        return 100, 0, 0, 0

    if y < 0:
        return 100, 0, 0, 0
    if y > 1.0:
        return 100, 0, 0, 0

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
        return int(points[col_ + row_ * cols]),

    if bpp == 3:
        return int(points[col_ + row_ * cols    ]),\
               int(points[col_ + row_ * cols + 1]),\
               int(points[col_ + row_ * cols + 2])

    if bpp == 4:
        return int(points[col_ + row_ * cols    ]),\
               int(points[col_ + row_ * cols + 1]),\
               int(points[col_ + row_ * cols + 2]),\
               int(points[col_ + row_ * cols + 3])


def _bi_linear_interp_pt(x: float, y: float, points: np.ndarray, rows: int, cols: int, bpp: int) -> Tuple[int, ...]:
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
        return 100, 0, 0, 0
    if x > 1.0:
        return 100, 0, 0, 0

    if y < 0:
        return 100, 0, 0, 0
    if y > 1.0:
        return 100, 0, 0, 0

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
        return int(r),

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
        return int(r), int(g), int(b)

    q00 = float(points[col_  + row_  * cols + 3])
    q01 = float(points[col_1 + row_  * cols + 3])
    q10 = float(points[col_  + row_1 * cols + 3])
    q11 = float(points[col_1 + row_1 * cols + 3])

    a = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)

    return int(r), int(g), int(b), int(a)


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
        self.__interp_mode: int = 1
        self.__source_file: str = ""
        self.__transform: Transform2 = Transform2()
        # self.__transform.origin = Vec2(0.10, 0.20)
        self.__colors = np.zeros(_w * _h * _bpp, dtype=np.uint8)
        self.__width = _w
        self.__bpp   = _bpp
        self.clear_color(color)

    def __getitem__(self, index: int):
        if index < 0 or index >= self.texture_pixels_size:
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
        if index < 0 or index >= self.texture_pixels_size:
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

    def __repr__(self) -> str:
        res: str = "<Texture\n"
        res += f"source file    : {self.source_file_path}\n"
        res += f"width          : {self.width}\n"
        res += f"height         : {self.height}\n"
        res += f"bpp            : {self.bpp}\n"
        res += "affine transform:\n"
        res += f"{self.__transform}\n>\n"
        return res

    def __str__(self) -> str:
        res: str = ""
        res += f"source file    : {self.source_file_path}\n"
        res += f"width          : {self.width}\n"
        res += f"height         : {self.height}\n"
        res += f"bpp            : {self.bpp}\n"
        res += "affine transform:\n"
        res += f"{self.__transform}\n"
        return res

    @property
    def pixel_data(self) -> np.ndarray:
        return self.__colors

    @pixel_data.setter
    def pixel_data(self, data: np.ndarray) -> None:
        if data.ndim <= 2:
            if data.dtype == np.uint8:
                for pix_id, pix in enumerate(data.flat):
                    if pix_id == self.texture_byte_size:
                        break
                    self.__colors[pix_id] = pix
                return

            if data.dtype == float:
                for pix_id, pix in enumerate(data.flat):
                    if pix_id == self.texture_byte_size:
                        break
                    self.__colors[pix_id] = np.uint8(255 * pix)
                return

            if data.dtype == np.int:
                for pix_id, pix in enumerate(data.flat):
                    if pix_id == self.texture_byte_size:
                        break
                    self.__colors[pix_id * 3]     = (pix &       255)
                    self.__colors[pix_id * 3 + 1] = (pix &      65280) >> 8
                    self.__colors[pix_id * 3 + 1] = (pix &   16711680) >> 16
                    self.__colors[pix_id * 4 + 1] = (pix & 4278190080) >> 24
                return

        if data.ndim == 3:

            rows, cols, depth = data.shape

            if depth != 1 and depth != 3 and depth != 4:
                raise Exception()

            n_pix = min(rows * cols, self.texture_pixels_size)
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
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.texture_byte_size // self.bpp // self.width

    @property
    def aspect(self) -> float:
        return self.width * 1.0 / self.height

    @property
    def texture_pixels_size(self) -> int:
        return self.texture_byte_size // self.bpp

    @property
    def tile(self) -> Vec2:
        return self.__transform.scale

    @property
    def offset(self) -> Vec2:
        return self.__transform.origin

    @property
    def texture_byte_size(self) -> int:
        return self.__colors.size

    @property
    def rotation(self) -> float:
        return self.__transform.az

    @tile.setter
    def tile(self, xy: Vec2):
        self.__transform.scale = xy

    @offset.setter
    def offset(self, xy: Vec2):
        self.__transform.origin = xy

    @rotation.setter
    def rotation(self, angle: float):
        self.__transform.az = gutils.deg_to_rad(angle)

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

    def load(self, origin: str) -> None:
        if not (self.__colors is None):
            del self.__colors
            self.__width = -1
            self.__bpp = 0
        self.__source_file = origin
        image = Image.open(self.__source_file)
        self.__width = image.size[0]
        self.__colors: np.ndarray = np.asarray(image, dtype=np.uint8).ravel()
        self.__bpp = self.pixel_data.size // image.height // image.width

    def set_color(self, row: int, col: int, color: RGBA) -> None:
        pix = row * self.width + col
        try:
            self[pix] = color
        except IndexError as _ex:
            print(_ex.args)

    def get_color(self, row: int, col: int) -> RGBA:
        pix = row * self.width + col
        try:
            return self[pix]
        except IndexError as _ex:
            print(_ex.args)
            return RGBA()

    def set_color_uv(self, uv: Vec2, color: RGBA) -> None:
        """
        uv:: uv.x in range[0,1], uv.y in range[0,1]
        """
        uv_ = self.__transform.inv_transform_vect(uv)
        self.set_color(int(uv_.y * self.height), int(uv_.x * self.width), color)

    # uv:: uv.x in range[0,1], uv.y in range[0,1]
    def get_color_uv(self, uv: Vec2) -> RGBA:
        """
        uv:: uv.x in range[0,1], uv.y in range[0,1]
        """
        uv_ = self.__transform.inv_transform_vect(uv)

        if self.interp_mode == 1:
            return RGBA(*_bi_linear_interp_pt(uv_.y, uv_.x, self.pixel_data, self.height, self.width, self.bpp))

        if self.interp_mode == 2:
            return RGBA(*_bi_linear_interp_pt(uv_.y, uv_.x, self.pixel_data, self.height, self.width, self.bpp))

        return RGBA(*_nearest_interp_pt(uv_.y, uv_.x, self.pixel_data, self.height, self.width, self.bpp))

    def show(self) -> None:
        self.image_data.show()

    def clear_color(self, color: RGBA = RGBA(125, 135, 145)) -> None:
        if self.texture_byte_size == 0:
            return

        rgb = (0,)

        if self.bpp == 1:
            rgb = (color.red,)

        if self.bpp == 3:
            rgb = (color.red, color.green, color.blue)

        if self.bpp == 4:
            rgb = (color.red, color.green, color.blue, color.alpha)

        for i in range(self.texture_byte_size):
            self.__colors[i] = rgb[i % self.bpp]


def tex_rot(image: Texture, angele: float) -> Texture:
    t_rot = Texture(image.width, image.height, image.bpp)
    old_ang = image.rotation
    image.rotation = angele
    # image.tile = Vec2(0.75, 0.75)
    row: int
    col: int
    uv  = Vec2(0.0, 0.0)
    duv = Vec2(1.0 / (image.width - 1), 1.0 / (image.height - 1))
    aspect = 1.0 / image.aspect
    for pix in range(image.texture_pixels_size):
        row = pix // image.width
        col = pix  % image.width
        uv.y = col * duv.x + 0.25
        uv.x = row * duv.y - 0.25
        t_rot.set_color(row, col, image.get_color_uv(uv))
    image.rotation = old_ang
    return t_rot


if __name__ == "__main__":
    texture = Texture()
    texture.load("test.jpg")
    # texture.show()
    texture_r = tex_rot(texture, 45.0)
    texture_r.show()
