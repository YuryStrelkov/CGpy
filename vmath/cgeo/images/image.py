from ctypes import Structure, POINTER, c_int8, c_int32, CDLL, c_uint32, c_float, c_char_p, c_uint8, create_string_buffer

import numpy as np
# from matplotlib import pyplot as plt

from cgeo import Vec2
from cgeo.images import RGBA
import os

from cgeo.transforms import Transform2

path = os.getcwd()
#E:\GitHub\CGpy\vmath\cgeo\images\Images\x64\Release
#image_op_lib = CDLL(path + "\Images.dll")
image_op_lib = CDLL(r"E:\GitHub\CGpy\vmath\cgeo\images\Images\x64\Release\Images.dll")


class _Image(Structure):
    _fields_ = ("data", POINTER(c_int8)), \
               ("bpp",  c_int8 ), \
               ("rows", c_int32), \
               ("cols", c_int32)


image_new          = image_op_lib.image_new
image_new.argtypes = [c_uint32, c_uint32, c_int8]
image_new.restype  = POINTER(_Image)

image_del          = image_op_lib.image_del
image_del.argtypes = [POINTER(_Image)]

image_load          = image_op_lib.image_load
image_load.argtypes = [c_char_p]
image_load.restype  = POINTER(_Image)

image_save          = image_op_lib.image_save
image_save.argtypes = [POINTER(_Image), c_char_p]
image_save.restype  = c_int8
##########################################################
get_pix          = image_op_lib.get_pix
get_pix.argtypes = [c_uint32, c_uint32, POINTER(_Image)]
get_pix.restype  = c_int32

get_uv          = image_op_lib.get_uv
get_uv.argtypes = [c_float, c_float, POINTER(_Image), c_uint8]
get_uv.restype  = c_uint32

set_pix          = image_op_lib.set_pix
set_pix.argtypes = [c_uint32, c_uint32, c_uint32, POINTER(_Image)]
set_pix.restype   = c_uint8

get_color_nearest          = image_op_lib.nearest32
get_color_nearest.argtypes = [c_float, c_float, POINTER(_Image)]
get_color_nearest.restype  = c_uint32

get_color_bi_cubic          = image_op_lib.bicubic32
get_color_bi_cubic.argtypes = [c_float, c_float, POINTER(_Image)]
get_color_bi_cubic.restype  = c_uint32

get_color_bi_linear          = image_op_lib.bilinear32
get_color_bi_linear.argtypes = [c_float, c_float, POINTER(_Image)]
get_color_bi_linear.restype  = c_uint32

rescale = image_op_lib.rescale
rescale.argtypes = [c_float, c_float, POINTER(_Image), c_uint8]
rescale.restype  = POINTER(_Image)

image_clear_color = image_op_lib.image_clear_color
image_clear_color.argtypes = [POINTER(_Image), c_int32]

transform = image_op_lib.transform
transform.argtypes = [POINTER(c_float), POINTER(_Image), c_int8, c_int8]
transform.restype  = POINTER(_Image)


class Image:
    def __init__(self, width: int = 10, height: int = 10, bpp: int = 3):
        self.__image: _Image = image_new(height, width, bpp)
        self.__interp_mode: int = 2

    def __del__(self):
        image_del(self.__image)
        print("image free...")

    def __str__(self):
        return f"width:  {self.width},\n" \
               f"height: {self.height},\n" \
               f"bpp:    {self.bpp}"

    def __getitem__(self, index: int):
        row, col = divmod(index, self.width)
        color: int = get_pix(row, col, self.__image)
        if color < 0:
            raise IndexError(f"Texture :: GetItem :: Trying to access index: {index}")
        return RGBA(color)

    def __setitem__(self, index: int, color: RGBA):
        row, col = divmod(index, self.width)
        color: int = set_pix(row, col, int(color), self.__image)
        if color == 0:
            raise IndexError(f"Texture :: SetItem ::  Trying to access index: {index}")

    def transform(self, t: Transform2):
        tr_ptr = c_float * 9
        transformed = transform(tr_ptr(*t.transform_matrix.as_list), self.__image, self.__interp_mode, 1)
        image_del(self.__image)
        self.__image = transformed

    def get_uv(self, u: float, v: float) -> RGBA:
        return RGBA(get_uv(u, v, self.__image, c_uint8(self.__interp_mode)))

    def rescale(self, sx: float, sy: float):
        rescaled = rescale(sx, sy, self.__image, c_uint8(self.__interp_mode))
        image_del(self.__image)
        self.__image = rescaled

    def clear_color(self, color: RGBA):
        image_clear_color(self.__image, int(color))

    @property
    def bpp(self) -> int:
        return self.__image.contents.bpp

    @property
    def width(self) -> int:
        return self.__image.contents.cols

    @property
    def height(self) -> int:
        return self.__image.contents.rows

    def load(self, img_path: str):
        image_del(self.__image)
        self.__image = image_load(create_string_buffer((os.getcwd() + "\\" + img_path).encode('utf-8')))

    def save(self, img_path: str):
        image_save(self.__image, create_string_buffer((os.getcwd() + "\\" + img_path).encode('utf-8')))


def get_bounds(t: Transform2):
    pts = [Vec2(-100, -50.0), Vec2(-100, 50.0), Vec2(100.0, 50.0), Vec2(100.0, -50.0), Vec2(-100.0, -50.0)]
    pts_t = [t.transform_vect(v) for v in pts]
    x_ = [v.x for v in pts_t]
    y_ = [v.y for v in pts_t]
    x_min = min(x_)
    x_max = max(x_)
    y_min = min(y_)
    y_max = max(y_)
    return [Vec2(x_min, y_min), Vec2(x_min, y_max), Vec2(x_max, y_max), Vec2(x_max, y_min), Vec2(x_min, y_min)]


if __name__ == "__main__":
    trnsfrm = Transform2()
    # trnsfrm.az =  -np.pi * 0.05
    trnsfrm.sx = 0.50
    trnsfrm.sy = 0.50

    image = Image()
    image.load("test_images\\iceland.png")
    image.transform(trnsfrm)
    image.save("test_images\\iceland_read.png")
    exit()

    pts = [Vec2(-100, -50.0), Vec2(-100, 50.0), Vec2(100.0, 50.0), Vec2(100.0, -50.0), Vec2(-100.0, -50.0)]
    pts_t = [trnsfrm.transform_vect(v) for v in pts]
    x_ = [v.x for v in pts]
    y_ = [v.y for v in pts]
    x_t = [v.x for v in pts_t]
    y_t = [v.y for v in pts_t]

    b = get_bounds(trnsfrm)

    x_b = [v.x for v in b]
    y_b = [v.y for v in b]

    # fig, axs = plt.subplots(1, 1)
    # axs.plot(x_, y_, 'r')
    #  axs.plot(x_t, y_t, 'g')
    # axs.plot(x_b, y_b, 'b')
    # axs.set_aspect('equal', 'box')
    # axs.grid(True)
    # plt.show()

    # image.clear_color(RGBA(0, 255, 0))
    # image.save("test_images\\micro_clear.png")

    #image.load("test_images\\iceland.jpg")
    #t = LoopTimer()
    #with t:
    #    image.rescale(3., 3.)
    #print(f"image rescale t: {t.last_loop_time}")
    #image.save("test_images\\micro_32.png")
#