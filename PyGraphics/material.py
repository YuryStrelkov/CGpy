import numpy as np
from PIL import Image
import mathUtils
from mathUtils import Vec2
from transform import Transform2


class RGB(object):
    def __init__(self, r: np.uint8, g: np.uint8, b: np.uint8):
        self.rgb: [np.uint8] = [np.uint8(r), np.uint8(g), np.uint8(b)]

    def __repr__(self): return "<RGB r:%s g:%s b:%s>" % (self.rgb[0], self.rgb[1], self.rgb[2])

    def __str__(self): return "[%s, %s, %s]" % (self.rgb[0], self.rgb[1], self.rgb[2])

    @property
    def r(self) -> np.uint8: return self.rgb[0]

    @property
    def g(self) -> np.uint8: return self.rgb[1]

    @property
    def b(self) -> np.uint8: return self.rgb[2]

    @r.setter
    def r(self, r: np.uint8): self.rgb[0] = r

    @g.setter
    def g(self, g: np.uint8): self.rgb[1] = g

    @b.setter
    def b(self, b: np.uint8): self.rgb[2] = b


class Texture(object):
    def __init__(self, _w: int = 0, _h: int = 0, _bpp: int = 0):
        self.transform: Transform2 = Transform2()
        self.colors: [np.uint8] = None
        self.width_ = _w
        self.height_ = _h
        self.bpp_ = _bpp
        # self.transform.scale = vec2(_w, -_h);
        self.clear_color()

    @property
    def width(self) -> int:
        return self.width_

    @property
    def height(self) -> int:
        return self.height_

    @property
    def bpp(self) -> int:
        return self.bpp_

    @property
    def texture_pixel_size(self):
        return self.height * self.width

    @property
    def tile(self) -> Vec2:
        return self.transform.scale

    @property
    def offset(self) -> Vec2:
        return self.transform.origin

    @property
    def texture_byte_size(self):
        return self.bpp * self.height * self.width

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
        self.transform.az = mathUtils.deg_to_rad(angle)

    @property
    def image_data(self) -> Image:
        if self.bpp == 3:
            return Image.frombytes('RGB', (self.width, self.height), self.colors)
        if self.bpp == 4:
            return Image.frombytes('RGBA', (self.width, self.height), self.colors)

    def load(self, origin: str):
        if not (self.colors is None):
            del self.colors
            self.width_ = -1
            self.height_ = -1
            self.bpp_ = 0
        im = Image.open(origin)
        self.width_, self.height_ = im.size
        self.bpp_ = im.layers
        self.colors: [np.uint8] = (np.asarray(im, dtype=np.uint8)).ravel()

    def set_color(self, i: int, j: int, color: RGB):
        pix = round((i + j * self.width_) * self.bpp_)
        if pix < 0:
            return
        if pix >= self.width_ * self.height_ * self.bpp_ - 2:
            return
        self.colors[pix] = color.r
        self.colors[pix + 1] = color.g
        self.colors[pix + 2] = color.b

    def get_color(self, i: int, j: int) -> RGB:
        pix = round((i + j * self.width_) * self.bpp_)
        if pix < 0:
            return RGB(np.uint8(0), np.uint8(0), np.uint8(0))
        if pix >= self.width_ * self.height_ * self.bpp_ - 2:
            return RGB(np.uint8(0), np.uint8(0), np.uint8(0))
        return RGB(self.colors[pix],
                   self.colors[pix + 1],
                   self.colors[pix + 2])

    # uv:: uv.x in range[0,1], uv.y in range[0,1]
    def set_color_uv(self, uv: Vec2, color: RGB):
        uv_ = self.transform.inv_transform_vect(uv, 1)
        pix = round((uv_.x + uv_.y * self.width_) * self.bpp_)
        if pix < 0:
            return
        if pix >= self.width_ * self.height_ * self.bpp_ - 2:
            return
        self.colors[pix] = color.r
        self.colors[pix + 1] = color.g
        self.colors[pix + 2] = color.b

    # uv:: uv.x in range[0,1], uv.y in range[0,1]
    def get_color_uv(self, uv: Vec2) -> RGB:
        uv_ = self.transform.transform_vect(uv, 1)
        uv_x = abs(round(uv_.x * self.width_) % self.width_)
        uv_y = abs(round(uv_.y * self.height_) % self.height_)
        pix = (uv_x + uv_y * self.width_) * self.bpp_
        return RGB(self.colors[pix], self.colors[pix + 1], self.colors[pix + 2])

    def show(self):
        self.image_data.show()

    def clear_color(self, color: RGB = RGB(np.uint8(125), np.uint8(125), np.uint8(125))):
        if self.texture_byte_size == 0:
            return
        if not(self.colors is None):
            del self.colors
        self.colors = np.zeros((self.height_ * self.width_ * self.bpp), dtype=np.uint8)
        rgb = [color.r, color.g, color.g]
        for i in range(0, len(self.colors)):
            self.colors[i] = rgb[i % 3]


class Material(object):
    def __init__(self):
        self.diffuse: Texture = None
        self.specular: Texture = None
        self.normals: Texture = None

    def set_diff(self, orig: str):
        if self.diffuse is None:
            self.diffuse = Texture()
        self.diffuse.load(orig)

    def set_norm(self, orig: str):
        if self.normals is None:
            self.normals = Texture()
        self.normals.load(orig)

    def set_spec(self, orig: str):
        if self.specular is None:
            self.specular = Texture()
        self.specular.load(orig)

    def diff_color(self, uv: Vec2) -> RGB:
        if self.diffuse is None:
            return RGB(np.uint8(255), np.uint8(255), np.uint8(255))
        return self.diffuse.get_color_uv(uv)

    def norm_color(self, uv: Vec2) -> RGB:
        if self.normals is None:
            return RGB(np.uint8(255), np.uint8(255), np.uint8(255))
        return self.normals.get_color_uv(uv)

    def spec_color(self, uv: Vec2) -> RGB:
        if self.specular is None:
            return RGB(np.uint8(255), np.uint8(255), np.uint8(255))
        return self.specular.get_color_uv(uv)
