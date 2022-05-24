import numpy as np


class RGB(object):
    def __init__(self, r: np.uint8, g: np.uint8, b: np.uint8):
        self.rgb: [np.uint8] = [np.uint8(r), np.uint8(g), np.uint8(b)]

    def __repr__(self):
        return "<RGB r:%s g:%s b:%s>" % (self.rgb[0], self.rgb[1], self.rgb[2])

    def __str__(self):
        return "[%s, %s, %s]" % (self.rgb[0], self.rgb[1], self.rgb[2])

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
