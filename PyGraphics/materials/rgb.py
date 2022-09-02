import numpy as np


class RGB:

    __slots__ = "__rgb"

    def __init__(self, r: np.uint8 = 255, g: np.uint8 = 255, b: np.uint8 = 255):
        self.__rgb: [np.uint8] = [np.uint8(r), np.uint8(g), np.uint8(b)]

    def __str__(self):
        return f"{{\"r\":{self.r:3}, \"g\":{self.g:3}, \"b\":{self.b:3}}}"

    def __getitem__(self, index):
        if index < 0 or index >= 3:
            return np.uint8(0)
        return self.__rgb[index]

    def __eq__(self, other) -> bool:
        if not (type(other) is RGB):
            return False
        if not (self.r == other.r):
            return False
        if not (self.g == other.g):
            return False
        if not (self.b == other.b):
            return False
        return True

    def __hash__(self) -> int:
        return hash((self.r, self.g, self.b))

    @property
    def r(self) -> np.uint8:
        return self.__rgb[0]

    @property
    def g(self) -> np.uint8:
        return self.__rgb[1]

    @property
    def b(self) -> np.uint8:
        return self.__rgb[2]

    @r.setter
    def r(self, r: np.uint8) -> None:
        self.__rgb[0] = r

    @g.setter
    def g(self, g: np.uint8) -> None:
        self.__rgb[1] = g

    @b.setter
    def b(self, b: np.uint8) -> None:
        self.__rgb[2] = b
