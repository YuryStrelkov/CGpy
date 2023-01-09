from computational_geometry.transforms.transform2 import Transform2
from computational_geometry.vectors import Vec2


class BoundingRect:
    __slots__ = "__max", "__min"

    def __init__(self):
        self.__max: Vec2 = Vec2(-1e12, -1e12)
        self.__min: Vec2 = Vec2(1e12, 1e12)

    def __str__(self):
        return f"{{\n" \
               f"\t\"min\": {self.min},\n" \
               f"\t\"max\": {self.max}" \
               f"\n}}"

    def reset(self):
        self.__max: Vec2 = Vec2(-1e12, -1e12)
        self.__min: Vec2 = Vec2( 1e12, 1e12)

    @property
    def points(self):
        c = self.center
        s = self.size
        yield Vec2(c.x - s.x, c.y + s.y)
        yield Vec2(c.x + s.x, c.y - s.y)
        yield Vec2(c.x - s.x, c.y - s.y)
        yield Vec2(c.x + s.x, c.y + s.y)

    def encapsulate(self, v: Vec2) -> None:
        if v.x > self.__max.x:
            self.__max.x = v.x
        if v.y > self.__max.y:
            self.__max.y = v.y
        if v.x < self.__min.x:
            self.__min.x = v.x
        if v.y < self.__min.y:
            self.__min.y = v.y

    def transform_bbox(self, transform: Transform2):
        bounds = BoundingRect()
        for pt in self.points:
            bounds.encapsulate(transform.transform_vect(pt))
        return bounds

    def inv_transform_bbox(self, transform: Transform2):
        bounds = BoundingRect()
        for pt in self.points:
            bounds.encapsulate(transform.inv_transform_vect(pt))
        return bounds

    @property
    def min(self) -> Vec2:
        return self.__min

    @property
    def max(self) -> Vec2:
        return self.__max

    @property
    def size(self) -> Vec2:
        return self.__max - self.__min

    @property
    def center(self) -> Vec2:
        return (self.__max + self.__min) * 0.5
