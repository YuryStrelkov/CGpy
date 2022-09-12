from core.transforms.transform import Transform
from core.vectors import Vec3


class BoundingBox:

    __slots__ = "__max", "__min"

    def __init__(self):
        self.__max: Vec3 = Vec3(-1e12, -1e12, -1e12)
        self.__min: Vec3 = Vec3(1e12, 1e12, 1e12)

    def __str__(self):
        return f"{{\n" \
               f"\t\"min\": {self.min},\n" \
               f"\t\"max\": {self.max}" \
               f"\n}}"

    @property
    def points(self):
        c = self.center
        s = self.size
        yield Vec3(c.x - s.x, c.y + s.y, c.z - s.z)
        yield Vec3(c.x + s.x, c.y - s.y, c.z - s.z)
        yield Vec3(c.x - s.x, c.y - s.y, c.z - s.z)
        yield Vec3(c.x + s.x, c.y + s.y, c.z - s.z)

        yield Vec3(c.x - s.x, c.y + s.y, c.z + s.z)
        yield Vec3(c.x + s.x, c.y - s.y, c.z + s.z)
        yield Vec3(c.x - s.x, c.y - s.y, c.z + s.z)
        yield Vec3(c.x + s.x, c.y + s.y, c.z + s.z)

    def encapsulate(self, v: Vec3) -> None:
        if v.x > self.__max.x:
            self.__max.x = v.x
        if v.y > self.__max.y:
            self.__max.y = v.y
        if v.z > self.__max.z:
            self.__max.z = v.z
        # update min bound
        if v.x < self.__min.x:
            self.__min.x = v.x
        if v.y < self.__min.y:
            self.__min.y = v.y
        if v.z < self.__min.z:
            self.__min.z = v.z

    def transform_bbox(self, transform: Transform):
        bounds = BoundingBox()
        for pt in self.points:
            bounds.encapsulate(transform.transform_vect(pt))
        return bounds

    def inv_transform_bbox(self, transform: Transform):
        bounds = BoundingBox()
        for pt in self.points:
            bounds.encapsulate(transform.inv_transform_vect(pt))
        return bounds

    @property
    def min(self) -> Vec3:
        return self.__min

    @property
    def max(self) -> Vec3:
        return self.__max

    @property
    def size(self) -> Vec3:
        return self.__max - self.__min

    @property
    def center(self) -> Vec3:
        return (self.__max + self.__min) * 0.5

