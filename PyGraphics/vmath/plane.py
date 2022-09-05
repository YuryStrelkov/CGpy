from vmath.vectors import Vec3


class Plane:

    __slots__ = "__r", "__n"

    def __init__(self):
        self.__r = Vec3(0, 0, 0)
        self.__n = Vec3(0, 0, 1)

    @property
    def r(self) -> Vec3:
        return self.__r

    @property
    def n(self) -> Vec3:
        return self.__n

    def point_plane_distance(self, pt: Vec3) -> float:
        return Vec3.dot(self.__n, (pt - self.__r))

    def ray_plane_intersect(self, ray_orig: Vec3, ray_dir: Vec3) -> float:
        en: float = Vec3.dot(ray_dir, self.__n)
        if abs(en) < 1e-9:
            return 0.0
        return Vec3.dot(self.__n, (self.__r - ray_orig)) / Vec3.dot(ray_dir, self.__n)
