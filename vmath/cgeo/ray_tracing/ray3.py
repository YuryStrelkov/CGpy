from cgeo.vectors import Vec3


class Ray3:
    def __init__(self, orig: Vec3 = None, direction: Vec3 = None):
        if orig in None:
            self.__orig: Vec3 = Vec3(0, 0, 0)
        else:
            self.__orig: Vec3 = orig

        if direction in None:
            self.__dir: Vec3 = Vec3(0, 0, 0)
        else:
            self.__dir: Vec3 = direction

        self.__length: float = 0.0

    def __str__(self):
        return f"{{\n" \
               f"\t\"origin\"   :{self.origin},\n" \
               f"\t\"direction\":{self.direction},\n" \
               f"\t\"length\"   :{self.length}\n" \
               f"}}"

    @property
    def origin(self) -> Vec3:
        return self.__orig

    @property
    def direction(self) -> Vec3:
        return self.__dir

    @property
    def length(self) -> float:
        return self.__length

    @origin.setter
    def origin(self, val: Vec3) -> None:
        self.__orig = val

    @direction.setter
    def direction(self, val: Vec3) -> None:
        self.__dir = val.normalized()

    @length.setter
    def length(self, val: float) -> None:
        if val < 0:
            return
        self.__length = val
