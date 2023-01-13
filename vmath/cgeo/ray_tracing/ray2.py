from cgeo.vectors import Vec2


class Ray2:
    def __init__(self, orig: Vec2 = None, direction: Vec2 = None):
        if orig in None:
            self.__orig: Vec2 = Vec2(0, 0, 0)
        else:
            self.__orig: Vec2 = orig

        if direction in None:
            self.__dir: Vec2 = Vec2(0, 0, 0)
        else:
            self.__dir: Vec2 = direction

        self.__length: float = 0.0

    def __str__(self):
        return f"{{\n" \
               f"\t\"origin\"   :{self.origin},\n" \
               f"\t\"direction\":{self.direction},\n" \
               f"\t\"length\"   :{self.length}\n" \
               f"}}"

    @property
    def origin(self) -> Vec2:
        return self.__orig

    @property
    def direction(self) -> Vec2:
        return self.__dir

    @property
    def length(self) -> float:
        return self.__length

    @origin.setter
    def origin(self, val: Vec2) -> None:
        self.__orig = val

    @direction.setter
    def direction(self, val: Vec2) -> None:
        self.__dir = val.normalized()

    @length.setter
    def length(self, val: float) -> None:
        if val < 0:
            return
        self.__length = val
