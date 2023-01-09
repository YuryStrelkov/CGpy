from cgeo.vectors import Vec2


class BezierPoint2:

    __slots__ = "__point", "__anchor_1", "__anchor_2", "smooth"

    def __init__(self, p: Vec2):
        self.__point: Vec2 = p
        self.__anchor_1: Vec2 = p + Vec2(0.125, 0.125)
        self.__anchor_2: Vec2 = p + Vec2(-0.125, -0.125)
        self.smooth: bool = True

    def __str__(self):
        return f"{{\n\t\"point\":     {self.__point},\n" \
                   f"\t\"smooth\":    {self.smooth},\n" \
                   f"\t\"anchor_1\":  {self.__anchor_1},\n" \
                   f"\t\"anchor_2\":  {self.__anchor_2}\n}}"

    def align_anchors(self, dir_: Vec2, weight: float = 1.0) -> None:
        w_1: float = self.anchor_1_weight * weight
        w_2: float = self.anchor_2_weight * weight
        dir_.normalize()
        self.__anchor_1 = self.__point + dir_ * w_1
        self.__anchor_2 = self.__point - dir_ * w_2

    @property
    def anchor_1_weight(self) -> float:
        return (self.__point - self.__anchor_1).magnitude

    @property
    def anchor_2_weight(self) -> float:
        return (self.__point - self.__anchor_2).magnitude

    @anchor_1_weight.setter
    def anchor_1_weight(self, w: float) -> None:
        _dw: Vec2 = self.__anchor_1 - self.__point
        _w: float = _dw.magnitude
        _dw.x *= (w / _w)
        _dw.y *= (w / _w)
        self.__anchor_1 = _dw + self.__point

    @anchor_2_weight.setter
    def anchor_2_weight(self, w: float) -> None:
        _dw: Vec2 = self.__anchor_2 - self.__point
        _w: float = _dw.magnitude
        _dw.x *= (w / _w)
        _dw.y *= (w / _w)
        self.__anchor_2 = _dw + self.__point

    @property
    def anchor_1(self) -> Vec2:
        return self.__anchor_1

    @anchor_1.setter
    def anchor_1(self, anchor: Vec2) -> None:
        self.__anchor_1 = anchor
        if self.smooth:
            distance = (self.point - self.__anchor_2).norm()
            self.__anchor_2 = self.point + (self.point - self.__anchor_1).normalize() * distance

    @property
    def anchor_2(self) -> Vec2:
        return self.__anchor_2

    @anchor_2.setter
    def anchor_2(self, anchor: Vec2) -> None:
        self.__anchor_2 = anchor
        if self.smooth:
            distance = (self.point - self.__anchor_1).norm()
            self.__anchor_1 = self.point + (self.point - self.__anchor_2).normalize() * distance

    @property
    def point(self) -> Vec2:
        return self.__point

    @point.setter
    def point(self, p: Vec2) -> None:
        dp: Vec2 = p - self.__point
        self.__point = p
        self.__anchor_1 = self.__anchor_1 + dp
        self.__anchor_2 = self.__anchor_2 + dp
