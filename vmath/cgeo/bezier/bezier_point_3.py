from cgeo.vectors import Vec3


class BezierPoint3:

    __slots__ = "__point", "__anchor_1", "__anchor_2", "smooth"

    def __init__(self, p: Vec3):
        self.__point: Vec3 = p
        self.__anchor_1: Vec3 = p + Vec3(0.125, 0, 0.125)
        self.__anchor_2: Vec3 = p + Vec3(-0.125, 0, -0.125)
        self.smooth: bool = True

    def __str__(self):
        return f"{{\n\t\"point\":     {self.__point},\n" \
                   f"\t\"smooth\":    {self.smooth},\n" \
                   f"\t\"anchor_1\":  {self.__anchor_1},\n" \
                   f"\t\"anchor_2\":  {self.__anchor_2}\n}}"

    def align_anchors(self, dir_: Vec3) -> None:
        w_1: float = self.anchor_1_weight
        w_2: float = self.anchor_2_weight
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
        _dw: Vec3 = self.__anchor_1 - self.__point
        _w: float = _dw.magnitude
        _dw.x *= (w / _w)
        _dw.y *= (w / _w)
        _dw.z *= (w / _w)
        self.__anchor_1 = _dw + self.__point

    @anchor_2_weight.setter
    def anchor_2_weight(self, w: float) -> None:
        _dw: Vec3 = self.__anchor_2 - self.__point
        _w: float = _dw.magnitude
        _dw.x *= (w / _w)
        _dw.y *= (w / _w)
        _dw.z *= (w / _w)
        self.__anchor_2 = _dw + self.__point

    @property
    def anchor_1(self) -> Vec3:
        return self.__anchor_1

    @anchor_1.setter
    def anchor_1(self, anchor: Vec3) -> None:
        self.__anchor_1 = anchor
        if self.smooth:
            distance = (self.point - self.__anchor_2).norm()
            self.__anchor_2 = self.point + (self.point - self.__anchor_1).normalize() * distance

    @property
    def anchor_2(self) -> Vec3:
        return self.__anchor_2

    @anchor_2.setter
    def anchor_2(self, anchor: Vec3) -> None:
        self.__anchor_2 = anchor
        if self.smooth:
            distance = (self.point - self.__anchor_1).norm()
            self.__anchor_1 = self.point + (self.point - self.__anchor_2).normalize() * distance

    @property
    def point(self) -> Vec3:
        return self.__point

    @point.setter
    def point(self, p: Vec3) -> None:
        dp: Vec3 = p - self.__point
        self.__point = p
        self.__anchor_1 = self.__anchor_1 + dp
        self.__anchor_2 = self.__anchor_2 + dp
