from vmath.math_utils import Vec2
from vmath import math_utils


def bezier_2_cubic(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, t: float) -> Vec2:
    one_min_t: float = 1.0 - t
    a: float = one_min_t * one_min_t * one_min_t
    b: float = 3.0 * one_min_t * one_min_t * t
    c: float = 3.0 * one_min_t * t * t
    d: float = t * t * t
    return Vec2(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                p1.y * a + p2.y * b + p3.y * c + p4.y * d)


def bezier_2_tangent(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, t: float) -> Vec2:
    d: float = 3 * t * t
    a: float = -3 + 6 * t - d
    b: float = 3 - 12 * t + 3 * d
    c: float = 6 * t - 3 * d
    return Vec2(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                p1.y * a + p2.y * b + p3.y * c + p4.y * d)


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

    def align_anchors(self, dir_: Vec2, weight: float = 1) -> None:
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


class BezierCurve2(object):
    def __init__(self):
        self.__sections_per_seg: int = 8
        self.__points: [BezierPoint2] = []
        self.closed: bool = False

    def __iter__(self):
        if len(self.__points) <= 1:
            raise StopIteration
        self.__iter_sect_i: int = 0
        self.__iter_sect: int = 1
        self.__iter_p1: BezierPoint2 = self.__points[0]
        self.__iter_p2: BezierPoint2 = self.__points[1]
        return self

    def __next__(self) -> Vec2:
        if self.__iter_sect_i == self.segments:
            if self.closed:
                if self.__iter_sect == len(self.__points):
                    raise StopIteration
            else:
                if self.__iter_sect == len(self.__points) - 1:
                    raise StopIteration
            self.__iter_sect_i = 0
            self.__iter_sect += 1
            self.__iter_p1 = self.__iter_p2
            self.__iter_p2 = self.__points[self.__iter_sect % len(self.__points)]

        t: float = self.__iter_sect_i / (self.segments - 1)

        self.__iter_sect_i += 1

        return bezier_2_cubic(self.__iter_p1.point, self.__iter_p1.anchor_1,
                              self.__iter_p2.anchor_2, self.__iter_p2.point, t)

    def __repr__(self):
        res: str = "BezierCurve2:\n"
        res += "[\n closed : %s\n" % self.closed
        res += " points :\n"
        for i in range(0, len(self.__points)):
            res += "%s\n" % self.__points[i]
        res += "]\n"
        return res

    def __str__(self):
        res: str = ""
        res += "[\n closed : %s\n" % self.closed
        res += " points :\n"
        for i in range(0, len(self.__points)):
            res += "%s\n" % self.__points[i]
        res += "]\n"
        return res

    @property
    def n_control_points(self) -> int:
        return len(self.__points)

    @property
    def points(self) -> [BezierPoint2]:
        return self.__points

    @property
    def segments(self) -> int:
        return self.__sections_per_seg

    @segments.setter
    def segments(self, val: int) -> None:
        if val < 3:
            self.__sections_per_seg = 3
            return
        self.__sections_per_seg = val

    def __in_range(self, point_id: int) -> bool:
        if point_id < 0:
            return False
        if point_id >= len(self.__points):
            return False
        return True

    def add_point(self, p: Vec2, smooth: bool = True) -> None:

        point: BezierPoint2 = BezierPoint2(p)
        point.smooth = smooth
        self.__points.append(BezierPoint2(p))

        if len(self.__points) < 2:
            return

        if len(self.__points) == 2:
            dir_: Vec2 = self.__points[1].point - self.__points[0].point
            self.__points[0].align_anchors(dir_)
            self.__points[1].align_anchors(dir_)
            return

        pid = len(self.__points) - 1
        dir_31: Vec2 = self.__points[pid].point - self.__points[pid - 2].point
        dir_21: Vec2 = self.__points[pid].point - self.__points[pid - 1].point
        self.__points[pid - 1].align_anchors(dir_31)
        self.__points[pid].align_anchors(dir_21)

    def insert_point(self, p: Vec2, pid: int) -> None:
        if pid < 0:
            return
        if self.n_control_points == 0 or \
           self.n_control_points == 1 or \
           self.n_control_points == pid:
            self.add_point(p)
            return

        _dir: Vec2

        if pid == 0:
            _dir = self.__points[pid].point - self.__points[self.n_control_points - 1].point
        else:
            _dir = self.__points[pid].point - self.__points[pid - 1].point

        pt: BezierPoint2 = BezierPoint2(p)

        pt.align_anchors(_dir)

        self.__points.insert(pid, pt)

    def set_flow(self) -> None:
        if self.n_control_points < 3:
            return
        p_prev: BezierPoint2 = self.__points[self.n_control_points - 1]
        p_curr: BezierPoint2
        p_next: BezierPoint2
        for i in range(0, self.n_control_points):
            p_curr = self.__points[i]
            p_next = self.__points[(i + 1) % self.n_control_points]
            p_curr.align_anchors(p_next.point - p_prev.point)
            p_prev = p_curr

    def rem_point(self, pid: int) -> None:
        if not self.__in_range(pid):
            return
        del self.__points[pid]

    def move_point(self, pid: int, pos: Vec2) -> None:
        if not self.__in_range(pid):
            return
        self.__points[pid].point = pos

    def get_point(self, pid: int) -> Vec2:
        if not self.__in_range(pid):
            return Vec2(0)
        return self.__points[pid].point

    def set_anchor_1(self, pid: int, pos: Vec2) -> None:
        if not self.__in_range(pid):
            return
        self.__points[pid].anchor_1 = pos

    def set_anchor_2(self, pid: int, pos: Vec2) -> None:
        if not self.__in_range(pid):
            return
        self.__points[pid].anchor_2 = pos

    def align_anchors(self, pid: int, pos: Vec2, weight: float = 1) -> None:
        if not self.__in_range(pid):
            return
        self.__points[pid].align_anchors(pos, weight)

    def curve_value(self, pid: int, t: float) -> Vec2:
        if not self.__in_range(pid):
            return Vec2(0, 0)
        if not self.closed:
            if pid == len(self.__points) - 1:
                return self.__points[pid].point
        t = min(max(t, 0.0), 1.0)
        p1: BezierPoint2 = self.__points[pid]
        p2: BezierPoint2 = self.__points[(pid + 1) % len(self.__points)]
        return bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)

    def curve_tangent(self, pid: int, t: float) -> Vec2:
        if not self.__in_range(pid):
            return Vec2(0, 0)
        if not self.closed:
            if pid == len(self.__points) - 1:
                return self.__points[pid].point
        t = min(max(t, 0.0), 1.0)
        p1: BezierPoint2 = self.__points[pid]
        p2: BezierPoint2 = self.__points[(pid + 1) % len(self.__points)]
        return bezier_2_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)

    def curve_normal(self, pid: int, t: float) -> Vec2:
        if not self.__in_range(pid):
            return Vec2(0, 0)
        dt: float = 1.0 / self.__sections_per_seg

        p1: BezierPoint2 = self.__points[pid % len(self.__points)]
        p2: BezierPoint2 = self.__points[(pid + 1) % len(self.__points)]

        if t + dt <= 1:
            return math_utils.perpendicular_2(
                bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t + dt) -
                bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t))

        return math_utils.perpendicular_2(
            bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t) -
            bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t - dt))

    def curve_values(self, step: float = 0.01):
        if self.n_control_points < 2:
            raise StopIteration

        point_id: int = 0
        t: float = 0
        p1: BezierPoint2 = self.__points[0]
        p2: BezierPoint2 = self.__points[1]
        while True:
            if t >= 1.0:
                t = 0.0
                point_id += 1
                if point_id == self.n_control_points:
                    break
                p1 = self.__points[point_id]
                p2 = self.__points[(point_id + 1) % len(self.__points)]
            yield bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)
            t += step

        if self.closed:
            t = 0
            p1: BezierPoint2 = self.__points[self.n_control_points - 1]
            p2: BezierPoint2 = self.__points[0]
            while t >= 1.0:
                yield bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)
                t += step

    def curve_normals(self, step: float = 0.01):
        if self.n_control_points < 2:
            raise StopIteration

        point_id: int = 0
        t: float = 0
        p1: BezierPoint2 = self.__points[0]
        p2: BezierPoint2 = self.__points[1]
        while True:
            if t >= 1.0:
                t = 0.0
                point_id += 1
                if point_id == self.n_control_points:
                    break
                p1 = self.__points[point_id]
                p2 = self.__points[(point_id + 1) % len(self.__points)]
            yield math_utils.perpendicular_2(
                bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t + step) -
                bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t))
            t += step

        if self.closed:
            t = 0
            p1: BezierPoint2 = self.__points[self.n_control_points - 1]
            p2: BezierPoint2 = self.__points[0]
            while t >= 1.0:
                yield math_utils.perpendicular_2(
                    bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t + step) -
                    bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t))
                t += step

    def curve_tangents(self, step: float = 0.01):
        if self.n_control_points < 2:
            raise StopIteration

        point_id: int = 0
        t: float = 0
        p1: BezierPoint2 = self.__points[0]
        p2: BezierPoint2 = self.__points[1]
        while True:
            if t >= 1.0:
                t = 0.0
                point_id += 1
                if point_id == self.n_control_points:
                    break
                p1 = self.__points[point_id]
                p2 = self.__points[(point_id + 1) % len(self.__points)]
            yield bezier_2_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)
            t += step

        if self.closed:
            t = 0
            p1: BezierPoint2 = self.__points[self.n_control_points - 1]
            p2: BezierPoint2 = self.__points[0]
            while t >= 1.0:
                yield bezier_2_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)
                t += step
    # def outline(self, value:):
    # def __repr__(self): return "<vec2 x:%s y:%s>" % (self.xy[0], self.xy[1])
    # def __str__(self): return "[%s, %s]" % (self.xy[0], self.xy[1])
