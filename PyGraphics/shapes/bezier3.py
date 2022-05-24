from vmath import mathUtils
from vmath.mathUtils import Vec3, Mat4, Mat3


def bezier_3_quadratic(p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    return mathUtils.lerp_vec_3(mathUtils.lerp_vec_3(p1, p2, t), mathUtils.lerp_vec_3(p2, p3, t), t)


def bezier_3_cubic(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3, t: float) -> Vec3:
    return mathUtils.lerp_vec_3(bezier_3_quadratic(p1, p2, p3, t), bezier_3_quadratic(p2, p3, p4, t), t)


def bezier_mat_3_quadratic(p1: Mat3, p2: Mat3, p3: Mat3, t: float) -> Mat3:
    return mathUtils.lerp_mat_3(mathUtils.lerp_mat_3(p1, p2, t), mathUtils.lerp_mat_3(p2, p3, t), t)


def bezier_mat_3_cubic(p1: Mat3, p2: Mat3, p3: Mat3, p4: Mat3, t: float) -> Mat3:
    return mathUtils.lerp_mat_3(bezier_mat_3_quadratic(p1, p2, p3, t), bezier_mat_3_quadratic(p2, p3, p4, t), t)


def bezier_mat_4_quadratic(p1: Mat4, p2: Mat4, p3: Mat4, t: float) -> Mat4:
    return mathUtils.lerp_mat_4(mathUtils.lerp_mat_4(p1, p2, t), mathUtils.lerp_mat_4(p2, p3, t), t)


def bezier_mat_4_cubic(p1: Mat4, p2: Mat4, p3: Mat4, p4: Mat4, t: float) -> Mat4:
    return mathUtils.lerp_mat_4(bezier_mat_4_quadratic(p1, p2, p3, t), bezier_mat_4_quadratic(p2, p3, p4, t), t)


class BezierPoint3(object):
    def __init__(self, p: Vec3):
        self.__point: Vec3 = p
        self.__anchor_1: Vec3 = p + Vec3(0.125, 0, 0.125)
        self.__anchor_2: Vec3 = p + Vec3(-0.125, 0, -0.125)
        self.smooth: bool = True

    def __repr__(self):
        res: str = "BezierPoint2:\n"
        res += "[\n point   : %s,\n" % self.__point
        res += " smooth  : %s,\n" % self.smooth
        res += " anchor_1: %s,\n" % self.__anchor_1
        res += " anchor_2: %s\n]" % self.__anchor_2
        return res

    def __str__(self):
        res: str = ""
        res += "[\n point   : %s,\n" % self.__point
        res += " smooth  : %s,\n" % self.smooth
        res += " anchor_1: %s,\n" % self.__anchor_1
        res += " anchor_2: %s\n]" % self.__anchor_2
        return res

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


class BezierCurve3(object):
    def __init__(self):
        self.__sections_per_seg: int = 32
        self.__points: [BezierPoint3] = []
        self.closed: bool = False

    def __iter__(self):
        if len(self.__points) <= 1:
            raise StopIteration
        self.__iter_sect_i: int = 0
        self.__iter_sect: int = 1
        self.__iter_p1: BezierPoint3 = self.__points[0]
        self.__iter_p2: BezierPoint3 = self.__points[1]
        return self

    def __next__(self) -> Vec3:
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

        return bezier_3_cubic(self.__iter_p1.point, self.__iter_p1.anchor_1,
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
    def points(self) -> [BezierPoint3]:
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

    def add_point(self, p: Vec3, smooth: bool = True) -> None:

        point: BezierPoint3 = BezierPoint3(p)
        point.smooth = smooth
        self.__points.append(BezierPoint3(p))

        if len(self.__points) < 2:
            return

        if len(self.__points) == 2:
            dir_: Vec3 = self.__points[1].point - self.__points[0].point
            self.__points[0].align_anchors(dir_)
            self.__points[1].align_anchors(dir_)
            return

        pid = len(self.__points) - 1
        dir_31: Vec3 = self.__points[pid].point - self.__points[pid - 2].point
        dir_21: Vec3 = self.__points[pid].point - self.__points[pid - 1].point
        self.__points[pid - 1].align_anchors(dir_31)
        self.__points[pid].align_anchors(dir_21)

    def insert_point(self, p: Vec3, pid: int) -> None:
        if pid < 0:
            return
        if self.n_control_points == 0 or \
                self.n_control_points == 1 or \
                self.n_control_points == pid:
            self.add_point(p)
            return

        _dir: Vec3

        if pid == 0:
            _dir = self.__points[pid].point - self.__points[self.n_control_points - 1].point
        else:
            _dir = self.__points[pid].point - self.__points[pid - 1].point

        pt: BezierPoint3 = BezierPoint3(p)

        pt.align_anchors(_dir)

        self.__points.insert(pid, pt)

    def set_flow(self) -> None:
        p_prev: BezierPoint3 = self.__points[self.n_control_points - 1]
        p_curr: BezierPoint3
        p_next: BezierPoint3
        for i in range(0, self.n_control_points):
            p_curr = self.__points[i]
            p_next = self.__points[(i + 1) % self.n_control_points]
            p_curr.align_anchors(p_next.point - p_prev.point)
            p_prev = p_curr

    def rem_point(self, pid: int) -> None:
        if not self.__in_range(pid):
            return
        del self.__points[pid]

    def move_point(self, pid: int, pos: Vec3) -> None:
        if not self.__in_range(pid):
            return
        self.__points[pid].point = pos

    def get_point(self, pid: int) -> Vec3:
        if not self.__in_range(pid):
            return None
        return self.__points[pid].point

    def set_anchor_1(self, pid: int, pos: Vec3) -> None:
        if not self.__in_range(pid):
            return
        self.__points[pid].anchor_1 = pos

    def set_anchor_2(self, pid: int, pos: Vec3) -> None:
        if not self.__in_range(pid):
            return
        self.__points[pid].anchor_2 = pos

    def align_anchors(self, pid: int, pos: Vec3, weight: float = 1) -> None:
        if not self.__in_range(pid):
            return
        self.__points[pid].align_anchors(pos, weight)

    def curve_value(self, pid: int, t: float) -> Vec3:
        if not self.__in_range(pid):
            return Vec3(0, 0, 0)
        if not self.closed:
            if pid == len(self.__points) - 1:
                return self.__points[pid].point
        t = min(max(t, 0.0), 1.0)
        p1: BezierPoint3 = self.__points[pid]
        p2: BezierPoint3 = self.__points[(pid + 1) % len(self.__points)]
        return bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)

    def sect_normal(self, pid: int, t: float) -> Vec3:
        if not self.__in_range(pid):
            return Vec3(0, 0, 0)
        dt: float = 1.0 / self.__sections_per_seg

        p1: BezierPoint3 = self.__points[pid % len(self.__points)]
        p2: BezierPoint3 = self.__points[(pid + 1) % len(self.__points)]

        if t + dt <= 1:
            return mathUtils.perpendicular_3(
                bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t + dt) -
                bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t))

        return mathUtils.perpendicular_3(
            bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t) -
            bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t - dt))

    # def outline(self, value:):
    # def __repr__(self): return "<vec2 x:%s y:%s>" % (self.xy[0], self.xy[1])
    # def __str__(self): return "[%s, %s]" % (self.xy[0], self.xy[1])
