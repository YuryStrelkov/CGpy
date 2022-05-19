import mathUtils
from mathUtils import Vec2, Vec3, Mat4, Mat3


def bezier_2_quadratic(p1: Vec2, p2: Vec2, p3: Vec2, t: float) -> Vec2:
    return mathUtils.lerp_vec_2(mathUtils.lerp_vec_2(p1, p2, t), mathUtils.lerp_vec_2(p2, p3, t), t)


def bezier_2_cubic(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, t: float) -> Vec2:
    return mathUtils.lerp_vec_2(bezier_2_quadratic(p1, p2, p3, t), bezier_2_quadratic(p2, p3, p4, t), t)


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


class BezierPoint2(object):
    def __init__(self, v: Vec2):
        self.point: Vec2 = v
        perpendicular: Vec2 = mathUtils.perpendicular_2(v) * 0.333
        self.pointAnchor1: Vec2 = v + perpendicular
        self.pointAnchor2: Vec2 = v - perpendicular
        self.smooth: bool = True

    def set_anchor_1_pos(self, pos: Vec2):
        self.pointAnchor1 = pos
        if self.smooth:
            distance = -(self.point - self.pointAnchor2).norm()
            self.pointAnchor2 = distance * (self.pointAnchor1 - self.point).normalize()

    def set_anchor_2_pos(self, pos: Vec2):
        self.pointAnchor2 = pos
        if self.smooth:
            distance = -(self.point - self.pointAnchor1).magnitude
            self.pointAnchor1 = distance * (self.pointAnchor2 - self.point).normalize()

    def set_point(self, pos: Vec2):
        self.pointAnchor1 = self.pointAnchor1 - self.point
        self.pointAnchor2 = self.pointAnchor2 - self.point
        self.point = pos
        self.pointAnchor1 = self.pointAnchor1 + self.point
        self.pointAnchor2 = self.pointAnchor2 + self.point


class BezierCurve2(object):
    def __init__(self):
        self.sections_per_seg = 32
        self.curve: [Vec2] = []
        self.points: [BezierPoint2] = []
        self.closed: bool = False

    def __update_segment(self, pid: int):
        if pid < 0:
            return

        if len(self.points) == 0:
            return

        dt: float = 1.0 / self.sections_per_seg
        p1: BezierPoint2
        p2: BezierPoint2
        if not self.closed:
            if pid == len(self.points) - 1:
                return
            p1: BezierPoint2 = self.points[pid]
            p2: BezierPoint2 = self.points[pid + 1]
        else:
            p1: BezierPoint2 = self.points[pid]
            p2: BezierPoint2 = self.points[(pid + 1) % len(self.points)]

        seg: [Vec2] = []
        for i in range(0, self.sections_per_seg + 1):
            seg.append(bezier_2_cubic(p1.point, p1.pointAnchor1, p2.pointAnchor2, p2.point, dt * i))
        self.curve[pid] = seg

    def add_point(self, p: Vec2, smooth: bool = True):
        point: BezierPoint2 = BezierPoint2(p)
        point.smooth = smooth
        self.points.append(BezierPoint2(p))
        if len(self.points) < 2:
            return
        seg: [Vec2] = []
        self.curve.append(seg)
        self.__update_segment(len(self.points) - 2)

    def rem_point(self, pid: int):
        if pid < 0:
            return
        if pid >= len(self.points):
            return
        del self.points[pid]
        del self.curve[pid]
        self.__update_segment(pid)
        self.__update_segment(pid - 1)

    def move_point(self, pid: int, pos: Vec2):
        if pid < 0:
            return
        if pid >= len(self.points):
            return
        self.points[pid].set_point(pos)
        self.__update_segment(pid)
        self.__update_segment(pid - 1)

    def get_point(self, pid: int) -> Vec2:
        if pid < 0:
            return None
        if pid >= len(self.points):
            return None
        return self.points[pid].point

    # noinspection PyArgumentList
    def set_anchor_1(self, pid: int, pos: Vec2):
        if pid < 0:
            return
        if pid >= len(self.points):
            return
        self.points[pid].set_anchor_1_pos(pos)
        self.__update_segment(pid)
        self.__update_segment(pid - 1)

    def set_anchor_2(self, pid: int, pos: Vec2):
        if pid < 0:
            return
        if pid >= len(self.points):
            return
        self.points[pid].set_anchor_2_pos(pos)
        self.__update_segment(pid)
        self.__update_segment(pid - 1)

    def sect_normal(self, pid: int, t: float) -> Vec2:
        if pid < 0:
            return
        if len(self.points) == 0:
            return
        dt: float = 1.0 / self.sections_per_seg
        p1: BezierPoint2
        p2: BezierPoint2
        if pid == len(self.points) - 1:
            p1: BezierPoint2 = self.points[pid - 1]
            p2: BezierPoint2 = self.points[pid]
        else:
            p1: BezierPoint2 = self.points[pid]
            p2: BezierPoint2 = self.points[pid + 1]

        if t + dt <= 1:
            return mathUtils.perpendicular_2(
                bezier_2_cubic(p1.point, p1.pointAnchor1, p2.pointAnchor2, p2.point, t + dt) -
                bezier_2_cubic(p1.point, p1.pointAnchor1, p2.pointAnchor2, p2.point, t))

        return mathUtils.perpendicular_2(
            bezier_2_cubic(p1.point, p1.pointAnchor1, p2.pointAnchor2, p2.point, t) -
            bezier_2_cubic(p1.point, p1.pointAnchor1, p2.pointAnchor2, p2.point, t - dt))

    # def outline(self, value:):
    # def __repr__(self): return "<vec2 x:%s y:%s>" % (self.xy[0], self.xy[1])
    # def __str__(self): return "[%s, %s]" % (self.xy[0], self.xy[1])


class BezierPoint3(object):
    def __init__(self, v: Vec3):
        self.point: Vec3 = v
        self.pointAnchor1: Vec3 = Vec3(0, 0, 0)
        self.pointAnchor2: Vec3 = Vec3(0, 0, 0)
        self.smooth: bool = True

    def set_anchor_1_pos(self, pos: Vec3):
        self.pointAnchor1 = pos
        if self.smooth:
            distance = -(self.point - self.pointAnchor2).norm()
            self.pointAnchor2 = distance * (self.pointAnchor1 - self.point).normalize()

    def set_anchor_2_pos(self, pos: Vec3):
        self.pointAnchor2 = pos
        if self.smooth:
            distance = -(self.point - self.pointAnchor1).magnitude
            self.pointAnchor1 = distance * (self.pointAnchor2 - self.point).normalize()

    def set_point(self, pos: Vec3):
        self.pointAnchor1 = self.pointAnchor1 - self.point
        self.pointAnchor2 = self.pointAnchor2 - self.point
        self.point = pos
        self.pointAnchor1 = self.pointAnchor1 + self.point
        self.pointAnchor2 = self.pointAnchor2 + self.point


