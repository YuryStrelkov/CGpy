from typing import List

from core.bezier.bezier_point_3 import BezierPoint3
from core import geometry_utils
from core.vectors import Vec3
from typing import List


class BezierCurve3:
    def __init__(self):
        self.__sections_per_seg: int = 32
        self.__points: List[BezierPoint3] = []
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

        return geometry_utils.bezier_3_cubic(self.__iter_p1.point, self.__iter_p1.anchor_1,
                                         self.__iter_p2.anchor_2, self.__iter_p2.point, t)

    def __str__(self):
        nl = ",\n"
        return f"{{\n" \
               f"\t\"unique_id\"    : {self.unique_id},\n" \
               f"\t\"closed\"       : {self.closed},\n" \
               f"\t\"sects_per_seg\": {self.__sections_per_seg},\n" \
               f"\t\"points\"       : [\n{nl.join(str(pt) for pt in self.__points)}]\n" \
               f"}}"

    @property
    def unique_id(self) -> int:
        return id(self)

    @property
    def n_control_points(self) -> int:
        return len(self.__points)

    @property
    def points(self) -> List[BezierPoint3]:
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
            return Vec3(0)
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
        self.__points[pid].align_anchors(pos)

    def curve_value(self, pid: int, t: float) -> Vec3:
        if not self.__in_range(pid):
            return Vec3(0, 0, 0)
        if not self.closed:
            if pid == len(self.__points) - 1:
                return self.__points[pid].point
        t = min(max(t, 0.0), 1.0)
        p1: BezierPoint3 = self.__points[pid]
        p2: BezierPoint3 = self.__points[(pid + 1) % len(self.__points)]
        return geometry_utils.bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)

    def curve_normal(self, pid: int, t: float) -> Vec3:
        if not self.__in_range(pid):
            return Vec3(0, 0, 0)
        dt: float = 1.0 / self.__sections_per_seg

        p1: BezierPoint3 = self.__points[pid % len(self.__points)]
        p2: BezierPoint3 = self.__points[(pid + 1) % len(self.__points)]

        if t + dt <= 1:
            return geometry_utils.perpendicular_3(
                geometry_utils.bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t + dt) -
                geometry_utils.bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t))

        return geometry_utils.perpendicular_3(
            geometry_utils.bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t) -
            geometry_utils.bezier_3_cubic(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t - dt))

    def curve_tangent(self, pid: int, t: float) -> Vec3:
        if not self.__in_range(pid):
            return Vec3(0, 0)
        if not self.closed:
            if pid == len(self.__points) - 1:
                return self.__points[pid].point
        t = min(max(t, 0.0), 1.0)
        p1: BezierPoint3 = self.__points[pid]
        p2: BezierPoint3 = self.__points[(pid + 1) % len(self.__points)]
        return geometry_utils.bezier_3_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)

    def curve_values(self, step: float = 0.01):
        if self.n_control_points < 2:
            raise StopIteration

        point_id: int = 0
        t: float = 0
        p1: BezierPoint3 = self.__points[0]
        p2: BezierPoint3 = self.__points[1]
        while True:
            if t >= 1.0:
                t = 0.0
                point_id += 1
                if point_id == self.n_control_points:
                    break
                p1 = self.__points[point_id]
                p2 = self.__points[(point_id + 1) % len(self.__points)]
            yield geometry_utils.bezier_3_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)
            t += step

        if self.closed:
            t = 0
            p1: BezierPoint3 = self.__points[self.n_control_points - 1]
            p2: BezierPoint3 = self.__points[0]
            while t >= 1.0:
                yield geometry_utils.bezier_3_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)
                t += step

    def curve_normals(self, step: float = 0.01):
        if self.n_control_points < 2:
            raise StopIteration

        point_id: int = 0
        t: float = 0
        p1: BezierPoint3 = self.__points[0]
        p2: BezierPoint3 = self.__points[1]
        while True:
            if t >= 1.0:
                t = 0.0
                point_id += 1
                if point_id == self.n_control_points:
                    break
                p1 = self.__points[point_id]
                p2 = self.__points[(point_id + 1) % len(self.__points)]
            yield geometry_utils.perpendicular_2(
                geometry_utils.bezier_3_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t + step) -
                geometry_utils.bezier_3_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t))
            t += step

        if self.closed:
            t = 0
            p1: BezierPoint3 = self.__points[self.n_control_points - 1]
            p2: BezierPoint3 = self.__points[0]
            while t >= 1.0:
                yield geometry_utils.perpendicular_2(
                    geometry_utils.bezier_3_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t + step) -
                    geometry_utils.bezier_3_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t))
                t += step

    def curve_tangents(self, step: float = 0.01):
        if self.n_control_points < 2:
            raise StopIteration

        point_id: int = 0
        t: float = 0
        p1: BezierPoint3 = self.__points[0]
        p2: BezierPoint3 = self.__points[1]
        while True:
            if t >= 1.0:
                t = 0.0
                point_id += 1
                if point_id == self.n_control_points:
                    break
                p1 = self.__points[point_id]
                p2 = self.__points[(point_id + 1) % len(self.__points)]
            yield geometry_utils.bezier_3_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)
            t += step

        if self.closed:
            t = 0
            p1: BezierPoint3 = self.__points[self.n_control_points - 1]
            p2: BezierPoint3 = self.__points[0]
            while t >= 1.0:
                yield geometry_utils.bezier_3_tangent(p1.point, p1.anchor_1, p2.anchor_2, p2.point, t)
                t += step
