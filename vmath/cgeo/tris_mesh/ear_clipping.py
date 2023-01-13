from typing import List, Tuple, Dict
from cgeo.tris_mesh import TrisMesh
from cgeo import Vec2, Vec3
import math
import cgeo


def _clamp_index(index: int, index_min: int, index_max: int) -> int:
    if index < index_min:
        return index_max
    if index > index_max:
        return index_min
    return index


def triangulate_polygon(polygon: List[Vec2]) -> TrisMesh:
    n_points: int
    m_points: int
    angles: List[float]
    triangles: List[Tuple[int, int, int]] = []
    poly_cw: bool = False
    ear_cw: bool = False
    a: Vec2
    b: Vec2
    c: Vec2
    theta: float
    min_ang: float
    max_x: float
    i: int

    n_points = len(polygon)

    if (polygon[0] - polygon[-1]).magnitude < 1e-6:
        n_points -= 1

    m_points = n_points

    i, max_x = max(enumerate(polygon), key=lambda id_val: id_val[1].x)

    h = _clamp_index(i - 1, 0, n_points - 1)
    j = _clamp_index(i + 1, 0, n_points - 1)
    a = polygon[h]
    b = polygon[i]
    c = polygon[j]
    poly_cw = cgeo.cw(a, b, c)

    angles = [0.0 for _ in range(n_points)]

    for k in range(n_points):
        a = polygon[k]
        b = polygon[(k + 1) % n_points]
        c = polygon[(k + 2) % n_points]
        theta = cgeo.angle2(a, b, c)
        ear_cw = cgeo.cw(a, b, c)
        angles[(k + 1) % n_points] = theta if ear_cw == poly_cw else 2 * math.pi - theta

    _polygon: Dict[Vec2, int] = {}
    _tris_index = 0
    _p1: int
    _p2: int
    _p3: int

    for k in range(m_points - 2):
        i, min_ang = min(enumerate(angles), key=lambda id_val: id_val[1])
        h = _clamp_index(i - 1, 0, n_points - 1)
        j = _clamp_index(i + 1, 0, n_points - 1)
        a = polygon[h]
        b = polygon[i]
        c = polygon[j]

        if a in _polygon:
            _p1 = _polygon[a]
        else:
            _p1 = len(_polygon)
            _polygon.update({a: _p1})

        if b in _polygon:
            _p2 = _polygon[b]
        else:
            _p2 = len(_polygon)
            _polygon.update({b: _p2})

        if c in _polygon:
            _p3 = _polygon[c]
        else:
            _p3 = len(_polygon)
            _polygon.update({c: _p3})

        triangles.append((_p1, _p2, _p3))

        # ==================== UPDATE ANGLE k - 1 ====================
        a = polygon[_clamp_index(h - 1, 0, n_points - 1)]
        b = polygon[h]
        c = polygon[j]
        ear_cw = cgeo.cw(a, b, c)
        theta = cgeo.angle2(a, b, c)
        angles[h] = theta if ear_cw == poly_cw else 2 * math.pi - theta
        # ==================== UPDATE ANGLE k + 1 ====================
        a = polygon[h]
        b = polygon[j]
        c = polygon[_clamp_index(j + 1, 0, n_points - 1)]
        ear_cw = cgeo.cw(a, b, c)
        theta = cgeo.angle2(a, b, c)
        angles[j] = theta if ear_cw == poly_cw else 2 * math.pi - theta
        del polygon[i]
        del angles[i]
        n_points -= 1

    mesh = TrisMesh()
    _poly = list(_polygon.keys())
    poly_min, poly_max = cgeo.polygon_bounds(_poly)
    for p in _poly:
        mesh.append_normal(Vec3(0.0, 1.0, 0.0))
        mesh.append_uv(Vec2((p.x  - poly_min.x) / (poly_max.x  - poly_min.x),
                            (p.y  - poly_min.y) / (poly_max.y  - poly_min.y)))
        mesh.append_vertex(Vec3(p.x, 0.0, p.y))
    for tris in triangles:
        mesh.append_face(tris)
    return mesh


def triangulate_polygons(polygons: List[List[Vec2]]) -> TrisMesh:
    mesh = triangulate_polygon(polygons[0])
    for i in range(1, len(polygons)):
        mesh.merge(triangulate_polygon(polygons[i]))
    return mesh


"""
def _test_clipping():
    t = np.linspace(0, 0.8 * 2 * np.pi, 128)
    points = [Vec2(np.sin(ti), np.cos(ti)) for ti in t.flat]
    n_points = len(points)
    for i in range(n_points):
        points.append(points[n_points - 1 - i] * 0.7)
    points.append(points[0])

    mesh = triangulate_polygon(points)

    tris_mesh.write_obj_mesh(mesh, "polygons_test.obj")
    print(np.arccos(0.5))
    x = [v.x for v in points]
    y = [v.y for v in points]
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    _test_clipping()
"""


