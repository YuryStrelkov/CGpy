from cgeo.mutils import numerical_precision
from typing import Tuple, Callable, List
from cgeo.vectors import Vec2
import numpy as np

Vector2Int = Tuple[int, int]
Circle = Tuple[Tuple[float, float], float]  # cx, cy, r
Section = Tuple[Vec2, Vec2]
_circles_array: List[Circle] = []


def _circles(x_: float, y_: float) -> float:
    """
    Функция вида F(x,y) = c
    :param x_:
    :param y_:
    :return:
    """
    return np.cos(x_ * np.pi * 5) * np.sin(y_ * np.pi * 5)
    # return np.cos(np.sqrt((1.4 * x_ - 0.5) ** 2 + (0.5 * y_ - 0.5) ** 2) * np.pi * 10)
    # return np.cos(np.sqrt((x_ - 0.5) ** 2 + (y_ - 0.5) ** 2) * np.pi)
    # return sum((r / np.sqrt((x_ - pxpy[0]) * (x_ - pxpy[0]) +
    #                         (y_ - pxpy[1]) * (y_ - pxpy[1])) for pxpy, r in _circles_array)) / len(_circles)


def _field_view(x_: np.ndarray, y_: np.ndarray, field_func: Callable[[float, float], float] = _circles) -> np.ndarray:
    return np.array([[field_func(x_i, y_j) for x_i in x_.flat] for y_j in y_.flat])


# @numba.njit(fastmath=True)
def _rect_bounds(x_arg: float, y_arg: float, h0: float = 1.0) -> float:
    return max((x_arg ** 2 + h0 if x_arg <= 0.0 else 0.0 + (x_arg - 1.0) ** 2 + h0 if x_arg >= 1.0 else 0.0),
               (y_arg ** 2 + h0 if y_arg <= 0.0 else 0.0 + (y_arg - 1.0) ** 2 + h0 if y_arg >= 1.0 else 0.0))


# @numba.njit(fastmath=True)
def _bi_lin_interp(pt_x: float, pt_y: float, points: np.ndarray) -> float:
    """
    Билинейная иетерполяция точки (x,y)
    :param pt_x: x - координата точки
    :param pt_y: y - координата точки
    :param points: двухмерный список узловых точек
    :return:
    """
    rows = points.shape[0]
    cols = points.shape[1]

    col_ = int(pt_x * (cols - 1))
    row_ = int(pt_y * (rows - 1))

    col_1 = min(col_ + 1, cols - 1)
    row_1 = min(row_ + 1, rows - 1)
    # q11 = nodes[row_, col_]

    # q00____q01
    # |       |
    # |       |
    # q10____q11

    dx_ = 1.0 / (cols - 1.0)
    dy_ = 1.0 / (rows - 1.0)

    tx = (pt_x - dx_ * col_) / dx_
    ty = (pt_y - dy_ * row_) / dy_

    q00: float = points[col_, row_]
    q01: float = points[col_1, row_]
    q10: float = points[col_, row_1]
    q11: float = points[col_1, row_1]

    return q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)


def march_squares_2d(f_map: np.ndarray, threshold: float = 1.0, resolution: Vector2Int = (128, 128)) -> List[Section]:
    def lin_interp(_a: float, _b: float, t) -> float:
        return _a + (_b - _a) * t

    rows, cols = max(resolution[1], 3), max(resolution[0], 3)
    cols_ = cols - 1
    rows_ = cols - 1
    dx = 1.0 / cols_
    dy = 1.0 / rows_
    t_val: float
    state: int
    row: float
    col: float
    d_t: float
    p1: float
    p2: float
    p3: float
    p4: float

    shape: List[Section] = []

    for i in range(cols_ * rows_):

        state = 0
        row, col = divmod(i, cols_)
        row *= dy
        col *= dx

        a_val = max(_bi_lin_interp(col, row, f_map), _rect_bounds(col, row, threshold))
        b_val = max(_bi_lin_interp(col + dx, row, f_map), _rect_bounds(col + dx, row, threshold))
        c_val = max(_bi_lin_interp(col + dx, row + dy, f_map), _rect_bounds(col + dx, row + dy, threshold))
        d_val = max(_bi_lin_interp(col, row + dy, f_map), _rect_bounds(col, row + dy, threshold))

        state += 8 if a_val >= threshold else 0
        state += 4 if b_val >= threshold else 0
        state += 2 if c_val >= threshold else 0
        state += 1 if d_val >= threshold else 0

        if state == 0 or state == 15:
            continue
        # без интерполяции
        # a = (col + dx * 0.5, row           )
        # b = (col + dx,       row + dy * 0.5)
        # c = (col + dx * 0.5, row + dy      )
        # d = (col,            row + dy * 0.5)

        d_t = b_val - a_val
        if np.abs(d_t) <= numerical_precision:
            a = Vec2(lin_interp(col, col + dx, np.sign(threshold - a_val)), row)
        else:
            t_val = (threshold - a_val) / d_t
            a = Vec2(lin_interp(col, col + dx, t_val), row)

        d_t = c_val - b_val
        if np.abs(d_t) <= numerical_precision:
            b = Vec2(col + dx, lin_interp(row, row + dy, np.sign(threshold - b_val)))
        else:
            t_val = (threshold - b_val) / d_t
            b = Vec2(col + dx, lin_interp(row, row + dy, t_val))

        d_t = c_val - d_val
        if np.abs(d_t) <= numerical_precision:
            c = Vec2(lin_interp(col, col + dx, np.sign(threshold - d_val)), row + dy)
        else:
            t_val = (threshold - d_val) / d_t
            c = Vec2(lin_interp(col, col + dx, t_val), row + dy)

        d_t = d_val - a_val
        if np.abs(d_t) <= numerical_precision:
            d = Vec2(col, lin_interp(row, row + dy, np.sign(threshold - a_val)))
        else:
            t_val = (threshold - a_val) / d_t
            d = Vec2(col, lin_interp(row, row + dy, t_val))

        while True:
            if state == 1:
                shape.append((c, d))
                break
            if state == 2:
                shape.append((b, c))
                break
            if state == 3:
                shape.append((b, d))
                break
            if state == 4:
                shape.append((a, b))
                break
            if state == 5:
                shape.append((a, d))
                shape.append((b, c))
                break
            if state == 6:
                shape.append((a, c))
                break
            if state == 7:
                shape.append((a, d))
                break
            if state == 8:
                shape.append((a, d))
                break
            if state == 9:
                shape.append((a, c))
                break
            if state == 10:
                shape.append((a, b))
                shape.append((c, d))
                break
            if state == 11:
                shape.append((a, b))
                break
            if state == 12:
                shape.append((b, d))
                break
            if state == 13:
                shape.append((b, c))
                break
            if state == 14:
                shape.append((c, d))
                break
            break
    return shape


def _dist(a: Vec2, b: Vec2) -> float:
    return (a - b).magnitude


def _clean_up_shape(shape: List[Vec2], clean_up_threshold: float = 1e-12) -> List[Vec2]:
    indices_to_delete = []
    for i in range(len(shape) - 1):
        if abs(Vec2.cross(shape[i] - shape[i - 1], shape[i] - shape[i + 1])) < clean_up_threshold:
            indices_to_delete.insert(0, i)
    for i in indices_to_delete:
        del shape[i]
    if _dist(shape[-1], shape[0]) > 1e-3:
        shape.append(shape[0])
    return shape


def connect_sects(sects: List[Section], clean_up_threshold: float = 1e-12) -> List[List[Vec2]]:
    if len(sects) < 3:
        return [[Vec2(0.0, 0.0)]]
    sect_checked = {}
    sect_checked_amount = 0
    shapes = []
    while sect_checked_amount <= len(sects):
        shape = []
        prev_sect_id = -1
        start_sect = -1
        loop_closed = False
        while not loop_closed:
            for sect_id, (p1, p2) in enumerate(sects):

                if sect_id in sect_checked:
                    if sect_checked[sect_id] == 3:
                        continue

                if len(shape) == 0:
                    shape.append(p1)
                    shape.append(p2)
                    start_sect = sect_id
                    prev_sect_id = sect_id
                    sect_checked.update({sect_id: 0})
                    continue

                while True:
                    if _dist(p1, shape[-1]) < numerical_precision:
                        if _dist(p1, shape[0]) < numerical_precision:
                            loop_closed = True
                            break
                        shape.append(p2)
                        sect_checked.update({sect_id: 1})
                        break

                    if _dist(p2, shape[-1]) < numerical_precision:
                        if _dist(p2, shape[0]) < numerical_precision:
                            loop_closed = True
                            break
                        shape.append(p1)
                        sect_checked.update({sect_id: 2})
                        break
                    break

                if sect_id in sect_checked:
                    if sect_checked[prev_sect_id] == 2:
                        sect_checked[prev_sect_id] |= 1

                    if sect_checked[prev_sect_id] == 1:
                        sect_checked[prev_sect_id] |= 2

                    if sect_checked[prev_sect_id] == 3:
                        sect_checked_amount += 1

                    prev_sect_id = sect_id

                if loop_closed:
                    sect_checked[start_sect] = 3
                    sect_checked[sect_id] = 3
                    sect_checked_amount += 2
                    break

        shapes.append(shape)

        for i in range(len(shapes)):
            shapes[i] = _clean_up_shape(shapes[i], clean_up_threshold)

    return shapes


def isoline(f_map: np.ndarray, threshold: float = 1.0, resolution: Vector2Int = (128, 128)) -> \
        List[Tuple[List[float], List[float]]]:
    sects = march_squares_2d(f_map, threshold=threshold, resolution=resolution)
    shapes = connect_sects(sects)
    return [([xy[0] for xy in shape], [xy[1] for xy in shape]) for shape in shapes]


def isoline_of_vect(f_map: np.ndarray, threshold: float = 1.0, resolution: Vector2Int = (128, 128)) -> \
        List[List[Vec2]]:
    sects = march_squares_2d(f_map, threshold=threshold, resolution=resolution)
    shapes = connect_sects(sects)
    return [[Vec2(xy[0], xy[1]) for xy in shape] for shape in shapes]


"""

if __name__ == "__main__":

    x = np.linspace(-0.0, 1., 256)
    y = np.linspace(-0.0, 1., 256)

    _threshold = 0.50

    _circles_array = \
        [((random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)), random.uniform(0.0, 0.333)) for _ in range(32)]

    f = _field_view(x, y, lambda _x, _y: _circles(_x, _y))

    xys = isoline(f, threshold=_threshold)

    polygons = isoline_of_vect(f, threshold=_threshold)

    # mesh = triangulate_isolines(polygons)

    # tris_mesh.write_obj_mesh(mesh, "polygons.obj")

    area_dist = np.zeros((y.size, x.size,), dtype=float)
    for row in range(area_dist.shape[0]):
        for col in range(area_dist.shape[1]):
            p = Vec2(x[col], y[row])
            area_dist[row, col] = -1.0 if gutils.point_within_polygons(p, polygons) else 0.0
            area_dist[row, col] *= 1.0 / (1.0 + ((p.x - 0.666) ** 2 + (p.y - 0.69) ** 2) * 25)
            # f area_dist[row, col] != 0.0:
                #
            # if dist < area_dist[row, col]:
            #    area_dist[row, col] = dist
    area_dist[0:4, :] = 0.0
    area_dist[-4:-1, :] = 0.0
    area_dist[:, -4:-1] = 0.0
    area_dist[:, 0:4] = 0.0

    area_dist = gauss_blur(area_dist)
    #  area_map = _field_view(x, y, lambda _x, _y: gutils.point_to_polygons_distance(Vec2(_x, _y), polygons)[0])

    # plt.imshow(np.flipud(f.T), extent=[np.amin(x), np.amax(x), np.amin(y), np.amax(y)])
    plt.imshow(np.flipud(area_dist), extent=[np.amin(x), np.amax(x), np.amin(y), np.amax(y)])

"""
"""
    for j in range(len(xys)):
        x, y = xys[j]
        for i in range(len(x) - 1):
            xi, yi = x[i], y[i]
            xi1, yi1 = x[i + 1], y[i + 1]
            cx = (xi + xi1) * 0.5
            cy = (yi + yi1) * 0.5
            n = gutils.perpendicular_2(Vec2(xi1 - xi, yi1 - yi))
            plt.plot([cx, cx + 0.01 * n.x], [cy, cy + 0.01 * n.y], 'g')
"""
"""
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
"""
