from computational_geometry import gutils
from computational_geometry.mutils import numerical_precision
from typing import Tuple, Callable, List
from matplotlib import pyplot as plt
from computational_geometry.vectors import Vec2
import numpy as np
import random
import numba

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
    return np.cos(np.sqrt((1.4 * x_ - 0.5) ** 2 + (0.5 * y_ - 0.5) ** 2) * np.pi * 10)
    # return sum((r / np.sqrt((x_ - pxpy[0]) * (x_ - pxpy[0]) +
    #                         (y_ - pxpy[1]) * (y_ - pxpy[1])) for pxpy, r in _circles_array)) / len(_circles)


def _field_view(x_: np.ndarray, y_: np.ndarray, field_func: Callable[[float, float], float] = _circles) -> np.ndarray:
    return np.array([[field_func(x_i, y_j) for x_i in x_.flat] for y_j in y_.flat])


@numba.njit(fastmath=True)
def _rect_bounds(x_arg: float, y_arg: float, h0: float = 1.0) -> float:
    return max((x_arg ** 2 + h0 if x_arg <= 0.0 else 0.0 + (x_arg - 1.0) ** 2 + h0 if x_arg >= 1.0 else 0.0),
               (y_arg ** 2 + h0 if y_arg <= 0.0 else 0.0 + (y_arg - 1.0) ** 2 + h0 if y_arg >= 1.0 else 0.0))


@numba.njit(fastmath=True)
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

        row = (i // cols_) * dy
        col = (i % cols_) * dx

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


def connect_sects(sects: List[Section]) -> List[List[Vec2]]:
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

    return shapes


"""
         public static bool AreLinesIntersecting(Vector2 l1_p1, Vector2 l1_p2, Vector2 l2_p1, Vector2 l2_p2, bool shouldIncludeEndPoints)
        {
            bool isIntersecting = false;

            float denominator = (l2_p2.y - l2_p1.y) * (l1_p2.x - l1_p1.x) - (l2_p2.x - l2_p1.x) * (l1_p2.y - l1_p1.y);

            //Make sure the denominator is > 0, if not the lines are parallel
            if (denominator != 0f)
            {
                float u_a = ((l2_p2.x - l2_p1.x) * (l1_p1.y - l2_p1.y) - (l2_p2.y - l2_p1.y) * (l1_p1.x - l2_p1.x)) / denominator;
                float u_b = ((l1_p2.x - l1_p1.x) * (l1_p1.y - l2_p1.y) - (l1_p2.y - l1_p1.y) * (l1_p1.x - l2_p1.x)) / denominator;

                //Are the line segments intersecting if the end points are the same
                if (shouldIncludeEndPoints)
                {
                    //Is intersecting if u_a and u_b are between 0 and 1 or exactly 0 or 1
                    if (u_a >= 0f && u_a <= 1f && u_b >= 0f && u_b <= 1f)
                    {
                        isIntersecting = true;
                    }
                }
                else
                {
                    //Is intersecting if u_a and u_b are between 0 and 1
                    if (u_a > 0f && u_a < 1f && u_b > 0f && u_b < 1f)
                    {
                        isIntersecting = true;
                    }
                }

            }

            return isIntersecting;
        }

        public static bool IsPointInSpline(List<Point2d> polygonPoints, Vector2 point)
        {
            //Step 1. Find a point outside of the polygon
            //Pick a point with a x position larger than the polygons max x position, which is always outside
            Vector2 maxXPosVertex = polygonPoints[0].point;

            for (int i = 1; i < polygonPoints.Count; i++)
            {
                if (polygonPoints[i].point.x > maxXPosVertex.x)
                {
                    maxXPosVertex = polygonPoints[i].point;
                }
            }

            //The point should be outside so just pick a number to make it outside
            Vector2 pointOutside = maxXPosVertex + new Vector2(10f, 0f);

            //Step 2. Create an edge between the point we want to test with the point thats outside
            Vector2 l1_p1 = point;
            Vector2 l1_p2 = pointOutside;

            //Step 3. Find out how many edges of the polygon this edge is intersecting
            int numberOfIntersections = 0;

            for (int i = 0; i < polygonPoints.Count; i++)
            {
                //Line 2
                Vector2 l2_p1 = polygonPoints[i].point;

                int iPlusOne = (i + 1) % polygonPoints.Count;

                Vector2 l2_p2 = polygonPoints[iPlusOne].point;

                //Are the lines intersecting?
                if (AreLinesIntersecting(l1_p1, l1_p2, l2_p1, l2_p2, true))
                {
                    numberOfIntersections += 1;
                }
            }

            //Step 4. Is the point inside or outside?
            bool isInside = true;

            //The point is outside the polygon if number of intersections is even or 0
            if (numberOfIntersections == 0 || numberOfIntersections % 2 == 0)
            {
                isInside = false;
            }

            return isInside;
        }
"""


def _sort_sections_to_x_y(sects: List[Section]) -> List[Tuple[List[float], List[float]]]:
    """

    """
    shapes = connect_sects(sects)
    return [([xy[0] for xy in shape], [xy[1] for xy in shape]) for shape in shapes]


#    shapes = connect_sects(sects)
#    return [([xy[0] for xy in shapes], [xy[1] for xy in sects])]

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


if __name__ == "__main__":

    x = np.linspace(-0.0, 1., 512)
    y = np.linspace(-0.0, 1., 512)

    _threshold = 0.50

    # _circles = [((random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)), random.uniform(0.0, 0.333)) for _ in range(32)]
    _circles = \
        [((random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)), random.uniform(0.0, 0.333)) for _ in range(32)]

    f = _field_view(x, y, lambda _x, _y: _circles(_x, _y))

    sections = march_squares_2d(f, threshold=_threshold)

    plt.imshow(np.flipud(f.T), extent=[np.amin(x), np.amax(x), np.amin(y), np.amax(y)])

    xys = _sort_sections_to_x_y(sections)

    print(f"len(sections): {len(sections)}")

    print(f"len(xys): {len(xys)}")
    for index, (xi, yi) in enumerate(xys):
        plt.plot(xi, yi, 'r')

    for j in range(len(xys)):
        x, y = xys[j]
        for i in range(len(x) - 1):
            xi, yi = x[i], y[i]
            xi1, yi1 = x[i + 1], y[i + 1]
            cx = (xi + xi1) * 0.5
            cy = (yi + yi1) * 0.5
            n = gutils.perpendicular_2(Vec2(xi1 - xi, yi1 - yi))
            plt.plot([cx, cx + 0.01 * n.x], [cy, cy + 0.01 * n.y], 'g')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
