import numpy as np

from models.model import Model
from models.vertex import Vertex, lerp_vertex
from vmath import mathUtils
from shapes.bezier2 import BezierCurve2
from vmath.mathUtils import Vec2, Vec3
from materials.material import Material
from camera import Camera
from frameBuffer import FrameBuffer
from frameBuffer import RGB
import tkinter as tk
from PIL import ImageTk


# рисование линии, первый вариант алгоритма
def draw_line_1(buffer: FrameBuffer, x0: int, y0: int, x1: int, y1: int,
                color: RGB = RGB(np.uint8(255), np.uint8(255), np.uint8(255)),
                dt: float = 0.01):
    for t in np.arange(0, 1, dt):
        x = x0 * (1 - t) + x1 * t
        y = y0 * (1 - t) + y1 * t
        buffer.set_pixel(int(x), int(y), color)


# рисование линии, второй вариант алгоритма
def draw_line_2(buffer: FrameBuffer, x0: int, y0: int, x1: int, y1: int,
                color: RGB = RGB(np.uint8(255), np.uint8(255), np.uint8(255))):
    for x in range(x0, x1):
        dt = (x - x0) / (float(x1 - x0))
        y = y0 * (1 - dt) + y1 * dt
        buffer.set_pixel(int(x), int(y), color)


def draw_line_3(buffer: FrameBuffer, x0: int, y0: int, x1: int, y1: int,
                color: RGB = RGB(np.uint8(255), np.uint8(255), np.uint8(255))):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        dt = (x - x0) / (float(x1 - x0))
        y = y0 * (1 - dt) + y1 * dt
        if steep:
            buffer.set_pixel(int(y), int(x), color)
        else:
            buffer.set_pixel(int(x), int(y), color)


# рисование линии, третий вариант алгоритма


# рисование линии, четвертый вариант алгоритма (алгоримтм Брезенхема)
def draw_line_4(buffer: FrameBuffer, x0: int, y0: int, x1: int, y1: int,
                color: RGB = RGB(np.uint8(255), np.uint8(255), np.uint8(255))):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0:
        return
    d_error = abs(dy / float(dx))
    error = 0.0
    y = y0
    for x in range(int(x0), int(x1)):
        if steep:
            buffer.set_pixel(y, x, color)
        else:
            buffer.set_pixel(x, y, color)
        error = error + d_error
        if error > 0.5:
            y += 1 if y1 > y0 else -1
            error -= 1


# рисование линии, четвертый вариант алгоритма (алгоримтм Брезенхема)
def draw_line_5(buffer: FrameBuffer, x0: int, y0: int, depth0: float,
                x1: int, y1: int, depth1: float,
                color: RGB = RGB(np.uint8(255), np.uint8(255), np.uint8(255))):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    if x0 > x1:
        depth0, depth1 = depth1, depth0
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0:
        return
    d_error = abs(dy / float(dx))
    error = 0.0
    y = y0
    d_depth = (depth1 - depth0) / (x1 - x0)
    c_depth = depth0 - d_depth
    for x in range(int(x0), int(x1)):
        c_depth += d_depth
        if steep:
            if buffer.set_depth(y, x, c_depth):
                buffer.set_pixel(y, x, color)
        else:
            if buffer.set_depth(x, y, c_depth):
                buffer.set_pixel(x, y, color)
        error = error + d_error
        if error > 0.5:
            y += 1 if y1 > y0 else -1
            error -= 1


def draw_point(buffer: FrameBuffer, x: int, y: int, color: RGB = RGB(np.uint8(255), np.uint8(255), np.uint8(255)),
               depth: float = 0):
    if not buffer.set_depth(x, y, depth):
        return
    buffer.set_pixel(x - 1, y - 1, color)
    buffer.set_pixel(x - 1, y, color)
    buffer.set_pixel(x - 1, y + 1, color)
    buffer.set_pixel(x, y - 1, color)
    buffer.set_pixel(x, y, color)
    buffer.set_pixel(x, y + 1, color)
    buffer.set_pixel(x + 1, y - 1, color)
    buffer.set_pixel(x + 1, y, color)
    buffer.set_pixel(x + 1, y + 1, color)


def point_to_scr_space(buffer: FrameBuffer, pt: Vec3) -> Vec3:
    return Vec3(round(mathUtils.clamp(0, buffer.width - 1, round(buffer.width * (pt.x * 0.5 + 0.5)))),
                round(mathUtils.clamp(0, buffer.height - 1, round(buffer.height * (-pt.y * 0.5 + 0.5)))),
                pt.z)


def point_to_scr_space_2(buffer: FrameBuffer, pt: Vec2) -> Vec2:
    return Vec2(round(mathUtils.clamp(0, buffer.width - 1, round(buffer.width * (pt.x * 0.5 + 0.5)))),
                round(mathUtils.clamp(0, buffer.height - 1, round(buffer.height * (-pt.y * 0.5 + 0.5)))))


# отрисовка одноцветного треугольника(интерполируется только глубина)
def draw_triangle_solid(buffer: FrameBuffer, p0: Vertex, p1: Vertex, p2: Vertex,
                        color: RGB = RGB(np.uint8(255), np.uint8(0), np.uint8(0))):
    if p0.v.y == p1.v.y and p0.v.y == p2.v.y:
        return  # i dont care about degenerate triangles
    # sort the vertices, p0, p1, p2 lower-to-upper (bubblesort yay!)
    if p0.v.y > p1.v.y:
        p0, p1 = p1, p0
    if p0.v.y > p2.v.y:
        p0, p2 = p2, p0
    if p1.v.y > p2.v.y:
        p1, p2 = p2, p1

    total_height: int = round(p2.v.y - p0.v.y)

    for i in range(0, total_height):
        second_half: bool = i > p1.v.y - p0.v.y or p1.v.y == p0.v.y

        segment_height: int = round(p1.v.y - p0.v.y)

        if second_half:
            segment_height: int = round(p2.v.y - p1.v.y)

        if segment_height == 0:
            continue

        alpha: float = float(i) / total_height

        beta: float = 0

        if second_half:
            beta = float(i - (p1.v.y - p0.v.y)) / segment_height
        else:
            beta = float(i / segment_height)  # be careful: with above conditions no division by zero her

        a = lerp_vertex(p0, p2, alpha)

        if second_half:
            b = lerp_vertex(p1, p2, beta)
        else:
            b = lerp_vertex(p0, p1, beta)

        if a.v.x > b.v.x:
            a, b = b, a

        for j in range(round(a.v.x), round(b.v.x)):
            phi: float = 0.0
            if b.v.x == a.v.x:
                phi = 1.0
            else:
                phi = float(j - a.v.x) / float(b.v.x - a.v.x)
            p: Vertex = lerp_vertex(a, b, phi)
            zx, xy = round(p.v.x), round(p.v.y)
            if buffer.set_depth(zx, xy, p.v.z):
                col_shading: float = mathUtils.clamp(0.0, 1.0, mathUtils.vectors.dot3(p.n, Vec3(0.333, 0.333, 0.333)))
                buffer.set_pixel(zx, xy, RGB(color.r * col_shading, color.g * col_shading, color.b * col_shading))


# отрисовка треугольника(интерполируется только глубина, нормали, барицентрические координаты)
def draw_triangle_shaded(buffer: FrameBuffer, p0: Vertex, p1: Vertex, p2: Vertex,
                         mat: Material):  # позиции(в прострастве экрана) вершин треугольника
    if p0.v.y == p1.v.y and p0.v.y == p2.v.y:
        return  # i dont care about degenerate triangles
    # sort the vertices, p0, p1, p2 lower-to-upper (bubblesort yay!)
    if p0.v.y > p1.v.y:
        p0, p1 = p1, p0

    if p0.v.y > p2.v.y:
        p0, p2 = p2, p0

    if p1.v.y > p2.v.y:
        p1, p2 = p2, p1

    total_height: int = round(p2.v.y - p0.v.y)

    for i in range(0, total_height):
        second_half: bool = i > p1.v.y - p0.v.y or p1.v.y == p0.v.y

        segment_height: int = round(p1.v.y - p0.v.y)

        if second_half:
            segment_height: int = round(p2.v.y - p1.v.y)

        if segment_height == 0:
            continue

        alpha: float = float(i) / total_height

        beta: float = 0

        if second_half:
            beta = float(i - (p1.v.y - p0.v.y)) / segment_height
        else:
            beta = float(i / segment_height)  # be careful: with above conditions no division by zero her
        a = lerp_vertex(p0, p2, alpha)
        if second_half:
            b = lerp_vertex(p1, p2, beta)
        else:
            b = lerp_vertex(p0, p1, beta)
        if a.v.x > b.v.x:
            a, b = b, a

        for j in range(round(a.v.x), round(b.v.x)):
            phi: float = 0.0
            if b.v.x == a.v.x:
                phi = 1.0
            else:
                phi = float(j - a.v.x) / float(b.v.x - a.v.x)
            p = lerp_vertex(a, b, phi)
            ix, jy = round(p.v.x), round(p.v.y)
            if buffer.set_depth(ix, jy, p.v.z):
                col: RGB = mat.diff_color(p.uv)
                col_shading: float = mathUtils.clamp(0.0, 1.0, mathUtils.vectors.dot3(p.n, Vec3(0.333, 0.333, 0.333)))
                buffer.set_pixel(ix, jy, RGB(col.r * col_shading, col.g * col_shading, col.b * col_shading))


def draw_bezier(buffer: FrameBuffer, curve: BezierCurve2, color: RGB = RGB(np.uint8(0), np.uint8(0), np.uint8(255))):
    p1 = point_to_scr_space_2(buffer, curve.get_point(0))
    for pt in curve:
        p2 = point_to_scr_space_2(buffer, pt)
        # draw_point(buffer, round(p1.x), round(p1.y), RGB(np.uint8(255), np.uint8(0), np.uint8(0)), -1)
        draw_line_4(buffer, round(p1.x), round(p1.y), round(p2.x), round(p2.y), color)
        p1 = p2

    for point in curve.points:
        a1 = point_to_scr_space_2(buffer, point.anchor_1)
        a2 = point_to_scr_space_2(buffer, point.anchor_2)
        p = point_to_scr_space_2(buffer, point.point)
        draw_line_4(buffer, round(a1.x), round(a1.y), round(a2.x),
                    round(a2.y), RGB(np.uint8(255), np.uint8(255), np.uint8(255)))
        draw_point(buffer, round(a1.x), round(a1.y), RGB(np.uint8(0), np.uint8(0), np.uint8(255)))
        draw_point(buffer, round(a2.x), round(a2.y), RGB(np.uint8(0), np.uint8(0), np.uint8(255)))
        draw_point(buffer, round(p.x), round(p.y), RGB(np.uint8(255), np.uint8(0), np.uint8(0)))


################
#### MODELS ####
################
def draw_model_edges(buffer: FrameBuffer, model: Model, cam: Camera = None,
                     color: RGB = RGB(np.uint8(0), np.uint8(0), np.uint8(0))):
    if cam is None:
        cam = Camera()
        cam.look_at(model.min_world_space, model.max_world_space * 1.5)
    # направление освещения совпадает с направлением взгляда камеры
    forward = cam.front

    for i in range(model.meshes_count):
        for tris in model.triangles_world_space(i):
            tris.camera_screen_transform(cam, buffer)
            a = -mathUtils.vectors.dot3(tris.n1, forward)
            b = -mathUtils.vectors.dot3(tris.n2, forward)
            c = -mathUtils.vectors.dot3(tris.n3, forward)

            if a > 0 or b > 0:
                draw_line_4(buffer, round(tris.p1.x), round(tris.p1.y), round(tris.p2.x), round(tris.p2.y), color)
            if a > 0 or c > 0:
                draw_line_4(buffer, round(tris.p1.x), round(tris.p1.y), round(tris.p3.x), round(tris.p3.y), color)
            if b > 0 or c > 0:
                draw_line_4(buffer, round(tris.p2.x), round(tris.p2.y), round(tris.p3.x), round(tris.p3.y), color)


# отрисовка вершин
def draw_model_vertices(buffer: FrameBuffer, model: Model, cam: Camera = None,
                        color: RGB = RGB(np.uint8(0), np.uint8(0), np.uint8(255))):
    if cam is None:
        cam = Camera()
        cam.look_at(model.min_world_space, model.max_world_space * 1.5)
    # направление освещения совпадает с направлением взгляда камеры
    forward = cam.front

    for i in range(model.meshes_count):
        for tris in model.triangles_world_space(i):
            tris.camera_screen_transform(cam, buffer)
            a = -mathUtils.vectors.dot3(tris.n1, forward)
            b = -mathUtils.vectors.dot3(tris.n2, forward)
            c = -mathUtils.vectors.dot3(tris.n3, forward)
            if a > 0:
                draw_point(buffer, round(tris.p1.x), round(tris.p1.y), color, 0)
            if b > 0:
                draw_point(buffer, round(tris.p2.x), round(tris.p2.y), color, 0)
            if c > 0:
                draw_point(buffer, round(tris.p3.x), round(tris.p3.y), color, 0)


# рисует полигональную сетку интерполируя только по глубине и заливает одним цветом
def draw_model_solid_color(buffer: FrameBuffer, model: Model, cam: Camera = None,
                           color: RGB = RGB(np.uint8(255), np.uint8(200), np.uint8(125))):
    # направление освещения совпадает с направлением взгляда камеры
    if cam is None:
        cam = Camera()
        cam.look_at(model.min_world_space, model.max_world_space * 1.5)
        # направление освещения совпадает с направлением взгляда камеры
    forward = cam.front

    for i in range(model.meshes_count):
        for tris in model.triangles_world_space(i):

            a = mathUtils.vectors.dot3(tris.n1, forward)
            b = mathUtils.vectors.dot3(tris.n2, forward)
            c = mathUtils.vectors.dot3(tris.n3, forward)
            # треугольник к нам задом(back-face culling)

            if a > 0 and b > 0 and c > 0:
                continue

            tris.camera_screen_transform(cam, buffer)

            draw_triangle_solid(buffer, tris.vertex1, tris.vertex2, tris.vertex3, color)


def draw_model_shaded(buffer: FrameBuffer, model: Model, cam: Camera = None):
    if model.materials_count == 0:
        draw_model_solid_color(buffer, model)
        return

    if cam is None:
        cam = Camera()
        cam.look_at(model.min_world_space, model.max_world_space * 1.5)
        # направление освещения совпадает с направлением взгляда камеры
    forward = cam.front

    for i in range(model.meshes_count):
        mat = model.get_material(min(i, model.materials_count - 1))
        for tris in model.triangles_world_space(i):

            a = mathUtils.vectors.dot3(tris.n1, forward)
            b = mathUtils.vectors.dot3(tris.n2, forward)
            c = mathUtils.vectors.dot3(tris.n3, forward)
            # треугольник к нам задом(back-face culling)

            if a > 0 and b > 0 and c > 0:
                continue

            tris.camera_screen_transform(cam, buffer)

            draw_triangle_shaded(buffer, tris.vertex1, tris.vertex2, tris.vertex3, mat)


import threading

# ГУЙ
debugWindow = None
debugWindowLabel = None


def create_image_window(fb: FrameBuffer):
    global debugWindow
    global debugWindowLabel
    if not (debugWindow is None):
        return
    debugWindow = tk.Tk()
    debugWindow.title("Image Viewer")
    img = ImageTk.PhotoImage(fb.frame_buffer_image)
    debugWindow.geometry(str(img.height() + 3) + "x" + str(img.width() + 3))
    debugWindowLabel = tk.Label(image=img)
    debugWindowLabel.pack(side="bottom", fill="both", expand="yes")
    while 'normal' == debugWindow.state():
        try:
            debugWindow.update()
            update_image_window(fb)
        except Exception:
            print("GUI execution stops")


def update_image_window(fb: FrameBuffer):
    if debugWindow is None:
        return
    img = ImageTk.PhotoImage(fb.frame_buffer_image)
    debugWindowLabel.configure(image=img)
    debugWindowLabel.image = img


def draw_model_solid_interactive(buffer: FrameBuffer, model: Model, cam: Camera = None,
                                 color: RGB = RGB(np.uint8(255), np.uint8(200), np.uint8(125))):
    renderer_thread = threading.Thread(target=draw_model_solid_color, args=(buffer, model, cam, color,))
    renderer_thread.start()
    create_image_window(buffer)


def draw_model_shaded_interactive(buffer: FrameBuffer, model: Model, cam: Camera = None):
    renderer_thread = threading.Thread(target=draw_model_shaded, args=(buffer, model, cam,))
    renderer_thread.start()
    create_image_window(buffer)
