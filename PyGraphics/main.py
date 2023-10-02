from PIL import Image

from transforms.transform import Transform
from shapes.bezier2 import BezierCurve2
from surfaces.patch import CubicPatch
from frameBuffer import FrameBuffer
from vmath.math_utils import Vec2
from vmath.vectors import Vec3
from models.model import Model
from camera import Camera
import graphics as gr
import numpy as np
import math
import time


def bezier_intersection_test():
    frame_buffer = FrameBuffer(1000, 1000)
    curve_1: BezierCurve2 = BezierCurve2()
    curve_1.add_point(Vec2(-0.333, 0.25))
    curve_1.add_point(Vec2(0.333, 0.25))
    curve_1.align_anchors(0, Vec2(0, -1), 10)
    curve_1.align_anchors(1, Vec2(0, -1), 10)

    curve_2: BezierCurve2 = BezierCurve2()
    curve_2.add_point(Vec2(-0.333, -0.25))
    curve_2.add_point(Vec2(0.333, -0.25))
    curve_2.align_anchors(0, Vec2(0, 1), 10)
    curve_2.align_anchors(1, Vec2(0, 1), 10)
    gr.draw_bezier(frame_buffer, curve_1)
    gr.draw_bezier(frame_buffer, curve_2)
    frame_buffer.imshow()


def bezier_curve_test():
    frame_buffer = FrameBuffer(1000, 1000)
    curve: BezierCurve2 = BezierCurve2()

    curve.closed = True
    n_pnts: int = 3
    for i in range(0, n_pnts):
        curve.add_point(Vec2(0.666 * np.cos(np.pi / n_pnts * 2 * i), 0.666 * np.sin(np.pi / n_pnts * 2 * i)))

    curve.set_flow()

    gr.draw_bezier(frame_buffer, curve)

    for i in range(0, curve.n_control_points):
        curve.move_point(i, curve.get_point(i) + curve.curve_normal(i, 0) * 0.124)
    gr.draw_bezier(frame_buffer, curve)
    frame_buffer.imshow()


def bezier_patch_test():
    frame_buffer = FrameBuffer(1000, 1000)
    start_time: float = time.time()
    patch: CubicPatch = CubicPatch()
    print("patch :: patch creation time : ", time.time() - start_time)
    gr.draw_patch_solid_color(frame_buffer, patch)
    gr.draw_patch_edges(frame_buffer, patch)
    frame_buffer.imshow()


def static_solid_color(render_camera: Camera = None, draw_wire: bool = False):
    start_time: float = time.time()
    frame_buffer = FrameBuffer(1000, 1000)
    print("static_solid_color :: frame buffer creation time : ", time.time() - start_time)
    start_time = time.time()
    model: Model = Model("resources/rabbit.obj", "resources/teapots.mtl")
    print("static_solid_color :: model loading time : ", time.time() - start_time)
    start_time = time.time()
    gr.draw_model_solid_color(frame_buffer, model, render_camera)
    print("static_solid_color :: elapsed render time : ", time.time() - start_time)
    if draw_wire:
        gr.draw_model_edges(frame_buffer, model, render_camera)
    frame_buffer.imshow()


def static_shading(render_camera: Camera = None, draw_wire: bool = False):
    frame_buffer = FrameBuffer(500, 500)
    start_time: float = time.time()
    model: Model = Model("resources/rabbit.obj", "resources/teapots.mtl")
    model.get_material(0).diffuse.tile = Vec2(5, 5)
    model.get_material(1).diffuse.tile = Vec2(15, 15)
    print("static_shading :: model loading time : ", time.time() - start_time)
    start_time = time.time()
    gr.draw_model_shaded(frame_buffer, model, render_camera)
    print("static_shading :: elapsed render time : ", time.time() - start_time)
    if draw_wire:
        gr.draw_model_edges(frame_buffer, model, render_camera)
    frame_buffer.imshow()


def interactive_solid_color(render_camera: Camera = None):
    frame_buffer = FrameBuffer(1000, 1000)

    model: Model = Model("resources/teapots.obj", "resources/teapots.mtl")

    gr.draw_model_solid_interactive(frame_buffer, model, render_camera)


def interactive_shading(render_camera: Camera = None):
    frame_buffer = FrameBuffer(1000, 1000)

    model: Model = Model("resources/rabbit.obj", "resources/teapots.mtl")

    model.get_material(0).diffuse.tile = Vec2(5, 5)

    model.get_material(1).diffuse.tile = Vec2(15, 15)

    model.get_material(2).diffuse.tile = Vec2(25, 25)

    gr.draw_model_shaded_interactive(frame_buffer, model, render_camera)


def mapping():
    im = Image.open("terrain_map_1.png")
    pixels = im.load()
    colors = []
    for index in range(im.size[0] * im.size[1]):
        row, col = divmod(index, im.size[1])
        if pixels[row, col] == (255, 255, 255, 255):
            colors.append(0)
            continue
        colors.append(3)
    print(im.size[0] / 16)
    for index, color in enumerate(colors):
        row, col = divmod(index, im.size[1])
        if row % 16 != 0:
            continue
        if col % 16 != 0:
            continue
        if col == 0:
            print()
        print(color, end=", ")



if __name__ == '__main__':
    # mapping()
    # exit()
    im = Image.open("rock.png")
    tile_size = 8
    n_tile_x = im.size[0]//tile_size
    n_tile_y = im.size[1]//tile_size
    pixels = im.load()
    print(n_tile_x, n_tile_y)
    # print(pixels[0, 0])

    maps = []
    RED = (255, 0, 0, 255)
    for index in range(n_tile_x * n_tile_y):
        row, col = divmod(index, n_tile_x)
        # tile_map = ["1" if pixels[0  + col * 16,  0 + row * 16][-1] == 255 else "0",
        #             "1" if pixels[15 + col * 16,  0 + row * 16][-1] == 255 else "0",
        #             "1" if pixels[0  + col * 16, 15 + row * 16][-1] == 255 else "0",
        #             "1" if pixels[15 + col * 16, 15 + row * 16][-1] == 255 else "0"]
        a =  1 if pixels[                col * tile_size,                 row * tile_size][-1] == 255 else 0
        b =  2 if pixels[tile_size - 1 + col * tile_size,                 row * tile_size][-1] == 255 else 0
        c =  4 if pixels[                col * tile_size, tile_size - 1 + row * tile_size][-1] == 255 else 0
        d =  8 if pixels[tile_size - 1 + col * tile_size, tile_size - 1 + row * tile_size][-1] == 255 else 0
        tile_map = a | b | c | d
        if tile_map == 0:
            continue

        u0 = col * tile_size / im.size[0]
        v0 = row * tile_size / im.size[1]

        u1 = (tile_size + col * tile_size) / im.size[0]
        v1 = (tile_size + row * tile_size) / im.size[1]

        maps.append((index, tile_map, u0, v0, u1, v1))
    with open("tiles.json", "wt") as out:
        for (index, tile_map, u0, v0, u1, v1) in maps:
            print(f"{{ \"tile_id\": {index:3}, \"tile_mask\": {tile_map:3}, \"u0\": {u0:>.5f}, \"v0\": {v0:>.5f}, \"u1\": {u1:>.5f}, \"v1\": {v1:>.5f}}},", file=out)
        # for row in range(n_tile_y):
        #     print(",". join(f"{{\"tile_id\" : {str(i + row * 9):>3}, \"tile_mask\": {str(maps[i + row * 9]):>3}}}"for i in range(9)), file=out)

    t0 = time.perf_counter()
    v = [i for i in range(1000000)]
    c = sum(v)
    print(c)
    print(time.perf_counter() - t0)

    t0 = time.perf_counter()
    c = sum((i for i in range(1000000)))
    print(c)
    print(time.perf_counter() - t0)

    #t.up = Vec3(1, 1, 0).normalized()
    ##t.angles = Vec3(math.pi/6, math.pi/4, math.pi/3)
    #print(t.angles * 180 / math.pi)
    #print(t)f
    #cam = Camera()
    # print(cam)
    # bezier_intersection_test()
    # bezier_curve_test()
    # interactive_shading()
    #interactive_solid_color()
    #static_solid_color()
    #static_shading()
    # interactive_solid_color()
