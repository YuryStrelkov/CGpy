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

if __name__ == '__main__':
    t = Transform()
    t.up = Vec3(1, 1, 0).normalized()
    #t.angles = Vec3(math.pi/6, math.pi/4, math.pi/3)
    print(t.angles * 180 / math.pi)
    print(t)
    cam = Camera()
    print(cam)
    # bezier_intersection_test()
    # bezier_curve_test()
    interactive_shading()
    #interactive_solid_color()
    #static_solid_color()
    #static_shading()
    # interactive_solid_color()
