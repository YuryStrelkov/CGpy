import numpy as np

import materials.material
import models.trisMesh
from camera import Camera
from models.model import Model
from models.trisMesh import TrisMesh
from vmath.mathUtils import Vec2
from frameBuffer import RGB, FrameBuffer
import graphics as gr
from materials.material import Material
import time
from shapes.bezier2 import BezierCurve2


def bezier_intersection_test():
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
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
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    curve: BezierCurve2 = BezierCurve2()

    curve.closed = True
    n_pnts: int = 8
    for i in range(0, n_pnts):
        curve.add_point(Vec2(0.666 * np.cos(np.pi / n_pnts * 2 * i), 0.666 * np.sin(np.pi / n_pnts * 2 * i)))

    curve.set_flow()

    gr.draw_bezier(frame_buffer, curve)

    for i in range(0, curve.n_control_points):
        curve.move_point(i, curve.get_point(i) + curve.sect_normal(i, 0) * 0.124)
    gr.draw_bezier(frame_buffer, curve)
    frame_buffer.imshow()


def static_solid_color(render_camera: Camera = None, draw_wire: bool = False):
    start_time: float = time.time()
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    mesh = TrisMesh()
    mesh.read("resources/rabbit.obj")
    mesh_mat = Material()
    mesh_mat.set_diff("resources/checkerboard-rainbow_.jpg")
    mesh_mat.diffuse.tile = Vec2(5, 5)
    gr.draw_mesh_solid_color(frame_buffer, mesh, render_camera)
    if draw_wire:
        gr.draw_edges(frame_buffer, mesh, render_camera)
    print("static_solid_color :: elapsed render time : ", time.time() - start_time)
    frame_buffer.imshow()


def static_shading(render_camera: Camera = None, draw_wire: bool = False):
    start_time: float = time.time()
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    mesh = TrisMesh()
    mesh.read("resources/rabbit.obj")
    mesh_mat = Material()
    mesh_mat.set_diff("resources/checkerboard-rainbow_.jpg")
    mesh_mat.diffuse.tile = Vec2(5, 5)
    gr.draw_mesh_shaded(frame_buffer, mesh, mesh_mat, render_camera)
    if draw_wire:
        gr.draw_edges(frame_buffer, mesh, render_camera)
    print("static_shading :: elapsed render time : ", time.time() - start_time)
    frame_buffer.imshow()


def interactive_solid_color(render_camera: Camera = None):
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    mesh = TrisMesh()
    mesh.read("resources/rabbit.obj")
    mesh_mat = Material()
    mesh_mat.set_diff("resources/checkerboard-rainbow_.jpg")
    mesh_mat.diffuse.tile = Vec2(5, 5)
    gr.draw_mesh_solid_interactive(frame_buffer, mesh, render_camera)


def interactive_shading(render_camera: Camera = None):
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    mesh = TrisMesh()
    mesh.read("resources/rabbit.obj")
    mesh_mat = Material()
    mesh_mat.set_diff("resources/checkerboard-rainbow_.jpg")
    mesh_mat.diffuse.tile = Vec2(5, 5)
    gr.draw_mesh_shaded_interactive(frame_buffer, mesh, mesh_mat, render_camera)


start_val: int = 0


if __name__ == '__main__':

    model: Model = Model("resources/teapots.obj", "resources/teapots.mtl")
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    gr.draw_model_edges(frame_buffer, model)
    # meshes: [TrisMesh] = models.trisMesh.read_obj_mesh("resources/teapots.obj")
    # mats:   [Material] = materials.material.read_material("resources/teapots.mtl")
    # static_solid_color()
    # static_shading()
    # interactive_shading()
    frame_buffer.imshow()
    #bezier_curve_test()
    # bezier_intersection_test()
