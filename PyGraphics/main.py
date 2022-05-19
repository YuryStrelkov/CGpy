import numpy as np
from camera import Camera
from trisMeshData import TrisMeshData
from mathUtils import Vec2
from frameBuffer import RGB, FrameBuffer
import graphics as gr
from material import Material
import time
from bezier import BezierCurve2


def bezier_curve_test():
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    curve: BezierCurve2 = BezierCurve2()
    curve.add_point(Vec2(-0.25, -0.5))
    curve.add_point(Vec2(-0.125, 0.95))
    curve.add_point(Vec2(0.333, 0.5))
    curve.add_point(Vec2(0.5, -0.333))
    curve.add_point(Vec2(-0.45, 0.333))
    curve.add_point(Vec2(0.333, 0.333))
    gr.draw_bezier(frame_buffer, curve)
    curve.rem_point(3)
    for i in range(0, len(curve.points)):
        print(i)
        curve.move_point(i, curve.get_point(i) + curve.sect_normal(i, 0) * 0.124)
    gr.draw_bezier(frame_buffer, curve)
    frame_buffer.imshow()


def static_solid_color(render_camera: Camera = None, draw_wire: bool = False):
    start_time: float = time.time()
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    mesh = TrisMeshData()
    mesh.read("rabbit.obj")
    mesh_mat = Material()
    mesh_mat.set_diff("checkerboard-rainbow_.jpg")
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
    mesh = TrisMeshData()
    mesh.read("rabbit.obj")
    mesh_mat = Material()
    mesh_mat.set_diff("checkerboard-rainbow_.jpg")
    mesh_mat.diffuse.tile = Vec2(5, 5)
    gr.draw_mesh_shaded(frame_buffer, mesh, mesh_mat, render_camera)
    if draw_wire:
        gr.draw_edges(frame_buffer, mesh, render_camera)
    print("static_shading :: elapsed render time : ", time.time() - start_time)
    frame_buffer.imshow()


def interactive_solid_color(render_camera: Camera = None):
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    mesh = TrisMeshData()
    mesh.read("rabbit.obj")
    mesh_mat = Material()
    mesh_mat.set_diff("checkerboard-rainbow_.jpg")
    mesh_mat.diffuse.tile = Vec2(5, 5)
    gr.draw_mesh_solid_interactive(frame_buffer, mesh, render_camera)


def interactive_shading(render_camera: Camera = None):
    frame_buffer = FrameBuffer(1000, 1000)
    frame_buffer.clear_color(RGB(np.uint8(200), np.uint8(200), np.uint8(200)))
    mesh = TrisMeshData()
    mesh.read("rabbit.obj")
    mesh_mat = Material()
    mesh_mat.set_diff("checkerboard-rainbow_.jpg")
    mesh_mat.diffuse.tile = Vec2(5, 5)
    gr.draw_mesh_shaded_interactive(frame_buffer, mesh, mesh_mat, render_camera)


if __name__ == '__main__':
    # static_solid_color()
    # static_shading()
    bezier_curve_test()
