import time

import vmath.matrices
from open_gl.gpu_buffer import GPUBuffer
from open_gl.gl_mesh import MeshGL
from open_gl.shader import Shader
from vmath.vectors import Vec3
from models import tris_mesh
import transforms.transform
from camera import Camera
from OpenGL.GL import *
import glfw


class Window(object):
    def __init__(self, w: int = 800, h: int = 800, name: str = "gl_window"):
        self.__window = None
        self.__width: int = w
        self.__height: int = h
        self.__name: str = name
        self.shader: Shader
        self.__meshes: [MeshGL] = []
        try:
            self.__init_window()
        except Exception as ex:
            print(f"GLWindow creating error\n{ex.args}")
            exit(-1)

    def register_mesh(self, mesh: MeshGL):
        self.__meshes.append(mesh)

    def __del__(self):
        GPUBuffer.gpu_buffers_delete()
        MeshGL.vao_delete_all()
        glfw.terminate()

    def __init_window(self):
        if not glfw.init():
            raise Exception("glfw initialize error...")
        self.__window = glfw.create_window(self.__width, self.__height, self.__name, None, None)
        if not self.__window:
            glfw.terminate()
            raise Exception("glfw window creation error...")
        glfw.set_window_pos(self.__window, 400, 200)
        glfw.make_context_current(self.__window)
        self.shader = Shader.default_shader()

    def _on_begin_draw(self):
        glClearColor(125/255, 135/255, 145/255, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _on_draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for mesh in self.__meshes:
            mesh.draw()

    def main_loop(self):
        self._on_begin_draw()
        # t = 0

        t = transforms.transform.Transform()
        t.scale = Vec3(0.2, 0.2, 0.2)
        t.z = -30
        # t.ax = 30
        angle = 0
        while not glfw.window_should_close(self.__window):
            angle += 0.01
            t.angles = Vec3(0, angle, 0)
            if angle >= 360:
                angle = 0
            t_ = time.time()
            glfw.poll_events()
            self.shader.send_mat_4("model", t.transform_matrix, GL_TRUE)
            self._on_draw()
            glfw.swap_buffers(self.__window)
            # if time.time() - t_ > 1e-6:
               # print(f"fps : {round(1 / (time.time() - t_))}")


if __name__ == "__main__":
    w = Window()
    meshes = tris_mesh.read_obj_mesh("E:/GitHub/CGpy/PyGraphics/resources/fox.obj")
    gl_mesh = MeshGL()
    gl_mesh.mesh = meshes[0]
    # print(gl_mesh)
    w.register_mesh(gl_mesh)
    cam = Camera()
    cam.transform.z = -1.8
    # 1.299363    0.000000    0.000000    0.000000
    # 0.000000    1.732051    0.000000    0.000000
    # 0.000000    0.000000  - 1.000200  - 1.000000
    # 0.000000    0.000000  - 0.200020    0.000000
    t = transforms.transform.Transform()
    t.scale = Vec3(0.2, 0.2, 0.2)
    t.origin = Vec3(0, 0.25, 1)
    t.look_at(Vec3(0, 0, 0), t.origin)
    t.up *= -1
    print(t)
    # w.shader.send_mat_4("model",      t.transform_matrix,          GL_TRUE)
    w.shader.send_mat_4("view",       t.transform_matrix, GL_TRUE)
    w.shader.send_mat_4("projection", cam.projection)
    w.main_loop()