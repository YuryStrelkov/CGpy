import glfw
from OpenGL.GL import *
from models import tris_mesh
from open_gl.gl_mesh import MeshGL
from open_gl.gpu_buffer import GPUBuffer


class Window(object):
    def __init__(self, w: int = 800, h: int = 600, name: str = "gl_window"):
        self.__window = None
        self.__width: int = w
        self.__height: int = h
        self.__nane: str = name
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
        self.__window = glfw.create_window(self.__width, self.__height, self.__nane, None, None)
        if not self.__window:
            glfw.terminate()
            raise Exception("glfw window creation error...")
        glfw.set_window_pos(self.__window, 400, 200)
        glfw.make_context_current(self.__window)

    def _on_begin_draw(self):
        glClearColor(125/255, 135/255, 145/255, 1)

    def _on_draw(self):
        glClear(GL_COLOR_BUFFER_BIT)
        for mesh in self.__meshes:
            mesh.draw()

    def main_loop(self):
        self._on_begin_draw()
        while not glfw.window_should_close(self.__window):
            glfw.poll_events()
            self._on_draw()
            glfw.swap_buffers(self.__window)


if __name__ == "__main__":

    w = Window()
    meshes = tris_mesh.read_obj_mesh("E:/GitHub/CGpy/PyGraphics/resources/box.obj")
    # print(meshes[0])
    gl_mesh = MeshGL()
    gl_mesh.mesh = meshes[0]
    w.register_mesh(gl_mesh)
    print(gl_mesh)
    w.main_loop()
