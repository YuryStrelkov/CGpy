from OpenGL.GL.shaders import compileProgram, compileShader
import re

from OpenGL.raw.GL.VERSION.GL_2_0 import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, glUseProgram


class Shader(object):
    def __init__(self):
        self.__program_id = -1
        self.__vert_id = -1
        self.__frag_id = -1

    @staticmethod
    def __read_all_code(code_src: str) -> str:
        code: str = ""
        with open(code_src, mode='r') as file:
            for str_ in file:
                line = (re.sub(r"[\t]*", "", str_))
                code += line
            return code

    def __compile_vert(self, vs_src: str):
        code = self.__read_all_code(vs_src)
        if len(code) == 0:
            raise Exception("Vertex shader creation error::empty src-code...")
        self.__vert_id = compileShader(code, GL_VERTEX_SHADER)
        if self.__vert_id == -1:
            raise Exception("Vertex shader compilation error...")

    def __compile_frag(self, fs_src: str):
        code = self.__read_all_code(fs_src)
        if len(code) == 0:
            raise Exception("Vertex shader creation error::empty src-code...")
        self.__frag_id = compileShader(code, GL_FRAGMENT_SHADER)
        if self.__frag_id == -1:
            raise Exception("Vertex shader compilation error...")

    def __compile(self, vs_src: str, fs_src: str, geo_src: str = None):
        self.__program_id = compileProgram(self.__compile_vert(vs_src), self.__compile_frag(fs_src))
        if self.__program_id == -1:
            raise Exception("Shader program compilation error...")

    def bind(self):
        if self.__program_id != -1:
            glUseProgram(self.__program_id)

    def unbind(self):
        glUseProgram(-1)
