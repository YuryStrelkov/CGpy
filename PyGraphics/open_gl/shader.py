from OpenGL.GL.shaders import compileProgram, compileShader

import vmath.matrices
from vmath.matrices import Mat3, Mat4
from vmath.vectors import Vec2, Vec3
from OpenGL.GL import *
import re

vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texture;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_normal;
out vec2 v_texture;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_normal    = normalize((model * vec4(a_normal, 0.0)).xyz);
    v_texture   = a_texture;
}

"""
fragment_src = """
# version 330
in vec3 v_normal;
in vec2 v_texture;
out vec4 out_color;
void main()
{   
    float amount = 0.1 + (dot(-v_normal, vec3(0.333, 0.333,-0.333))) ;
    out_color = vec4(amount * v_texture.x, amount * v_texture.y, amount, 1);
    out_color =   vec4(amount, amount, amount, 1);
}
"""


class UniformTypes(GLenum):
    Matrix4 = 35676
    Matrix3 = -1
    Matrix2 = -1
    Vector4 = -1
    Vector3 = -1
    Vector2 = -1
    Float = -1
    Int = -1
    Texture = -1
    TextureCube = -1
    TextureArray = -1



class Shader(object):
    def __init__(self):
        self.__program_id = 0
        self.__vert_id = 0
        self.__frag_id = 0
        self.__shader_uniforms: dict = {}
        self.__shader_attribs: dict = {}

    def __str__(self):
        res = "Shader     :\n"
        res += f"program_id :{self.__program_id},\n"
        res += f"vert_id    :{self.__vert_id},\n"
        res += f"frag_id    :{self.__frag_id},\n"
        res += "Attribs   :\n"
        for key in self.__shader_attribs.keys():
            res += f"name: {key}, value: {self.__shader_attribs[key]}\n"
            # res += f"id : {self.__shader_attribs[key][0]}; name : {key}; type {id : {self.__shader_attribs[key][2]}}\n"
        res += "Uniforms   :\n"
        for key in self.__shader_uniforms.keys():
            res += f"name: {key}, value: {self.__shader_uniforms[key]}\n"
            # res += f"id : {self.__shader_uniforms[key][0]}; name : {key}; type {id : {self.__shader_uniforms[key][2]}}\n"
        return res

    def __del__(self):
        glDeleteProgram(self.__vert_id)
        glDeleteProgram(self.__frag_id)
        glDeleteProgram(self.__program_id)

    @staticmethod
    def default_shader():
        shader = Shader()
        shader.vert_shader(vertex_src, False)
        shader.frag_shader(fragment_src, False)
        return shader

    @property
    def attribytes(self):
        return self.__shader_attribs

    @property
    def uniforms(self):
        return self.__shader_uniforms

    @property
    def program_id(self):
        return self.__program_id

    def get_uniform_location(self, uniform_name: str):
        if uniform_name in self.__shader_uniforms:
            return self.__shader_uniforms[uniform_name][0]
        return -1

    def get_attrib_location(self, attrib_name: str):
        if attrib_name in self.__shader_attribs:
            return self.__shader_attribs[attrib_name][0]
        return -1

    @staticmethod
    def __read_all_code(code_src: str) -> str:
        code: str = ""
        with open(code_src, mode='r') as file:
            for str_ in file:
                line = (re.sub(r"[\t]*", "", str_))
                code += line
            if len(code) == 0:
                raise Exception("Vertex shader creation error::empty src-code...")
            return code

    @staticmethod
    def gl_get_active_attrib(program, index):
        buf_size = 256
        length = (ctypes.c_int * 1)()
        size = (ctypes.c_int * 1)()
        attrib_type = (ctypes.c_uint * 1)()
        attrib_name = ctypes.create_string_buffer(buf_size)
        # pyopengl has a bug, this is a patch
        glGetActiveAttrib(program, index, buf_size, length, size, attrib_type, attrib_name)
        attrib_name = attrib_name[:length[0]].decode('utf-8')
        return attrib_name, size[0], attrib_type[0]

    @staticmethod
    def gl_get_active_uniform(program, index):
        buf_size = 256
        length = (ctypes.c_int * 1)()
        size = (ctypes.c_int * 1)()
        attrib_type = (ctypes.c_uint * 1)()
        attrib_name = ctypes.create_string_buffer(buf_size)
        glGetActiveUniform(program, index, buf_size, length, size, attrib_type, attrib_name)
        attrib_name = attrib_name[:length[0]].decode('utf-8')
        return attrib_name, size[0], attrib_type[0]

    def __get_all_attrib_locations(self):
        count = glGetProgramiv(self.__program_id, GL_ACTIVE_ATTRIBUTES)
        print(f"Active Attributes: {count}\n", )
        if len(self.__shader_attribs) != 0:
            self.__shader_attribs.clear()
        for i in range(count):
            name_, size_, type_ = Shader.gl_get_active_attrib(self.__program_id, i)
            self.__shader_attribs[name_] = (i, size_, type_)

    def __get_all_uniform_locations(self):
        count = glGetProgramiv(self.__program_id, GL_ACTIVE_UNIFORMS)
        print(f"Active Uniforms: {count}\n")
        if len(self.__shader_uniforms) != 0:
            self.__shader_uniforms.clear()
        for i in range(count):
            name_, size_, type_ = Shader.gl_get_active_uniform(self.__program_id, i)

            self.__shader_uniforms[name_] = (i, size_, type_)

            if UniformTypes.Matrix4 == type_:
                self.send_mat_4(name_, vmath.matrices.identity_4())

    def frag_shader(self, code: str, from_file: bool = True):
        if from_file:
            self.__frag_id = compileShader(self.__read_all_code(code), GL_FRAGMENT_SHADER)
            if self.__frag_id == 0:
                raise Exception(f"{GL_FRAGMENT_SHADER} shader compilation error...")
            self.__compile()
            return
        self.__frag_id = compileShader(code, GL_FRAGMENT_SHADER)
        if self.__frag_id == 0:
            raise Exception(f"{GL_FRAGMENT_SHADER} shader compilation error...")
        self.__compile()

    def vert_shader(self, code: str, from_file: bool = True):
        if from_file:
            self.__vert_id = compileShader(self.__read_all_code(code), GL_VERTEX_SHADER)
            if self.__vert_id == 0:
                raise Exception(f"{GL_VERTEX_SHADER} shader compilation error...")
            self.__compile()
            return
        self.__vert_id = compileShader(code, GL_VERTEX_SHADER)
        if self.__vert_id == 0:
            raise Exception(f"{GL_VERTEX_SHADER} shader compilation error...")
        self.__compile()

    def __compile(self):
        if self.__vert_id == 0:
            return
        if self.__frag_id == 0:
            return
        if self.__program_id != 0:
            glDeleteProgram(self.__program_id)
        self.__program_id = compileProgram(self.__vert_id, self.__frag_id)
        if self.__program_id == 0:
            raise Exception("Shader program compilation error...")
        self.bind()
        self.__get_all_attrib_locations()
        self.__get_all_uniform_locations()

    def send_mat_3(self, mat_name: str, mat: Mat3, transpose=GL_FALSE):
        loc = self.get_uniform_location(mat_name)
        if loc == -1:
            return
        self.bind()
        data = mat.as_array
        glUniformMatrix3fv(loc, 1, transpose, (GLfloat * len(data))(*data))

    def send_mat_4(self, mat_name: str, mat: Mat4, transpose=GL_FALSE):
        loc = self.get_uniform_location(mat_name)
        if loc == -1:
            return
        self.bind()
        data = mat.as_array
        glUniformMatrix4fv(loc, 1, transpose, (GLfloat * len(data))(*data))

    def send_vec_2(self, vec_name: str, vec: Vec2):
        loc = self.get_uniform_location(vec_name)
        if loc == -1:
            return
        self.bind()
        data = vec.as_array
        glUniform2fv(loc, 1, (GLfloat * len(data))(*data))

    def send_vec_3(self, vec_name: str, vec: Vec3):
        loc = self.get_uniform_location(vec_name)
        if loc == -1:
            return
        self.bind()
        data = vec.as_array
        glUniform3fv(loc, 1, (GLfloat * len(data))(*data))

    def send_float(self, param_name: str, val: float):
        loc = self.get_uniform_location(param_name)
        if loc == -1:
            return
        self.bind()
        glUniform1f(loc, val)

    def send_int(self, param_name: str, val: int):
        loc = self.get_uniform_location(param_name)
        if loc == -1:
            return
        self.bind()
        glUniform1i(loc, val)

    def bind(self):
        glUseProgram(self.__program_id)

    def unbind(self):
        glUseProgram(0)
