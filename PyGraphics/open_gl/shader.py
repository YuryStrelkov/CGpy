from OpenGL.GL.shaders import compileProgram, compileShader
from vmath.matrices import Mat3, Mat4
from vmath.vectors import Vec2, Vec3
from OpenGL.GL import *
from enum import Enum
import vmath.matrices
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
    // v_normal    = normalize((model * vec4(a_normal, 0.0)).xyz);
    v_normal    = normalize((vec4(a_normal, 0.0)).xyz);
    v_texture   = a_texture;
}

"""
fragment_src = """
# version 330
in vec3 v_normal;
in vec2 v_texture;
out vec4 out_color;

uniform vec3 diffuse_color;
uniform vec3 specular_color;

void main()
{   
    float amount =  (dot(-v_normal, vec3(0.333, 0.333, 0.333))) ;
    //  out_color = vec4(v_texture.x, v_texture.y, 1, 1);
    out_color = vec4(0.2 + diffuse_color * specular_color * amount, 1);
}
"""


class ShaderDataTypes(Enum):
    Matrix4: int = 35676
    Matrix3: int = 35675
    Vector3: int = 35665
    Vector2: int = 35664
    Float: int = 5126
    Int: int = 5124
    Texture: int = 35678
    TextureCube: int = 35680
    TextureArray: int = 36289


class Shader(object):

    __shader_instances = {}

    @staticmethod
    def shaders_enumerate():
        print(Shader.__shader_instances.items())
        for buffer in Shader.__shader_instances.items():
            yield buffer[1]

    @staticmethod
    def shaders_delete():
        # print(GPUBuffer.__buffer_instances)
        while len(Shader.__shader_instances) != 0:
            item = Shader.__shader_instances.popitem()
            item[1].delete_buffer()

    @staticmethod
    def default_shader():
        shader = Shader()
        shader.vert_shader(vertex_src, False)
        shader.frag_shader(fragment_src, False)
        shader.load_defaults_settings()
        return shader

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

    def __init__(self):
        self.name: str = "default"
        self.__program_id = 0
        self.__vert_id = 0
        self.__frag_id = 0
        self.__shader_uniforms: dict = {}
        self.__shader_attributes: dict = {}

    def __str__(self):
        res = f"Shader      : 0x{id(self)},\n"
        res += f"name        : {self.name},\n"
        res += f"program_id  : {self.__program_id:4},\n"
        res += f"vert_id     : {self.__vert_id:4},\n"
        res += f"frag_id     : {self.__frag_id:4},\n"
        res += "Attributes__________________________________________________________\n"
        res += f"|{'id':4}|{'name':30}|{'type':30}|\n"
        for key in self.__shader_attributes.keys():
            info = self.__shader_attributes[key]
            res += f"|{info[0]:4}|{key:30}|{ShaderDataTypes(info[2]):30}|\n"
      #  res += "____________________________________________________________________\n"

        res += "Uniforms___________________________________________________________|\n"
        res += f"|{'id':4}|{'name':30}|{'type':30}|\n"
        for key in self.__shader_uniforms.keys():
            info = self.__shader_uniforms[key]
            res += f"|{info[0]:4}|{key:30}|{ShaderDataTypes(info[2]):30}|\n"
        res += "____________________________________________________________________\n"
        return res

    def __del__(self):
        self.delete_shader()

    def delete_shader(self):
        glDeleteProgram(self.__vert_id)
        glDeleteProgram(self.__frag_id)
        glDeleteProgram(self.__program_id)
        if self.__program_id in Shader.__shader_instances:
            del Shader.__shader_instances[self.__program_id]
        self.__program_id = 0
        self.__vert_id = 0
        self.__frag_id = 0

    @property
    def attribytes(self):
        return self.__shader_attributes

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
        if attrib_name in self.__shader_attributes:
            return self.__shader_attributes[attrib_name][0]
        return -1

    def __get_all_attrib_locations(self):
        count = glGetProgramiv(self.__program_id, GL_ACTIVE_ATTRIBUTES)
        print(f"Active Attributes: {count}\n", )
        if len(self.__shader_attributes) != 0:
            self.__shader_attributes.clear()
        for i in range(count):
            name_, size_, type_ = Shader.gl_get_active_attrib(self.__program_id, i)
            self.__shader_attributes[name_] = (i, size_, type_)

    def __get_all_uniform_locations(self):
        count = glGetProgramiv(self.__program_id, GL_ACTIVE_UNIFORMS)
        print(f"Active Uniforms: {count}\n")
        if len(self.__shader_uniforms) != 0:
            self.__shader_uniforms.clear()
        for i in range(count):
            name_, size_, type_ = Shader.gl_get_active_uniform(self.__program_id, i)
            self.__shader_uniforms[name_] = (i, size_, type_)

    def load_defaults_settings(self):
        for name_ in self.__shader_uniforms:
            type_ = ShaderDataTypes(self.__shader_uniforms[name_][2])
            if ShaderDataTypes.Matrix4 == type_:
                self.send_mat_4(name_, vmath.matrices.identity_4())
                continue
            if ShaderDataTypes.Matrix3 == type_:
                self.send_mat_3(name_, vmath.matrices.identity_3())
                continue
            if ShaderDataTypes.Vector3 == type_:
                self.send_vec_3(name_, Vec3(1, 1, 1))
                continue
            if ShaderDataTypes.Vector2 == type_:
                self.send_vec_2(name_, Vec2(1, 1))
                continue
            if ShaderDataTypes.Float == type_:
                self.send_float(name_, 0.0)
                continue
            if ShaderDataTypes.Int == type_:
                self.send_int(name_, 0)
                continue

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
            self.delete_shader()
        self.__program_id = compileProgram(self.__vert_id, self.__frag_id)
        if self.__program_id == 0:
            raise Exception("Shader program compilation error...")
        Shader.__shader_instances[self.__program_id] = self
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
