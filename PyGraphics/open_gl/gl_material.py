from materials.material import Material
from open_gl.gl_texture import TextureGL
from open_gl.shader import Shader
from vmath.vectors import Vec3


class MaterialGL(object):
    def __init__(self):
        self.name = ""
        self.diffuse_color: Vec3 = Vec3(1, 1, 1)  # Kd: specifies diffuse color
        self.specular_color: Vec3 = Vec3(1, 1, 1)  # Ks: specifies specular color
        self.ns: float = 10  # defines the focus of specular highlights in the material.
        # Ns values normally range from 0 to 1000, with a high value resulting in a tight, concentrated highlight.
        self.ni: float = 1.5  # Ni: defines the optical density
        self.dissolve: float = 1.0  # d or Tr: specifies a factor for dissolve, how much this material dissolves into the background.
        # A factor of 1.0 is fully opaque. A factor of 0.0 is completely transparent.
        self.illum: float = 2.0  # illum: specifies an illumination model, using a numeric value
        self.__diffuse: TextureGL = None
        self.__specular: TextureGL = None
        self.__normals: TextureGL = None
        self.__shader: Shader = None