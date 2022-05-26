from typing import IO
import re
import numpy as np
from vmath.mathUtils import Vec2
from materials.rgb import RGB
from materials.texture import Texture
from vmath.vectors import Vec3


class Material(object):
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
        self.__diffuse: Texture = None
        self.__specular: Texture = None
        self.__normals: Texture = None

    def __repr__(self):
        res: str = f"<Material {self.name}\n"
        res += f"diff_color : {self.diffuse_color}\n"
        res += f"spec_color : {self.specular_color}\n"
        res += f"ns         : {self.ns}\n"
        res += f"ni         : {self.ni}\n"
        res += f"dissolve   : {self.dissolve}\n"
        res += f"illum      : {self.illum}\n"
        res += f"diff_tex   : None\n"

        if self.__diffuse is None:
            res += f"diff_tex   : None\n"
        else:
            res += f"diff_tex   :\n{self.__diffuse}\n"

        if self.__specular is None:
            res += f"spec_tex   : None\n"
        else:
            res += f"spec_tex   :\n{self.__specular}\n"

        if self.__normals is None:
            res += f"norm_tex   : None\n"
        else:
            res += f"norm_tex   :\n{self.__normals}\n"
        res += ">\n"
        return res

    def __str__(self):
        res: str = f"Material {self.name}\n"
        res += f"diff_color : {self.diffuse_color}\n"
        res += f"spec_color : {self.specular_color}\n"
        res += f"ns         : {self.ns}\n"
        res += f"ni         : {self.ni}\n"
        res += f"dissolve   : {self.dissolve}\n"
        res += f"illum      : {self.illum}\n"
        res += f"diff_tex   : None\n"

        if self.__diffuse is None:
            res += f"diff_tex   : None\n"
        else:
            res += f"diff_tex   :\n{self.__diffuse}\n"

        if self.__specular is None:
            res += f"spec_tex   : None\n"
        else:
            res += f"spec_tex   :\n{self.__specular}\n"

        if self.__normals is None:
            res += f"norm_tex   : None\n"
        else:
            res += f"norm_tex   :\n{self.__normals}\n"
        res += "\n"
        return res

    def set_diff(self, orig: str):
        if self.__diffuse is None:
            self.__diffuse = Texture()
        self.__diffuse.load(orig)

    def set_norm(self, orig: str):
        if self.__normals is None:
            self.__normals = Texture()
        self.__normals.load(orig)

    def set_spec(self, orig: str):
        if self.__specular is None:
            self.__specular = Texture()
        self.__specular.load(orig)

    def diff_color(self, uv: Vec2) -> RGB:
        if self.__diffuse is None:
            return RGB(np.uint8(255), np.uint8(255), np.uint8(255))
        return self.__diffuse.get_color_uv(uv)

    def norm_color(self, uv: Vec2) -> RGB:
        if self.__normals is None:
            return RGB(np.uint8(255), np.uint8(255), np.uint8(255))
        return self.__normals.get_color_uv(uv)

    def spec_color(self, uv: Vec2) -> RGB:
        if self.__specular is None:
            return RGB(np.uint8(255), np.uint8(255), np.uint8(255))
        return self.__specular.get_color_uv(uv)


def read_material(path: str) -> [Material]:
    file: IO

    try:
        file = open(path)
    except:
        print("file \"%s\" not found" % path)
        return []

    tmp: [str]
    tmp2: [str]
    lines: [str] = []
    id_: int

    for str_ in file:
        lines.append(re.sub(r"[\n\t]*", "", str_))
    file.close()

    if len(lines) == 0:
        print("file \"%s\" empty" % path)
        return []

    materials: [Material] = []

    for i in range(len(lines)):
        if len(lines[i]) == 0:
            continue

        tmp = lines[i].split(" ")

        id_ = len(tmp) - 1

        if id_ == -1:
            continue

        if tmp[0] == "#":
            continue

        if tmp[0] == "newmtl":
            mat: Material = Material()
            mat.name = tmp[1]
            materials.append(mat)
            continue

        if tmp[0] == "Kd":
            materials[len(materials) - 1].diffuse_color = Vec3(float(tmp[id_ - 2]), float(tmp[id_ - 1]),
                                                               float(tmp[id_]))
            continue

        if tmp[0] == "Ks":
            materials[len(materials) - 1].specular_color = Vec3(float(tmp[id_ - 2]), float(tmp[id_ - 1]),
                                                                float(tmp[id_]))
            continue

        if tmp[0] == "illum":
            materials[len(materials) - 1].illum = float(tmp[id_])
            continue

        if tmp[0] == "dissolve" or tmp[0] == "Tr":
            materials[len(materials) - 1].illum = float(tmp[id_])
            continue

        if tmp[0] == "Ns":
            materials[len(materials) - 1].ns = float(tmp[id_])
            continue

        if tmp[0] == "Ni":
            materials[len(materials) - 1].ni = float(tmp[id_])
            continue

        if tmp[0] == "map_Kd":
            materials[len(materials) - 1].set_diff(tmp[id_])
            continue

        if tmp[0] == "map_bump" or tmp[0] == "bump":
            materials[len(materials) - 1].set_norm(tmp[id_])
            continue

        if tmp[0] == "map_Ks":
            materials[len(materials) - 1].set_spec(tmp[id_])
            continue

    return materials
