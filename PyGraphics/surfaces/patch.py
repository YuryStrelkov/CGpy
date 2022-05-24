from vmath.mathUtils import Vec3
from models.trisMesh import TrisMesh


def quadratic_bezier_surface(p1: Vec3, p2: Vec3, p3: Vec3,
                             p4: Vec3, p5: Vec3, p6: Vec3,
                             p7: Vec3, p8: Vec3, p9: Vec3, u: float, v: float) -> Vec3:
    phi1: float = (1 - u) * (1 - u)
    phi3: float = u * u
    phi2: float = -2 * phi3 + 2 * u

    psi1: float = (1 - v) * (1 - v)
    psi3: float = v * v
    psi2: float = -2 * psi3 + 2 * v
    return p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + \
           p4 * phi2 * psi1 + p5 * phi2 * psi2 + p6 * phi2 * psi3 + \
           p7 * phi3 * psi1 + p8 * phi3 * psi2 + p9 * phi3 * psi3


def cubic_bezier_surface(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3,
                         p5: Vec3, p6: Vec3, p7: Vec3, p8: Vec3,
                         p9: Vec3, p10: Vec3, p11: Vec3, p12: Vec3,
                         p13: Vec3, p14: Vec3, p15: Vec3, p16: Vec3, u: float, v: float) -> Vec3:
    phi1: float = (1 - u) * (1 - u) * (1 - u)
    phi4: float = u * u * u
    phi2: float = 3 * phi4 - 6 * u * u + 3 * u
    phi3: float = -3 * phi4 + 3 * u * u

    psi1: float = (1 - v) * (1 - v) * (1 - v)
    psi4: float = v * v * v
    psi2: float = 3 * psi4 - 6 * v * v + 3 * v
    psi3: float = -3 * psi4 + 3 * v * v
    return p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + p4 * phi1 * psi4 + \
           p5 * phi2 * psi1 + p6 * phi2 * psi2 + p7 * phi2 * psi3 + p8 * phi2 * psi4 + \
           p9 * phi3 * psi1 + p10 * phi3 * psi2 + p11 * phi3 * psi3 + p12 * phi3 * psi4 + \
           p13 * phi4 * psi1 + p14 * phi4 * psi2 + p15 * phi4 * psi3 + p16 * phi4 * psi4


class CubicPatch(object):
    def __init__(self):
        self.__mesh: TrisMesh = []
        self.__controllers: [Vec3] = \
            [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0),
             Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0),
             Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0),
             Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)]

    @property
    def p1(self):
        return self.__controllers[0]

    @property
    def p2(self):
        return self.__controllers[1]

    @property
    def p3(self):
        return self.__controllers[2]

    @property
    def p4(self):
        return self.__controllers[3]

    @property
    def p5(self):
        return self.__controllers[4]

    @property
    def p6(self):
        return self.__controllers[5]

    @property
    def p7(self):
        return self.__controllers[6]

    @property
    def p8(self):
        return self.__controllers[7]

    @property
    def p9(self):
        return self.__controllers[8]

    @property
    def p10(self):
        return self.__controllers[9]

    @property
    def p11(self):
        return self.__controllers[10]

    @property
    def p12(self):
        return self.__controllers[11]

    @property
    def p13(self):
        return self.__controllers[12]

    @property
    def p14(self):
        return self.__controllers[13]

    @property
    def p15(self):
        return self.__controllers[14]

    @property
    def p16(self):
        return self.__controllers[15]
