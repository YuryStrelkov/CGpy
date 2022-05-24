class Mat3(object):
    def __init__(self,
                 m0: float = 0, m1: float = 0, m2: float = 0,
                 m3: float = 0, m4: float = 0, m5: float = 0,
                 m6: float = 0, m7: float = 0, m8: float = 0):
        self.data: [float] = [m0, m1, m2, m3, m4, m5, m6, m7, m8]

    def __getitem__(self, key: int) -> float: return self.data[key]

    def __setitem__(self, key: int, value: float): self.data[key] = value

    def __repr__(self):
        res: str = "mat4:\n"
        res += "[[%s, %s, %s],\n" % (self.data[0], self.data[1], self.data[2])
        res += " [%s, %s, %s],\n" % (self.data[3], self.data[4], self.data[5])
        res += " [%s, %s, %s],\n" % (self.data[6], self.data[7], self.data[8])
        return res

    def __str__(self):
        res: str = ""
        res += "[[%s, %s, %s],\n" % (self.data[0], self.data[1], self.data[2])
        res += " [%s, %s, %s],\n" % (self.data[3], self.data[4], self.data[5])
        res += " [%s, %s, %s],\n" % (self.data[6], self.data[7], self.data[8])
        return res

    # row 1 set/get
    @property
    def m00(self) -> float: return self.data[0]

    @m00.setter
    def m00(self, val: float): self.data[0] = val

    @property
    def m01(self) -> float: return self.data[1]

    @m01.setter
    def m01(self, val: float): self.data[1] = val

    @property
    def m02(self) -> float: return self.data[2]

    @m02.setter
    def m02(self, val: float): self.data[2] = val

    # row 2 set/get
    @property
    def m10(self) -> float: return self.data[3]

    @m10.setter
    def m10(self, val: float): self.data[3] = val

    @property
    def m11(self) -> float: return self.data[4]

    @m11.setter
    def m11(self, val: float): self.data[4] = val

    @property
    def m12(self) -> float: return self.data[5]

    @m12.setter
    def m12(self, val: float): self.data[5] = val

    # row 3 set/get
    @property
    def m20(self) -> float: return self.data[6]

    @m20.setter
    def m20(self, val: float): self.data[6] = val

    @property
    def m21(self) -> float: return self.data[7]

    @m21.setter
    def m21(self, val: float): self.data[7] = val

    @property
    def m22(self) -> float: return self.data[8]

    @m22.setter
    def m22(self, val: float): self.data[8] = val


def mat_mul_3(a: Mat3, b: Mat3) -> Mat3:
    return Mat3(
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],

        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],

        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
    )


class Mat4(object):

    def __init__(self,
                 m0: float = 0, m1: float = 0, m2: float = 0, m3: float = 0,
                 m4: float = 0, m5: float = 0, m6: float = 0, m7: float = 0,
                 m8: float = 0, m9: float = 0, m10: float = 0, m11: float = 0,
                 m12: float = 0, m13: float = 0, m14: float = 0, m15: float = 0):
        self.data: [float] = [m0, m1, m2, m3,
                              m4, m5, m6, m7,
                              m8, m9, m10, m11,
                              m12, m13, m14, m15]

    def __getitem__(self, key: int) -> float: return self.data[key]

    def __setitem__(self, key: int, value: float): self.data[key] = value

    def __repr__(self):
        res: str = "mat4:\n"
        res += "[[%s, %s, %s, %s],\n" % (self.data[0], self.data[1], self.data[2], self.data[3])
        res += " [%s, %s, %s, %s],\n" % (self.data[4], self.data[5], self.data[6], self.data[7])
        res += " [%s, %s, %s, %s],\n" % (self.data[8], self.data[9], self.data[10], self.data[11])
        res += " [%s, %s, %s, %s]]" % (self.data[12], self.data[13], self.data[14], self.data[15])
        return res

    def __str__(self):
        res: str = ""
        res += "[[%s, %s, %s, %s],\n" % (self.data[0], self.data[1], self.data[2], self.data[3])
        res += " [%s, %s, %s, %s],\n" % (self.data[4], self.data[5], self.data[6], self.data[7])
        res += " [%s, %s, %s, %s],\n" % (self.data[8], self.data[9], self.data[10], self.data[11])
        res += " [%s, %s, %s, %s]]" % (self.data[12], self.data[13], self.data[14], self.data[15])
        return res

    # row 1 set/get
    @property
    def m00(self) -> float: return self.data[0]

    @m00.setter
    def m00(self, val: float): self.data[0] = val

    @property
    def m01(self) -> float: return self.data[1]

    @m01.setter
    def m01(self, val: float): self.data[1] = val

    @property
    def m02(self) -> float: return self.data[2]

    @m02.setter
    def m02(self, val: float): self.data[2] = val

    @property
    def m03(self) -> float: return self.data[3]

    @m03.setter
    def m03(self, val: float): self.data[3] = val

    # row 2 set/get
    @property
    def m10(self) -> float: return self.data[4]

    @m10.setter
    def m10(self, val: float): self.data[4] = val

    @property
    def m11(self) -> float: return self.data[5]

    @m11.setter
    def m11(self, val: float): self.data[5] = val

    @property
    def m12(self) -> float: return self.data[6]

    @m12.setter
    def m12(self, val: float): self.data[6] = val

    @property
    def m13(self) -> float: return self.data[7]

    @m13.setter
    def m13(self, val: float): self.data[7] = val

    # row 3 set/get
    @property
    def m20(self) -> float: return self.data[8]

    @m20.setter
    def m20(self, val: float): self.data[8] = val

    @property
    def m21(self) -> float: return self.data[9]

    @m21.setter
    def m21(self, val: float): self.data[9] = val

    @property
    def m22(self) -> float: return self.data[10]

    @m22.setter
    def m22(self, val: float): self.data[10] = val

    @property
    def m23(self) -> float: return self.data[11]

    @m23.setter
    def m23(self, val: float): self.data[11] = val

    # row 4 set/get
    @property
    def m30(self) -> float: return self.data[12]

    @m30.setter
    def m30(self, val: float): self.data[12] = val

    @property
    def m31(self) -> float: return self.data[13]

    @m31.setter
    def m31(self, val: float): self.data[13] = val

    @property
    def m32(self) -> float: return self.data[14]

    @m32.setter
    def m32(self, val: float): self.data[14] = val

    @property
    def m33(self) -> float: return self.data[15]

    @m33.setter
    def m33(self, val: float): self.data[15] = val


def mat_mul_4(a: Mat4, b: Mat4) -> Mat4:
    return Mat4(a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12],
                a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13],
                a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14],
                a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15],

                a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12],
                a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13],
                a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14],
                a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15],

                a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12],
                a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13],
                a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14],
                a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15],

                a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12],
                a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13],
                a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14],
                a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15])