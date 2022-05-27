class Mat3(object):
    @staticmethod
    def __unpack_values(*args) -> [float]:
        args = args[0]

        number_of_args = len(args)

        if number_of_args == 0:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0]  # no arguments

        elif number_of_args == 9:
            return [args[0], args[1], args[2],
                    args[3], args[4], args[5],
                    args[6], args[7], args[8]]  # x, y and z passed in

        elif number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is float or arg_type is int:  # single int or float argument
                return [args[0],     0,      0,
                             0, args[0],     0,
                             0,      0, args[0]]

            if arg_type is Mat3:
                return [args[0].m00, args[0].m01, args[0].m02,
                        args[0].m10, args[0].m11, args[0].m12,
                        args[0].m20, args[0].m21, args[0].m22]

        raise TypeError(f'Invalid Input: {args}')

    def __init__(self,
                 m0: float = 0, m1: float = 0, m2: float = 0,
                 m3: float = 0, m4: float = 0, m5: float = 0,
                 m6: float = 0, m7: float = 0, m8: float = 0):
        self.__data: [float] = [m0, m1, m2, m3, m4, m5, m6, m7, m8]

    def __eq__(self, other) -> bool:
        if not(type(other) is Mat3):
            return False
        for i in range(0, 9):
            if not (self.__data[i] == other.__data[i]):
                return False
        return True

    def __hash__(self) -> int:
        return hash(self.__data)

    def __getitem__(self, key: int) -> float:
        if key < 0 or key >= 2:
            raise IndexError(f"Mat3 :: trying to access index: {key}")
        return self.__data[key]

    def __setitem__(self, key: int, value: float): self.__data[key] = value

    def __repr__(self) -> str:
        res: str = "mat4:\n"
        res += "[[%s, %s, %s],\n" % (self.__data[0], self.__data[1], self.__data[2])
        res += " [%s, %s, %s],\n" % (self.__data[3], self.__data[4], self.__data[5])
        res += " [%s, %s, %s]]\n" % (self.__data[6], self.__data[7], self.__data[8])
        return res

    def __str__(self) -> str:
        res: str = ""
        res += "[[%s, %s, %s],\n" % (self.__data[0], self.__data[1], self.__data[2])
        res += " [%s, %s, %s],\n" % (self.__data[3], self.__data[4], self.__data[5])
        res += " [%s, %s, %s]]\n" % (self.__data[6], self.__data[7], self.__data[8])
        return res

    def __add__(self, *args):
        other = self.__unpack_values(args)
        return Mat3(self.__data[0] + other[0],
                    self.__data[1] + other[1],
                    self.__data[2] + other[2],
                    self.__data[3] + other[3],
                    self.__data[4] + other[4],
                    self.__data[5] + other[5],
                    self.__data[6] + other[6],
                    self.__data[7] + other[7],
                    self.__data[7] + other[7])

    def __sub__(self, *args):
        other = self.__unpack_values(args)
        return Mat3(self.__data[0] - other[0],
                    self.__data[1] - other[1],
                    self.__data[2] - other[2],
                    self.__data[3] - other[3],
                    self.__data[4] - other[4],
                    self.__data[5] - other[5],
                    self.__data[6] - other[6],
                    self.__data[7] - other[7],
                    self.__data[7] - other[7])

    def __mul__(self, *args):
        b = self.__unpack_values(args)
        return Mat3(self.__data[0] * b[0] + self.__data[1] * b[3] + self.__data[2] * b[6],
                    self.__data[0] * b[1] + self.__data[1] * b[4] + self.__data[2] * b[7],
                    self.__data[0] * b[2] + self.__data[1] * b[5] + self.__data[2] * b[8],

                    self.__data[3] * b[0] + self.__data[4] * b[3] + self.__data[5] * b[6],
                    self.__data[3] * b[1] + self.__data[4] * b[4] + self.__data[5] * b[7],
                    self.__data[3] * b[2] + self.__data[4] * b[5] + self.__data[5] * b[8],

                    self.__data[6] * b[0] + self.__data[7] * b[3] + self.__data[8] * b[6],
                    self.__data[6] * b[1] + self.__data[7] * b[4] + self.__data[8] * b[7],
                    self.__data[6] * b[2] + self.__data[7] * b[5] + self.__data[8] * b[8])

    # row 1 set/get
    @property
    def m00(self) -> float: return self.__data[0]

    @m00.setter
    def m00(self, val: float): self.__data[0] = val

    @property
    def m01(self) -> float: return self.__data[1]

    @m01.setter
    def m01(self, val: float): self.__data[1] = val

    @property
    def m02(self) -> float: return self.__data[2]

    @m02.setter
    def m02(self, val: float): self.__data[2] = val

    # row 2 set/get
    @property
    def m10(self) -> float: return self.__data[3]

    @m10.setter
    def m10(self, val: float): self.__data[3] = val

    @property
    def m11(self) -> float: return self.__data[4]

    @m11.setter
    def m11(self, val: float): self.__data[4] = val

    @property
    def m12(self) -> float: return self.__data[5]

    @m12.setter
    def m12(self, val: float): self.__data[5] = val

    # row 3 set/get
    @property
    def m20(self) -> float: return self.__data[6]

    @m20.setter
    def m20(self, val: float): self.__data[6] = val

    @property
    def m21(self) -> float: return self.__data[7]

    @m21.setter
    def m21(self, val: float): self.__data[7] = val

    @property
    def m22(self) -> float: return self.__data[8]

    @m22.setter
    def m22(self, val: float): self.__data[8] = val


class Mat4(object):
    @staticmethod
    def __unpack_values(*args) -> [float]:
        args = args[0]

        number_of_args = len(args)

        if number_of_args == 0:
            return [0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0]  # no arguments

        elif number_of_args == 9:
            return [args[0],  args[1],  args[2],  args[3],
                    args[4],  args[5],  args[6],  args[7],
                    args[8],  args[9],  args[10], args[11],
                    args[12], args[13], args[14], args[15]]  # x, y and z passed in

        elif number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is float or arg_type is int:  # single int or float argument
                return [args[0],     0,      0,      0,
                             0, args[0],     0,      0,
                             0,      0, args[0],     0,
                             0,      0,      0, args[0]]

            if arg_type is Mat4:
                return [args[0].m00, args[0].m01, args[0].m02, args[0].m03,
                        args[0].m10, args[0].m11, args[0].m12, args[0].m13,
                        args[0].m20, args[0].m21, args[0].m22, args[0].m23,
                        args[0].m30, args[0].m31, args[0].m32, args[0].m33]

        raise TypeError(f'Invalid Input: {args}')

    def __init__(self,
                 m0: float = 0, m1: float = 0, m2: float = 0, m3: float = 0,
                 m4: float = 0, m5: float = 0, m6: float = 0, m7: float = 0,
                 m8: float = 0, m9: float = 0, m10: float = 0, m11: float = 0,
                 m12: float = 0, m13: float = 0, m14: float = 0, m15: float = 0):
        self.__data: [float] = [m0, m1, m2, m3,
                                m4, m5, m6, m7,
                                m8, m9, m10, m11,
                                m12, m13, m14, m15]

    def __eq__(self, other) -> bool:
        if not(type(other) is Mat4):
            return False
        for i in range(0, 16):
            if not (self.__data[i] == other.__data[i]):
                return False
        return True

    def __hash__(self) -> int:
        return hash(self.__data)

    def __getitem__(self, key: int) -> float: return self.__data[key]

    def __setitem__(self, key: int, value: float): self.__data[key] = value

    def __repr__(self) -> str:
        res: str = "mat4:\n"
        res += "[[%s, %s, %s, %s],\n" % (self.__data[0], self.__data[1], self.__data[2], self.__data[3])
        res += " [%s, %s, %s, %s],\n" % (self.__data[4], self.__data[5], self.__data[6], self.__data[7])
        res += " [%s, %s, %s, %s],\n" % (self.__data[8], self.__data[9], self.__data[10], self.__data[11])
        res += " [%s, %s, %s, %s]]" % (self.__data[12], self.__data[13], self.__data[14], self.__data[15])
        return res

    def __str__(self) -> str:
        res: str = ""
        res += "[[%s, %s, %s, %s],\n" % (self.__data[0], self.__data[1], self.__data[2], self.__data[3])
        res += " [%s, %s, %s, %s],\n" % (self.__data[4], self.__data[5], self.__data[6], self.__data[7])
        res += " [%s, %s, %s, %s],\n" % (self.__data[8], self.__data[9], self.__data[10], self.__data[11])
        res += " [%s, %s, %s, %s]]" % (self.__data[12], self.__data[13], self.__data[14], self.__data[15])
        return res

    def __add__(self, *args):
        other = self.__unpack_values(args)
        return Mat4(self.__data[0] + other[0],
                    self.__data[1] + other[1],
                    self.__data[2] + other[2],
                    self.__data[3] + other[3],
                    self.__data[4] + other[4],
                    self.__data[5] + other[5],
                    self.__data[6] + other[6],
                    self.__data[7] + other[7],
                    self.__data[8] + other[8],
                    self.__data[9] + other[9],
                    self.__data[10] + other[10],
                    self.__data[11] + other[11],
                    self.__data[12] + other[12],
                    self.__data[13] + other[13],
                    self.__data[14] + other[14],
                    self.__data[15] + other[15])

    def __sub__(self, *args):
        other = self.__unpack_values(args)
        return Mat4(self.__data[0] - other[0],
                    self.__data[1] - other[1],
                    self.__data[2] - other[2],
                    self.__data[3] - other[3],
                    self.__data[4] - other[4],
                    self.__data[5] - other[5],
                    self.__data[6] - other[6],
                    self.__data[7] - other[7],
                    self.__data[8] - other[8],
                    self.__data[9] - other[9],
                    self.__data[10] - other[10],
                    self.__data[11] - other[11],
                    self.__data[12] - other[12],
                    self.__data[13] - other[13],
                    self.__data[14] - other[14],
                    self.__data[15] - other[15])

    def __mul__(self, *args):
        b = self.__unpack_values(args)
        return Mat4(self.__data[0] * b[0] + self.__data[1] * b[4] + self.__data[2] * b[8] + self.__data[3] * b[12],
                    self.__data[0] * b[1] + self.__data[1] * b[5] + self.__data[2] * b[9] + self.__data[3] * b[13],
                    self.__data[0] * b[2] + self.__data[1] * b[6] + self.__data[2] * b[10] + self.__data[3] * b[14],
                    self.__data[0] * b[3] + self.__data[1] * b[7] + self.__data[2] * b[11] + self.__data[3] * b[15],

                    self.__data[4] * b[0] + self.__data[5] * b[4] + self.__data[6] * b[8] + self.__data[7] * b[12],
                    self.__data[4] * b[1] + self.__data[5] * b[5] + self.__data[6] * b[9] + self.__data[7] * b[13],
                    self.__data[4] * b[2] + self.__data[5] * b[6] + self.__data[6] * b[10] + self.__data[7] * b[14],
                    self.__data[4] * b[3] + self.__data[5] * b[7] + self.__data[6] * b[11] + self.__data[7] * b[15],

                    self.__data[8] * b[0] + self.__data[9] * b[4] + self.__data[10] * b[8] + self.__data[11] * b[12],
                    self.__data[8] * b[1] + self.__data[9] * b[5] + self.__data[10] * b[9] + self.__data[11] * b[13],
                    self.__data[8] * b[2] + self.__data[9] * b[6] + self.__data[10] * b[10] + self.__data[11] * b[14],
                    self.__data[8] * b[3] + self.__data[9] * b[7] + self.__data[10] * b[11] + self.__data[11] * b[15],

                    self.__data[12] * b[0] + self.__data[13] * b[4] + self.__data[14] * b[8] + self.__data[15] * b[12],
                    self.__data[12] * b[1] + self.__data[13] * b[5] + self.__data[14] * b[9] + self.__data[15] * b[13],
                    self.__data[12] * b[2] + self.__data[13] * b[6] + self.__data[14] * b[10] + self.__data[15] * b[14],
                    self.__data[12] * b[3] + self.__data[13] * b[7] + self.__data[14] * b[11] + self.__data[15] * b[15])

    # row 1 set/get
    @property
    def m00(self) -> float: return self.__data[0]

    @m00.setter
    def m00(self, val: float): self.__data[0] = val

    @property
    def m01(self) -> float: return self.__data[1]

    @m01.setter
    def m01(self, val: float): self.__data[1] = val

    @property
    def m02(self) -> float: return self.__data[2]

    @m02.setter
    def m02(self, val: float): self.__data[2] = val

    @property
    def m03(self) -> float: return self.__data[3]

    @m03.setter
    def m03(self, val: float): self.__data[3] = val

    # row 2 set/get
    @property
    def m10(self) -> float: return self.__data[4]

    @m10.setter
    def m10(self, val: float): self.__data[4] = val

    @property
    def m11(self) -> float: return self.__data[5]

    @m11.setter
    def m11(self, val: float): self.__data[5] = val

    @property
    def m12(self) -> float: return self.__data[6]

    @m12.setter
    def m12(self, val: float): self.__data[6] = val

    @property
    def m13(self) -> float: return self.__data[7]

    @m13.setter
    def m13(self, val: float): self.__data[7] = val

    # row 3 set/get
    @property
    def m20(self) -> float: return self.__data[8]

    @m20.setter
    def m20(self, val: float): self.__data[8] = val

    @property
    def m21(self) -> float: return self.__data[9]

    @m21.setter
    def m21(self, val: float): self.__data[9] = val

    @property
    def m22(self) -> float: return self.__data[10]

    @m22.setter
    def m22(self, val: float): self.__data[10] = val

    @property
    def m23(self) -> float: return self.__data[11]

    @m23.setter
    def m23(self, val: float): self.__data[11] = val

    # row 4 set/get
    @property
    def m30(self) -> float: return self.__data[12]

    @m30.setter
    def m30(self, val: float): self.__data[12] = val

    @property
    def m31(self) -> float: return self.__data[13]

    @m31.setter
    def m31(self, val: float): self.__data[13] = val

    @property
    def m32(self) -> float: return self.__data[14]

    @m32.setter
    def m32(self, val: float): self.__data[14] = val

    @property
    def m33(self) -> float: return self.__data[15]

    @m33.setter
    def m33(self, val: float): self.__data[15] = val