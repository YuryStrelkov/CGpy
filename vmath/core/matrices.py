from typing import List, Tuple
import numpy as np


class Mat3:

    @staticmethod
    def identity():
        return Mat3(1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0)

    @staticmethod
    def zeros():
        return Mat3(0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0)

    @staticmethod
    def ones():
        return Mat3(1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0)

    @staticmethod
    def __unpack_args(*args) -> tuple:

        args = args[0]

        number_of_args = len(args)

        if number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is Mat3:
                return args[0].m00, args[0].m01, args[0].m02, \
                       args[0].m10, args[0].m11, args[0].m12, \
                       args[0].m20, args[0].m21, args[0].m22

            if arg_type is float or arg_type is int:  # single int or float argument
                return args[0], 0.0, 0.0, \
                       0.0, args[0], 0.0, \
                       0.0, 0.0, args[0]

        if number_of_args == 9:
            return args

        if number_of_args == 0:
            return 0.0, 0.0, 0.0, \
                   0.0, 0.0, 0.0, \
                   0.0, 0.0, 0.0  # no arguments

        if number_of_args == 9:
            return args[0], args[1], args[2], \
                   args[3], args[4], args[5], \
                   args[6], args[7], args[8]  # x, y and z passed in

        raise TypeError(f'Invalid Input: {args}')

    @staticmethod
    def from_np_array(data: np.ndarray):
        m = Mat3()
        i: int = 0
        for element in data.flat:
            m[i] = element
            i += 1
            if i == 9:
                break
        return m

    @property
    def unique_id(self) -> int:
        return id(self)

    @property
    def as_list(self) -> List[float]:
        return self.__data

    @property
    def np_array(self) -> np.ndarray:
        return np.array(self.__data, dtype=np.float32)

    @property
    def np_array_3x3(self) -> np.ndarray:
        return self.np_array.reshape(3, 3)

    @property
    def as_tuple(self) -> Tuple[float, float, float,
                                float, float, float,
                                float, float, float]:
        return self.__data[0], self.__data[1], self.__data[2],\
               self.__data[3], self.__data[4], self.__data[5],\
               self.__data[6], self.__data[7], self.__data[8]

    # row 1 set/get
    @property
    def m00(self) -> float:
        return self.__data[0]

    @m00.setter
    def m00(self, val: float):
        self.__data[0] = val

    @property
    def m01(self) -> float:
        return self.__data[1]

    @m01.setter
    def m01(self, val: float):
        self.__data[1] = val

    @property
    def m02(self) -> float:
        return self.__data[2]

    @m02.setter
    def m02(self, val: float):
        self.__data[2] = val

    # row 2 set/get
    @property
    def m10(self) -> float:
        return self.__data[3]

    @m10.setter
    def m10(self, val: float):
        self.__data[3] = val

    @property
    def m11(self) -> float:
        return self.__data[4]

    @m11.setter
    def m11(self, val: float):
        self.__data[4] = val

    @property
    def m12(self) -> float:
        return self.__data[5]

    @m12.setter
    def m12(self, val: float):
        self.__data[5] = val

    # row 3 set/get
    @property
    def m20(self) -> float:
        return self.__data[6]

    @m20.setter
    def m20(self, val: float):
        self.__data[6] = val

    @property
    def m21(self) -> float:
        return self.__data[7]

    @m21.setter
    def m21(self, val: float):
        self.__data[7] = val

    @property
    def m22(self) -> float:
        return self.__data[8]

    @m22.setter
    def m22(self, val: float):
        self.__data[8] = val

    def invert(self):
        det: float = (self.m00 * (self.m11 * self.m22 - self.m21 * self.m12) -
                      self.m01 * (self.m10 * self.m22 - self.m12 * self.m20) +
                      self.m02 * (self.m10 * self.m21 - self.m11 * self.m20))
        if abs(det) < 1e-12:
            raise ArithmeticError("Mat3 :: singular matrix")
        det = 1.0 / det

        self.__data = [(self.m11 * self.m22 - self.m21 * self.m12) * det,
                       (self.m02 * self.m21 - self.m01 * self.m22) * det,
                       (self.m01 * self.m12 - self.m02 * self.m11) * det,
                       (self.m12 * self.m20 - self.m10 * self.m22) * det,
                       (self.m00 * self.m22 - self.m02 * self.m20) * det,
                       (self.m10 * self.m02 - self.m00 * self.m12) * det,
                       (self.m10 * self.m21 - self.m20 * self.m11) * det,
                       (self.m20 * self.m01 - self.m00 * self.m21) * det,
                       (self.m00 * self.m11 - self.m10 * self.m01) * det]
        return self

    __slots__ = "__data"

    def __init__(self, *args):
        self.__data: List[float] = list(Mat3.__unpack_args(args))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Mat3):
            return False
        for i in range(9):
            if not (self.__data[i] == other.__data[i]):
                return False
        return True

    def __hash__(self) -> int:
        return hash(self.__data)

    def __getitem__(self, index: int) -> float:
        if index < 0 or index >= 9:
            raise IndexError(f"Mat3 :: trying to access index: {index}")
        return self.__data[index]

    def __setitem__(self, index: int, value: float):
        if index < 0 or index >= 9:
            raise IndexError(f"Mat3 :: trying to access index: {index}")
        self.__data[index] = value

    def __str__(self) -> str:
        return f"{{\n\t\"m00\": {self.m00:20}, \"m01\": {self.m01:20}, \"m02\": {self.m02:20}],\n" \
                   f"\t\"m10\": {self.m10:20}, \"m11\": {self.m11:20}, \"m12\": {self.m12:20}],\n" \
                   f"\t\"m20\": {self.m20:20}, \"m21\": {self.m21:20}, \"m22\": {self.m22:20}\n}}\n"

    def __add__(self, *args):
        other = self.__unpack_args(args)
        return Mat3(self.__data[0] + other[0],
                    self.__data[1] + other[1],
                    self.__data[2] + other[2],
                    self.__data[3] + other[3],
                    self.__data[4] + other[4],
                    self.__data[5] + other[5],
                    self.__data[6] + other[6],
                    self.__data[7] + other[7],
                    self.__data[8] + other[8])

    def __iadd__(self, *args):
        other = self.__unpack_args(args)
        self.__data[0] += other[0]
        self.__data[1] += other[1]
        self.__data[2] += other[2]
        self.__data[3] += other[3]
        self.__data[4] += other[4]
        self.__data[5] += other[5]
        self.__data[6] += other[6]
        self.__data[7] += other[7]
        self.__data[8] += other[8]
        return self

    __radd__ = __add__

    def __sub__(self, *args):
        other = self.__unpack_args(args)
        return Mat3(self.__data[0] - other[0],
                    self.__data[1] - other[1],
                    self.__data[2] - other[2],
                    self.__data[3] - other[3],
                    self.__data[4] - other[4],
                    self.__data[5] - other[5],
                    self.__data[6] - other[6],
                    self.__data[7] - other[7],
                    self.__data[8] - other[8])

    def __isub__(self, *args):
        other = self.__unpack_args(args)
        self.__data[0] -= other[0]
        self.__data[1] -= other[1]
        self.__data[2] -= other[2]
        self.__data[3] -= other[3]
        self.__data[4] -= other[4]
        self.__data[5] -= other[5]
        self.__data[6] -= other[6]
        self.__data[7] -= other[7]
        self.__data[8] -= other[8]
        return self

    def __rsub__(self, *args):
        other = self.__unpack_args(args)
        return Mat3(other[0] - self.__data[0],
                    other[1] - self.__data[1],
                    other[2] - self.__data[2],
                    other[3] - self.__data[3],
                    other[4] - self.__data[4],
                    other[5] - self.__data[5],
                    other[6] - self.__data[6],
                    other[7] - self.__data[7],
                    other[8] - self.__data[8])

    def __mul__(self, *args):
        b = self.__unpack_args(args)
        return Mat3(self.__data[0] * b[0] + self.__data[1] * b[3] + self.__data[2] * b[6],
                    self.__data[0] * b[1] + self.__data[1] * b[4] + self.__data[2] * b[7],
                    self.__data[0] * b[2] + self.__data[1] * b[5] + self.__data[2] * b[8],

                    self.__data[3] * b[0] + self.__data[4] * b[3] + self.__data[5] * b[6],
                    self.__data[3] * b[1] + self.__data[4] * b[4] + self.__data[5] * b[7],
                    self.__data[3] * b[2] + self.__data[4] * b[5] + self.__data[5] * b[8],

                    self.__data[6] * b[0] + self.__data[7] * b[3] + self.__data[8] * b[6],
                    self.__data[6] * b[1] + self.__data[7] * b[4] + self.__data[8] * b[7],
                    self.__data[6] * b[2] + self.__data[7] * b[5] + self.__data[8] * b[8])

    def __imul__(self, *args):
        b = self.__unpack_args(args)
        res: [float] = \
            [self.__data[0] * b[0] + self.__data[1] * b[3] + self.__data[2] * b[6],
             self.__data[0] * b[1] + self.__data[1] * b[4] + self.__data[2] * b[7],
             self.__data[0] * b[2] + self.__data[1] * b[5] + self.__data[2] * b[8],

             self.__data[3] * b[0] + self.__data[4] * b[3] + self.__data[5] * b[6],
             self.__data[3] * b[1] + self.__data[4] * b[4] + self.__data[5] * b[7],
             self.__data[3] * b[2] + self.__data[4] * b[5] + self.__data[5] * b[8],

             self.__data[6] * b[0] + self.__data[7] * b[3] + self.__data[8] * b[6],
             self.__data[6] * b[1] + self.__data[7] * b[4] + self.__data[8] * b[7],
             self.__data[6] * b[2] + self.__data[7] * b[5] + self.__data[8] * b[8]]
        self.__data = res
        return self

    def __rmul__(self, *args):
        b = self.__unpack_args(args)
        return Mat3(b[0] * self.__data[0] + b[1] * self.__data[3] + b[2] * self.__data[6],
                    b[0] * self.__data[1] + b[1] * self.__data[4] + b[2] * self.__data[7],
                    b[0] * self.__data[2] + b[1] * self.__data[5] + b[2] * self.__data[8],

                    b[3] * self.__data[0] + b[4] * self.__data[3] + b[5] * self.__data[6],
                    b[3] * self.__data[1] + b[4] * self.__data[4] + b[5] * self.__data[7],
                    b[3] * self.__data[2] + b[4] * self.__data[5] + b[5] * self.__data[8],

                    b[6] * self.__data[0] + b[7] * self.__data[3] + b[8] * self.__data[6],
                    b[6] * self.__data[1] + b[7] * self.__data[4] + b[8] * self.__data[7],
                    b[6] * self.__data[2] + b[7] * self.__data[5] + b[8] * self.__data[8])


class Mat4:

    @staticmethod
    def identity():
        return Mat4(1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def zeros():
        return Mat4(0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def ones():
        return Mat4(1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def __unpack_args(*args) -> tuple:
        args = args[0]

        number_of_args = len(args)

        if number_of_args == 1:  # one argument
            arg_type = type(args[0])

            if arg_type is Mat4:
                return args[0].m00, args[0].m01, args[0].m02, args[0].m03, \
                       args[0].m10, args[0].m11, args[0].m12, args[0].m13, \
                       args[0].m20, args[0].m21, args[0].m22, args[0].m23, \
                       args[0].m30, args[0].m31, args[0].m32, args[0].m33

            if arg_type is float or arg_type is int:  # single int or float argument
                return args[0], 0.0, 0.0, 0.0, \
                       0.0, args[0], 0.0, 0.0, \
                       0.0, 0.0, args[0], 0.0, \
                       0.0, 0.0, 0.0, args[0]

        if number_of_args == 16:
            return args

        if number_of_args == 0:
            return 0.0, 0.0, 0.0, 0.0, \
                   0.0, 0.0, 0.0, 0.0, \
                   0.0, 0.0, 0.0, 0.0, \
                   0.0, 0.0, 0.0, 0.0  # no arguments

        if number_of_args == 9:
            return args[0], args[1], args[2], args[3], \
                   args[4], args[5], args[6], args[7], \
                   args[8], args[9], args[10], args[11], \
                   args[12], args[13], args[14], args[15]  # x, y and z passed in

        raise TypeError(f'Invalid Input: {args}')

    @staticmethod
    def from_np_array(data: np.ndarray):
        m = Mat4()
        i: int = 0
        for element in data.flat:
            m[i] = element
            i += 1
            if i == 16:
                break
        return m

    @property
    def as_list(self) -> List[float]:
        return self.__data

    @property
    def unique_id(self) -> int:
        return id(self)

    @property
    def np_array(self) -> np.ndarray:
        return np.array(self.__data, dtype=np.float32)

    @property
    def np_array_4x4(self) -> np.ndarray:
        return self.np_array.reshape(4, 4)

    @property
    def as_tuple(self) -> Tuple[float, float, float, float,
                                float, float, float, float,
                                float, float, float, float,
                                float, float, float, float]:
        return self.__data[0], self.__data[1], self.__data[2], self.__data[3],\
               self.__data[4], self.__data[5], self.__data[6], self.__data[7],\
               self.__data[8], self.__data[9], self.__data[10], self.__data[11],\
               self.__data[12], self.__data[13], self.__data[14], self.__data[15]

    @property
    def m00(self) -> float:
        return self.__data[0]

    @m00.setter
    def m00(self, val: float):
        self.__data[0] = val

    @property
    def m01(self) -> float:
        return self.__data[1]

    @m01.setter
    def m01(self, val: float):
        self.__data[1] = val

    @property
    def m02(self) -> float:
        return self.__data[2]

    @m02.setter
    def m02(self, val: float):
        self.__data[2] = val

    @property
    def m03(self) -> float:
        return self.__data[3]

    @m03.setter
    def m03(self, val: float):
        self.__data[3] = val

    # row 2 set/get
    @property
    def m10(self) -> float:
        return self.__data[4]

    @m10.setter
    def m10(self, val: float):
        self.__data[4] = val

    @property
    def m11(self) -> float:
        return self.__data[5]

    @m11.setter
    def m11(self, val: float):
        self.__data[5] = val

    @property
    def m12(self) -> float:
        return self.__data[6]

    @m12.setter
    def m12(self, val: float):
        self.__data[6] = val

    @property
    def m13(self) -> float:
        return self.__data[7]

    @m13.setter
    def m13(self, val: float):
        self.__data[7] = val

    # row 3 set/get
    @property
    def m20(self) -> float:
        return self.__data[8]

    @m20.setter
    def m20(self, val: float):
        self.__data[8] = val

    @property
    def m21(self) -> float:
        return self.__data[9]

    @m21.setter
    def m21(self, val: float):
        self.__data[9] = val

    @property
    def m22(self) -> float:
        return self.__data[10]

    @m22.setter
    def m22(self, val: float):
        self.__data[10] = val

    @property
    def m23(self) -> float:
        return self.__data[11]

    @m23.setter
    def m23(self, val: float):
        self.__data[11] = val

    # row 4 set/get
    @property
    def m30(self) -> float:
        return self.__data[12]

    @m30.setter
    def m30(self, val: float):
        self.__data[12] = val

    @property
    def m31(self) -> float:
        return self.__data[13]

    @m31.setter
    def m31(self, val: float):
        self.__data[13] = val

    @property
    def m32(self) -> float:
        return self.__data[14]

    @m32.setter
    def m32(self, val: float):
        self.__data[14] = val

    @property
    def m33(self) -> float:
        return self.__data[15]

    @m33.setter
    def m33(self, val: float):
        self.__data[15] = val

    def invert(self):
        a2323: float = self.m22 * self.m33 - self.m23 * self.m32
        a1323: float = self.m21 * self.m33 - self.m23 * self.m31
        a1223: float = self.m21 * self.m32 - self.m22 * self.m31
        a0323: float = self.m20 * self.m33 - self.m23 * self.m30
        a0223: float = self.m20 * self.m32 - self.m22 * self.m30
        a0123: float = self.m20 * self.m31 - self.m21 * self.m30
        a2313: float = self.m12 * self.m33 - self.m13 * self.m32
        a1313: float = self.m11 * self.m33 - self.m13 * self.m31
        a1213: float = self.m11 * self.m32 - self.m12 * self.m31
        a2312: float = self.m12 * self.m23 - self.m13 * self.m22
        a1312: float = self.m11 * self.m23 - self.m13 * self.m21
        a1212: float = self.m11 * self.m22 - self.m12 * self.m21
        a0313: float = self.m10 * self.m33 - self.m13 * self.m30
        a0213: float = self.m10 * self.m32 - self.m12 * self.m30
        a0312: float = self.m10 * self.m23 - self.m13 * self.m20
        a0212: float = self.m10 * self.m22 - self.m12 * self.m20
        a0113: float = self.m10 * self.m31 - self.m11 * self.m30
        a0112: float = self.m10 * self.m21 - self.m11 * self.m20

        det: float = self.m00 * (self.m11 * a2323 - self.m12 * a1323 + self.m13 * a1223) \
                     - self.m01 * (self.m10 * a2323 - self.m12 * a0323 + self.m13 * a0223) \
                     + self.m02 * (self.m10 * a1323 - self.m11 * a0323 + self.m13 * a0123) \
                     - self.m03 * (self.m10 * a1223 - self.m11 * a0223 + self.m12 * a0123)

        if abs(det) < 1e-12:
            raise ArithmeticError("Mat4 :: singular matrix")

        det = 1.0 / det

        # computes the inverse of a matrix m

        self.__data = var = [det * (self.m11 * a2323 - self.m12 * a1323 + self.m13 * a1223),
                             det * -(self.m01 * a2323 - self.m02 * a1323 + self.m03 * a1223),
                             det * (self.m01 * a2313 - self.m02 * a1313 + self.m03 * a1213),
                             det * -(self.m01 * a2312 - self.m02 * a1312 + self.m03 * a1212),
                             det * -(self.m10 * a2323 - self.m12 * a0323 + self.m13 * a0223),
                             det * (self.m00 * a2323 - self.m02 * a0323 + self.m03 * a0223),
                             det * -(self.m00 * a2313 - self.m02 * a0313 + self.m03 * a0213),
                             det * (self.m00 * a2312 - self.m02 * a0312 + self.m03 * a0212),
                             det * (self.m10 * a1323 - self.m11 * a0323 + self.m13 * a0123),
                             det * -(self.m00 * a1323 - self.m01 * a0323 + self.m03 * a0123),
                             det * (self.m00 * a1313 - self.m01 * a0313 + self.m03 * a0113),
                             det * -(self.m00 * a1312 - self.m01 * a0312 + self.m03 * a0112),
                             det * -(self.m10 * a1223 - self.m11 * a0223 + self.m12 * a0123),
                             det * (self.m00 * a1223 - self.m01 * a0223 + self.m02 * a0123),
                             det * -(self.m00 * a1213 - self.m01 * a0213 + self.m02 * a0113),
                             det * (self.m00 * a1212 - self.m01 * a0212 + self.m02 * a0112)]
        return self

    __slots__ = "__data"

    def __init__(self, *args):
        self.__data: List[float] = list(Mat4.__unpack_args(args))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Mat4):
            return False
        for i in range(0, 16):
            if not (self.__data[i] == other.__data[i]):
                return False
        return True

    def __hash__(self) -> int:
        return hash(self.__data)

    def __getitem__(self, index: int) -> float:
        if index < 0 or index >= 16:
            raise IndexError(f"Mat4 :: trying to access index: {index}")
        return self.__data[index]

    def __setitem__(self, index: int, value: float):
        if index < 0 or index >= 16:
            raise IndexError(f"Mat4 :: trying to access index: {index}")
        self.__data[index] = value

    def __neg__(self):
        return Mat4(-self.m00, -self.m01, -self.m02, -self.m03,
                    -self.m10, -self.m11, -self.m12, -self.m13,
                    -self.m20, -self.m21, -self.m22, -self.m23,
                    -self.m30, -self.m31, -self.m32, -self.m33)

    def __copy__(self):
        return Mat4(self.m00, self.m01, self.m02, self.m03,
                    self.m10, self.m11, self.m12, self.m13,
                    self.m20, self.m21, self.m22, self.m23,
                    self.m30, self.m31, self.m32, self.m33)

    copy = __copy__

    def __str__(self) -> str:
        return "" \
               f"{{\n\t\"m00\": {self.m00:20}, \"m01\": {self.m01:20}, \"m02\": {self.m02:20}, \"m03\": {self.m03:20},\n"\
               f"\t\"m10\": {self.m10:20}, \"m11\": {self.m11:20}, \"m12\": {self.m12:20}, \"m13\": {self.m13:20},\n"\
               f"\t\"m20\": {self.m20:20}, \"m21\": {self.m21:20}, \"m22\": {self.m22:20}, \"m23\": {self.m23:20},\n"\
               f"\t\"m30\": {self.m30:20}, \"m31\": {self.m31:20}, \"m32\": {self.m32:20}, \"m33\": {self.m33:20}\n}}"

    def __add__(self, *args):
        other = self.__unpack_args(args)
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

    __radd__ = __add__

    def __iadd__(self, *args):
        other = self.__unpack_args(args)
        self.__data[0] += other[0]
        self.__data[1] += other[1]
        self.__data[2] += other[2]
        self.__data[3] += other[3]
        self.__data[4] += other[4]
        self.__data[5] += other[5]
        self.__data[6] += other[6]
        self.__data[7] += other[7]
        self.__data[8] += other[8]
        self.__data[9] += other[9]
        self.__data[10] += other[10]
        self.__data[11] += other[11]
        self.__data[12] += other[12]
        self.__data[13] += other[13]
        self.__data[14] += other[14]
        self.__data[15] += other[15]

    def __sub__(self, *args):
        other = self.__unpack_args(args)
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

    def __rsub__(self, *args):
        other = self.__unpack_args(args)
        return Mat4(other[0] - self.__data[0],
                    other[1] - self.__data[1],
                    other[2] - self.__data[2],
                    other[3] - self.__data[3],
                    other[4] - self.__data[4],
                    other[5] - self.__data[5],
                    other[6] - self.__data[6],
                    other[7] - self.__data[7],
                    other[8] - self.__data[8],
                    other[9] - self.__data[9],
                    other[10] - self.__data[10],
                    other[11] - self.__data[11],
                    other[12] - self.__data[12],
                    other[13] - self.__data[13],
                    other[14] - self.__data[14],
                    other[15] - self.__data[15])

    def __isub__(self, *args):
        other = self.__unpack_args(args)
        self.__data[0] -= other[0]
        self.__data[1] -= other[1]
        self.__data[2] -= other[2]
        self.__data[3] -= other[3]
        self.__data[4] -= other[4]
        self.__data[5] -= other[5]
        self.__data[6] -= other[6]
        self.__data[7] -= other[7]
        self.__data[8] -= other[8]
        self.__data[9] -= other[9]
        self.__data[10] -= other[10]
        self.__data[11] -= other[11]
        self.__data[12] -= other[12]
        self.__data[13] -= other[13]
        self.__data[14] -= other[14]
        self.__data[15] -= other[15]

    def __mul__(self, *args):
        b = self.__unpack_args(args)
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

    def __rmul__(self, *args):
        b = self.__unpack_args(args)
        return Mat4(b[0] * self.__data[0] + b[1] * self.__data[4] + b[2] * self.__data[8] + b[3] * self.__data[12],
                    b[0] * self.__data[1] + b[1] * self.__data[5] + b[2] * self.__data[9] + b[3] * self.__data[13],
                    b[0] * self.__data[2] + b[1] * self.__data[6] + b[2] * self.__data[10] + b[3] * self.__data[14],
                    b[0] * self.__data[3] + b[1] * self.__data[7] + b[2] * self.__data[11] + b[3] * self.__data[15],

                    b[4] * self.__data[0] + b[5] * self.__data[4] + b[6] * self.__data[8] + b[7] * self.__data[12],
                    b[4] * self.__data[1] + b[5] * self.__data[5] + b[6] * self.__data[9] + b[7] * self.__data[13],
                    b[4] * self.__data[2] + b[5] * self.__data[6] + b[6] * self.__data[10] + b[7] * self.__data[14],
                    b[4] * self.__data[3] + b[5] * self.__data[7] + b[6] * self.__data[11] + b[7] * self.__data[15],

                    b[8] * self.__data[0] + b[9] * self.__data[4] + b[10] * self.__data[8] + b[11] * self.__data[12],
                    b[8] * self.__data[1] + b[9] * self.__data[5] + b[10] * self.__data[9] + b[11] * self.__data[13],
                    b[8] * self.__data[2] + b[9] * self.__data[6] + b[10] * self.__data[10] + b[11] * self.__data[14],
                    b[8] * self.__data[3] + b[9] * self.__data[7] + b[10] * self.__data[11] + b[11] * self.__data[15],

                    b[12] * self.__data[0] + b[13] * self.__data[4] + b[14] * self.__data[8] + b[15] * self.__data[12],
                    b[12] * self.__data[1] + b[13] * self.__data[5] + b[14] * self.__data[9] + b[15] * self.__data[13],
                    b[12] * self.__data[2] + b[13] * self.__data[6] + b[14] * self.__data[10] + b[15] * self.__data[14],
                    b[12] * self.__data[3] + b[13] * self.__data[7] + b[14] * self.__data[11] + b[15] * self.__data[15])

    def __imul__(self, *args):
        b = self.__unpack_args(args)
        res: [float] = \
            [self.__data[0] * b[0] + self.__data[1] * b[4] + self.__data[2] * b[8] + self.__data[3] * b[12],
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
             self.__data[12] * b[3] + self.__data[13] * b[7] + self.__data[14] * b[11] + self.__data[15] * b[15]]
        self.__data = res
        return self
