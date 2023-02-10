from ctypes import Structure, POINTER, c_int8, c_int32, CDLL, c_float
import matplotlib.pyplot as plt
import numpy as np
import platform
import random
import os

from cgeo import Vec3, LoopTimer

path = os.getcwd()
#  interpolator_lib = CDLL(path + "\interpolation.dll")
interpolator_lib = None

if platform.system() == 'Linux':
    interpolator_lib = CDLL(path + "\interpolation.dll")  # ('./path_finder/lib_astar.so')
elif platform.system() == 'Windows':
    if platform.architecture()[0] == '64bit':
        interpolator_lib = CDLL(path + "\interpolation.dll")  # ('./path_finder/x64/AStar.dll')
    else:
        interpolator_lib = CDLL(path + "\interpolation.dll")  # ('./path_finder/x86/AStar.dll')
if interpolator_lib is None:
    raise ImportError("unable to find AStar.dll...")


NP_ARRAY_1_D_POINTER = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="aligned, contiguous")
NP_ARRAY_2_D_POINTER = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="aligned, contiguous")
NP_ARRAY_3_D_POINTER = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags="aligned, contiguous")

NP_ARRAY_1_D_POINTER_RW = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="aligned, contiguous, writeable")
NP_ARRAY_2_D_POINTER_RW = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="aligned, contiguous, writeable")
NP_ARRAY_3_D_POINTER_RW = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags="aligned, contiguous, writeable")


class _NumpyArray1D(Structure):
    _fields_ = ("data", NP_ARRAY_1_D_POINTER), \
               ("size", c_int32)


class _NumpyArray2D(Structure):
    _fields_ = ("data", NP_ARRAY_2_D_POINTER), \
               ("rows", c_int32), \
               ("cols", c_int32)


class _NumpyArray3D(Structure):
    _fields_ = ("data",   NP_ARRAY_2_D_POINTER), \
               ("rows",   c_int32), \
               ("cols",   c_int32), \
               ("layers", c_int32)


class _Interpolator(Structure):
    _fields_ = ("control_points", POINTER(_NumpyArray2D)), \
               ("width", c_float), \
               ("height", c_float), \
               ("x0", c_float), \
               ("y0", c_float), \
               ("z0", c_float)


np_array_1d_new                       = interpolator_lib.np_array_1d_new
np_array_1d_new.argtypes              = [c_int32, NP_ARRAY_1_D_POINTER]
np_array_1d_new.restype               =  POINTER(_NumpyArray1D)   
                                      
np_array_1d_del                       = interpolator_lib.np_array_1d_del
np_array_1d_del.argtypes              = [POINTER(_NumpyArray1D)]
                                      
np_array_2d_new                       = interpolator_lib.np_array_2d_new
np_array_2d_new.argtypes              = [c_int32, c_int32, NP_ARRAY_2_D_POINTER]
np_array_2d_new.restype               =  POINTER(_NumpyArray2D)   
                                      
np_array_2d_del                       = interpolator_lib.np_array_2d_del
np_array_2d_del.argtypes              = [POINTER(_NumpyArray2D)]

interpolator_new                      = interpolator_lib.interpolator_new
interpolator_new.argtypes             = [c_int32, c_int32, NP_ARRAY_2_D_POINTER]
interpolator_new.restype              =  POINTER(_Interpolator)   
                                      
interpolator_del                      = interpolator_lib.interpolator_del
interpolator_del.argtypes             = [POINTER(_Interpolator)]

interpolate_pt                        = interpolator_lib.interpolate_pt
interpolate_pt.argtypes               = [c_float, c_float, POINTER(_Interpolator), c_int8]
interpolate_pt.restype                =  c_float   
                                         
interpolate_x_derivative_pt           = interpolator_lib.interpolate_x_derivative_pt
interpolate_x_derivative_pt.argtypes  = [c_float, c_float, POINTER(_Interpolator), c_int8, c_float]
interpolate_x_derivative_pt.restype   =  c_float   
                                      
interpolate_y_derivative_pt           = interpolator_lib.interpolate_y_derivative_pt
interpolate_y_derivative_pt.argtypes  = [c_float, c_float, POINTER(_Interpolator), c_int8, c_float]
interpolate_y_derivative_pt.restype   =  c_float   
 
interpolate_xy_derivative_pt          = interpolator_lib.interpolate_xy_derivative_pt
interpolate_xy_derivative_pt.argtypes = [c_float, c_float, POINTER(_Interpolator), c_int8, c_float, c_float]
interpolate_xy_derivative_pt.restype  =  c_float   

interpolate                           = interpolator_lib.interpolate
interpolate.argtypes                  = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D),
                                         POINTER(_Interpolator), c_int8]
                                      
interpolate_x_derivative              = interpolator_lib.interpolate_x_derivative
interpolate_x_derivative.argtypes     = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D),
                                         POINTER(_Interpolator), c_int8, c_float]
                                      
interpolate_y_derivative              = interpolator_lib.interpolate_y_derivative
interpolate_y_derivative.argtypes     = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D),
                                         POINTER(_Interpolator), c_int8, c_float]
                                      
interpolate_xy_derivative             = interpolator_lib.interpolate_xy_derivative
interpolate_xy_derivative.argtypes    = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D),
                                         POINTER(_Interpolator), c_int8, c_float, c_float]
                                      
interpolate2                          = interpolator_lib.interpolate2
interpolate2.argtypes                 = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D),
                                         POINTER(_Interpolator), c_int8]
                                      
interpolate_x_derivative2             = interpolator_lib.interpolate_x_derivative2
interpolate_x_derivative2.argtypes    = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D),
                                         POINTER(_Interpolator), c_int8, c_float]
                                      
interpolate_y_derivative2             = interpolator_lib.interpolate_y_derivative2
interpolate_y_derivative2.argtypes    = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D),
                                         POINTER(_Interpolator), c_int8, c_float]
                                      
interpolate_xy_derivative2            = interpolator_lib.interpolate_xy_derivative2
interpolate_xy_derivative2.argtypes   = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D),
                                         POINTER(_Interpolator), c_int8, c_float, c_float]


# TODO check: https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
class Array1D:
    def __init__(self, array: np.ndarray):
        self.__array = None
        if array.ndim != 1:
            raise RuntimeError()
        self.__array =  np_array_1d_new(array.size, array)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        if self.__array is None:
            return
        np_array_1d_del(self.__array)

    @property
    def size(self) -> int:
        return self.__array.contents.size

    @property
    def array(self):
        return self.__array.contents.data

    @property
    def ptr(self) -> _NumpyArray1D:
        return self.__array


# TODO check: https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
class Array2D:
    def __init__(self, array: np.ndarray):
        self.__array = None
        if array.ndim != 2:
            raise RuntimeError()
        self.__array = np_array_2d_new(array.shape[0], array.shape[1], array)

    def __del__(self):
        if self.__array is None:
            return
        np_array_2d_del(self.__array)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def rows(self) -> int:
        return self.__array.contents.rows

    @property
    def cols(self) -> int:
        return self.__array.contents.cols

    @property
    def size(self) -> int:
        return self.rows * self.cols

    @property
    def array(self):
        return self.__array.contents.data

    @property
    def ptr(self) -> _NumpyArray2D:
        return self.__array


# TODO check: https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
# class Array3D:
#     def __init__(self, array: np.ndarray):
#         self.__array = None
#         if array.ndim != 3:
#             raise RuntimeError()
#         # self.__array = np_array_3d_new(array.shape[0], array.shape[1],  array.shape[2], array)
# 
#     def __del__(self):
#         if self.__array is None:
#             return
#         # np_array_3d_del(self.__array)
# 
#     def __enter__(self):
#         return self
# 
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         pass
# 
#     @property
#     def rows(self) -> int:
#         return self.__array.contents.rows
# 
#     @property
#     def cols(self) -> int:
#         return self.__array.contents.cols
# 
#     @property
#     def size(self) -> int:
#         return self.rows * self.cols
# 
#     @property
#     def array(self):
#         return self.__array.contents.data
# 
#     @property
#     def ptr(self) -> _NumpyArray3D:
#         return self.__array


class Interpolator:
    # 562.6785175001714 times faster than pure Python implementation =)
    def __init__(self, points: np.ndarray = None):
        if points is None:
            self.__control_points:     np.ndarray = np.array([[1.0, 0.0, 1.0, 0.0],
                                                              [0.0, 1.0, 0.0, 1.0],
                                                              [1.0, 0.0, 1.0, 0.0]],
                                                             dtype=np.float32)
            self.__interpolator_ptr:  _Interpolator = interpolator_new(3, 4, self.__control_points)
        if points.ndim != 2 or points.dtype != np.float32:
            raise RuntimeError("Interpolator :: points.ndim != 2")
        self.__control_points: np.ndarray = points
        self.__interpolator_ptr: _Interpolator = interpolator_new(points.shape[0], points.shape[1], self.__control_points)

    def __del__(self):
        interpolator_del(self.__interpolator_ptr)

    @property
    def rows(self) -> int:
        return self.__interpolator_ptr.contents.rows

    @property
    def cols(self) -> int:
        return self.__interpolator_ptr.contents.cols

    @property
    def x0(self) -> float:
        return self.__interpolator_ptr.contents.x0

    @x0.setter
    def x0(self, value: float) -> None:
        self.__interpolator_ptr.contents.x0 = value

    @property
    def y0(self) -> float:
        return self.__interpolator_ptr.contents.y0

    @y0.setter
    def y0(self, value: float) -> None:
        self.__interpolator_ptr.contents.y0 = value

    @property
    def z0(self) -> float:
        return self.__interpolator_ptr.contents.x0

    @z0.setter
    def z0(self, value: float) -> None:
        self.__interpolator_ptr.contents.z0 = value

    @property
    def width(self) -> float:
        return self.__interpolator_ptr.contents.width

    @width.setter
    def width(self, value: float) -> None:
        if value <= 0.0:
            return
        self.__interpolator_ptr.contents.width = value

    @property
    def height(self) -> float:
        return self.__interpolator_ptr.contents.height

    @height.setter
    def height(self, value: float) -> None:
        if value <= 0.0:
            return
        self.__interpolator_ptr.contents.height = value

    @property
    def orig(self) -> Vec3:
        return Vec3(self.x0, self.y0, self.z0)

    @orig.setter
    def orig(self, orig: Vec3) -> None:
        self.x0 = orig.x
        self.y0 = orig.y
        self.z0 = orig.z

    @property
    def control_points(self) ->  np.ndarray:
        return self.__control_points

    @control_points.setter
    def control_points(self, points: np.ndarray) -> None:
        if points.ndim != 2:
            return
        self.__control_points = points
        width, height = self.width, self.height
        x0, y0, z0 = self.x0, self.y0, self.z0
        interpolator_del(self.__interpolator_ptr)
        self.__interpolator_ptr:  _Interpolator = interpolator_new(self.rows, self.cols, self.__control_points)
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.width, self.height = width, height

    def interpolate_pt(self, x: float, y: float, mode: int = 1) -> float:
        return interpolate_pt(x, y, self.__interpolator_ptr, mode)

    def interpolate_x_derivative_pt(self, x: float, y: float, dx: float = 1e-3, mode: int = 1) -> float:
        return interpolate_x_derivative_pt(x, y, self.__interpolator_ptr, mode, dx)

    def interpolate_y_derivative_pt(self, x: float, y: float, dy: float = 1e-3, mode: int = 1) -> float:
        return interpolate_y_derivative_pt(x, y, self.__interpolator_ptr, mode, dy)

    def interpolate_xy_derivative_pt(self, x: float, y: float, dx: float = 1e-3,
                                     dy: float = 1e-3, mode: int = 1) -> float:
        return interpolate_xy_derivative_pt(x, y, self.__interpolator_ptr, mode, dx, dy)

    def interpolate(self, x: np.ndarray, y: np.ndarray, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((min(x.size, y.size),), dtype=np.float32)
        with Array1D(res) as res_, Array1D(x) as x_, Array1D(y) as y_:
            interpolate(res_.ptr, x_.ptr, y_.ptr, self.__interpolator_ptr, mode)
        return res

    def interpolate_x_derivative(self, x: np.ndarray, y: np.ndarray, dx: float = 1e-3, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((min(x.size, y.size),), dtype=np.float32)
        with Array1D(res) as res_, Array1D(x) as x_, Array1D(y) as y_:
            interpolate_x_derivative(res_.ptr, x_.ptr, y_.ptr, self.__interpolator_ptr, mode, dx)
        return res

    def interpolate_y_derivative(self, x: np.ndarray, y: np.ndarray, dy: float = 1e-3, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((min(x.size, y.size),), dtype=np.float32)
        with Array1D(res) as res_, Array1D(x) as x_, Array1D(y) as y_:
            interpolate_y_derivative(res_.ptr, x_.ptr, y_.ptr, self.__interpolator_ptr, mode, dy)
        return res

    def interpolate_xy_derivative(self, x:  np.ndarray, y:  np.ndarray, dx: float = 1e-3,
                                  dy: float = 1e-3, mode: int = 1) ->  np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((min(x.size, y.size),), dtype=np.float32)
        with Array1D(res) as res_, Array1D(x) as x_, Array1D(y) as y_:
            interpolate_xy_derivative(res_.ptr, x_.ptr, y_.ptr, self.__interpolator_ptr, mode, dx, dy)
        return res

    def interpolate2(self, x: np.ndarray, y: np.ndarray, mode: int = 1) -> np.ndarray:
        timer = LoopTimer()
        with timer:
            if x.ndim != 1:
                raise RuntimeError()
            if y.ndim != 1:
                raise RuntimeError()
            res = np.zeros((x.size, y.size,), dtype=np.float32)
            with Array2D(res) as res_, Array1D(x) as x_, Array1D(y) as y_:
                interpolate2(res_.ptr, x_.ptr, y_.ptr, self.__interpolator_ptr, mode)
        print(f"interpolate2 time: {timer.last_loop_time}")
        return res

    def interpolate_x_derivative2(self, x: np.ndarray, y: np.ndarray, dx: float = 1e-3, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((x.size, y.size,), dtype=np.float32)
        with Array2D(res) as res_, Array1D(x) as x_, Array1D(y) as y_:
            interpolate_x_derivative2(res_.ptr, x_.ptr, y_.ptr, self.__interpolator_ptr, mode, dx)
        return res

    def interpolate_y_derivative2(self, x: np.ndarray, y: np.ndarray, dy: float = 1e-3, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((x.size, y.size,), dtype=np.float32)
        with Array2D(res) as res_, Array1D(x) as x_, Array1D(y) as y_:
            interpolate_y_derivative2(res_.ptr, x_.ptr, y_.ptr, self.__interpolator_ptr, mode, dy)
        return res

    def interpolate_xy_derivative2(self, x:  np.ndarray, y:  np.ndarray, dx: float = 1e-3,
                                   dy: float = 1e-3, mode: int = 1) ->  np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((x.size, y.size,), dtype=np.float32)
        with Array2D(res) as res_, Array1D(x) as x_, Array1D(y) as y_:
            interpolate_xy_derivative2(res_.ptr, x_.ptr, y_.ptr, self.__interpolator_ptr, mode, dx, dy)
        return res


if __name__ == "__main__":

    pts = np.array([[random.uniform(-5.0, 5.0) for _ in range(32)]for _ in range(32)], dtype=np.float32)

    interpolator = Interpolator(pts)
    points_n = 1024
    x_ = np.linspace(0, 1, points_n, dtype=np.float32)
    #
    # z = [[interpolator.interpolate_pt(xi, yi, 2) for xi in x]for yi in x]
    z_ = interpolator.interpolate2(x_, x_, 2)
    # with timer:
    #    z = interpolator.interpolate_x_derivative2(x, x, 2)
    #print(f"loop time: {timer.last_loop_time}")
    plt.imshow(z_)
    plt.show()
    #print(interpolator.interpolate_pt(0.1, 0.1))
