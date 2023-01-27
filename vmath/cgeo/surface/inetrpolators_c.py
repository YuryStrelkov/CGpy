from ctypes import Structure, POINTER, c_int8, c_int32, CDLL, c_float
import numpy as np
import os

from cgeo import Vec3

path = os.getcwd()
interpolators_lib = CDLL(path + "\interpolation.dll")

NP_ARRAY_1_D_POINTER = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")
NP_ARRAY_2_D_POINTER = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")


class _NumpyArray1D(Structure):
    _fields_ = ("data", NP_ARRAY_1_D_POINTER), \
               ("size", c_int32)


class _NumpyArray2D(Structure):
    _fields_ = ("data", NP_ARRAY_2_D_POINTER), \
               ("rows", c_int32), \
               ("cols", c_int32)


class _Interpolator(Structure):
    _fields_ = ("control_points", POINTER(_NumpyArray2D)), \
               ("width", c_float), \
               ("height", c_float), \
               ("x0", c_float), \
               ("y0", c_float), \
               ("z0", c_float), \

np_array_1d_new                       = interpolators_lib.np_array_1d_new
np_array_1d_new.argtypes              = [c_int8, c_int8, NP_ARRAY_1_D_POINTER]
np_array_1d_new.restype               =  POINTER(_NumpyArray1D)   
                                      
np_array_1d_del                       = interpolators_lib.np_array_1d_del
np_array_1d_del.argtypes              = [POINTER(_NumpyArray1D)]
                                      
np_array_2d_new                       = interpolators_lib.np_array_2d_new
np_array_2d_new.argtypes              = [c_int8, c_int8, NP_ARRAY_2_D_POINTER]
np_array_2d_new.restype               =  POINTER(_NumpyArray2D)   
                                      
np_array_2d_del                       = interpolators_lib.np_array_2d_del
np_array_2d_del.argtypes              = [POINTER(_NumpyArray2D)]

interpolator_new                      = interpolators_lib.interpolator_new
interpolator_new.argtypes             = [c_int8, c_int8, NP_ARRAY_2_D_POINTER]
interpolator_new.restype              =  POINTER(_Interpolator)   
                                      
interpolator_del                      = interpolators_lib.np_array_2d_del
interpolator_del.argtypes             = [POINTER(_Interpolator)]

interpolate_pt                        = interpolators_lib.interpolate_pt
interpolate_pt.argtypes               = [c_float, c_float, POINTER(_Interpolator), c_int8]
interpolate_pt.restype                =  c_float   
                                         
interpolate_x_derivative_pt           = interpolators_lib.interpolate_x_derivative_pt
interpolate_x_derivative_pt.argtypes  = [c_float, c_float, POINTER(_Interpolator), c_int8, c_float]
interpolate_x_derivative_pt.restype   =  c_float   
                                      
interpolate_y_derivative_pt           = interpolators_lib.interpolate_y_derivative_pt
interpolate_y_derivative_pt.argtypes  = [c_float, c_float, POINTER(_Interpolator), c_int8, c_float]
interpolate_y_derivative_pt.restype   =  c_float   
 
interpolate_xy_derivative_pt          = interpolators_lib.interpolate_xy_derivative_pt
interpolate_xy_derivative_pt.argtypes = [c_float, c_float, POINTER(_Interpolator), c_int8, c_float, c_float]
interpolate_xy_derivative_pt.restype  =  c_float   

interpolate                           = interpolators_lib.interpolate
interpolate.argtypes                  = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8]
                                      
interpolate_x_derivative              = interpolators_lib.interpolate_x_derivative    
interpolate_x_derivative.argtypes     = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float]
                                      
interpolate_y_derivative              = interpolators_lib.interpolate_y_derivative
interpolate_y_derivative.argtypes     = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float]
                                      
interpolate_xy_derivative             = interpolators_lib.interpolate_xy_derivative  
interpolate_xy_derivative.argtypes    = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float, c_float]
                                      
interpolate2                          = interpolators_lib.interpolate2
interpolate2.argtypes                 = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8]
                                      
interpolate_x_derivative2             = interpolators_lib.interpolate_x_derivative2
interpolate_x_derivative2.argtypes    = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float]
                                      
interpolate_y_derivative2             = interpolators_lib.interpolate_y_derivative2
interpolate_y_derivative2.argtypes    = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float]
                                      
interpolate_xy_derivative2            = interpolators_lib.interpolate_xy_derivative2  
interpolate_y_derivative2.argtypes    = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float, c_float]


# TODO check: https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
class Array1D:
    def __init__(self, array: np.ndarray):
        if array.ndim != 1:
            raise RuntimeError()
        self.__array: _NumpyArray1D =  np_array_1d_new(array.size, array.data)

    def __del__(self):
        np_array_1d_del(self.__array)

    @property
    def size(self) -> int:
        return self.__array.contents.size

    @property
    def data(self):
        return self.__array.contents.data

    @property
    def array(self) -> _NumpyArray1D:
        return self.__array


# TODO check: https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
class Array2D:
    def __init__(self, array: np.ndarray):
        if array.ndim != 2:
            raise RuntimeError()
        self.__array: _NumpyArray2D =  np_array_2d_new(array.shape[0], array.shape[1], array.data)

    def __del__(self):
        np_array_2d_del(self.__array)

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
    def data(self):
        return self.__array.contents.data

    @property
    def array(self) -> _NumpyArray2D:
        return self.__array


class Interpolator:
    def __init__(self):
        self.__control_points: np.ndarray        = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]).data
        self.__control_points_ptr: _NumpyArray2D = _NumpyArray2D(3, 3, self.__control_points.data)
        self.__interpolator_ptr:   _Interpolator = _Interpolator(3, 3, self.__control_points_ptr)

    def __del__(self):
        np_array_2d_del(self.__control_points_ptr)
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
        np_array_2d_del(self.__control_points_ptr)
        rows, cols = self.__control_points_ptr.shape
        self.__control_points_ptr: _NumpyArray2D = _NumpyArray2D(rows, cols, self.__control_points.data)
        self.__interpolator_ptr.contents.control_points = self.__control_points_ptr
        self.__interpolator_ptr.contents.rows = rows
        self.__interpolator_ptr.contents.cols = cols

    def interpolate_pt(self, x: float, y: float, mode: int = 1) -> float:
        return interpolate_pt(x, y, self.__interpolator_ptr, mode)

    def interpolate_x_derivative_pt(self, x: float, y: float, dx: float = 1e-6, mode: int = 1) -> float:
        return interpolate_x_derivative_pt(x, y, self.__interpolator_ptr, mode, dx)

    def interpolate_y_derivative_pt(self, x: float, y: float, dy: float = 1e-6, mode: int = 1) -> float:
        return interpolate_y_derivative_pt(x, y, self.__interpolator_ptr, mode, dy)

    def interpolate_xy_derivative_pt(self, x: float, y: float, dx: float = 1e-6,
                                     dy: float = 1e-6, mode: int = 1) -> float:
        return interpolate_xy_derivative_pt(x, y, self.__interpolator_ptr, mode, dx, dy)

    def interpolate(self, x: np.ndarray, y: np.ndarray, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((min(x.size, y.size),), dtype=float)
        interpolate(Array1D(res), Array1D(x), Array1D(y), self.__interpolator_ptr, mode)
        return res

    def interpolate_x_derivative(self, x: np.ndarray, y: np.ndarray, dx: float = 1e-6, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((min(x.size, y.size),), dtype=float)
        interpolate_x_derivative(Array1D(res), Array1D(x), Array1D(y), self.__interpolator_ptr, mode, dx)
        return res

    def interpolate_y_derivative(self, x: np.ndarray, y: np.ndarray, dy: float = 1e-6, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((min(x.size, y.size),), dtype=float)
        interpolate_y_derivative(Array1D(res), Array1D(x), Array1D(y), self.__interpolator_ptr, mode, dy)
        return res

    def interpolate_xy_derivative(self, x:  np.ndarray, y:  np.ndarray, dx: float = 1e-6,
                                  dy: float = 1e-6, mode: int = 1) ->  np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((min(x.size, y.size),), dtype=float)
        interpolate_xy_derivative(Array1D(res), Array1D(x), Array1D(y), self.__interpolator_ptr, mode, dx,dy)
        return res

    def interpolate2(self, x: np.ndarray, y: np.ndarray, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((x.size, y.size,), dtype=float)
        interpolate2(Array1D(res), Array1D(x), Array1D(y), self.__interpolator_ptr, mode)
        return res

    def interpolate_x_derivative2(self, x: np.ndarray, y: np.ndarray, dx: float = 1e-6, mode: int = 1) -> np.ndarray:
        if x.ndim != 2:
            raise RuntimeError()
        if y.ndim != 2:
            raise RuntimeError()
        res = np.zeros((x.size, y.size,), dtype=float)
        interpolate_x_derivative2(Array1D(res), Array1D(x), Array1D(y), self.__interpolator_ptr, mode, dx)
        return res

    def interpolate_y_derivative2(self, x: np.ndarray, y: np.ndarray, dy: float = 1e-6, mode: int = 1) -> np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((x.size, y.size,), dtype=float)
        interpolate_y_derivative2(Array1D(res), Array1D(x), Array1D(y), self.__interpolator_ptr, mode, dy)
        return res

    def interpolate_xy_derivative2(self, x:  np.ndarray, y:  np.ndarray, dx: float = 1e-6,
                                   dy: float = 1e-6, mode: int = 1) ->  np.ndarray:
        if x.ndim != 1:
            raise RuntimeError()
        if y.ndim != 1:
            raise RuntimeError()
        res = np.zeros((x.size, y.size,), dtype=float)
        interpolate_xy_derivative2(Array1D(res), Array1D(x), Array1D(y), self.__interpolator_ptr, mode, dx,dy)
        return res