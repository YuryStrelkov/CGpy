from ctypes import Structure, POINTER, c_int8, c_int32, CDLL, c_float
import numpy as np
import os

path = os.getcwd()
interpolators_lib = CDLL(path + "\interpolation.dll")


class _NumpyArray1D(Structure):
    _fields_ = ("data", POINTER(c_float)), \
               ("size", c_int32)


class _NumpyArray2D(Structure):
    _fields_ = ("data", POINTER(POINTER(c_float))), \
               ("rows", c_int32), \
               ("cols", c_int32)


class _Interpolator(Structure):
    _fields_ = ("control_points", POINTER(_NumpyArray2D)), \
               ("width", c_float), \
               ("height", c_float), \
               ("x0", c_float), \
               ("y0", c_float), \
               ("z0", c_float), \


NP_ARRAY_2_D_POINTER = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")
NP_ARRAY_1_D_POINTER = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")

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

interpolate                        = interpolators_lib.interpolate
interpolate_pt.argtypes            = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8]

interpolate_x_derivative           = interpolators_lib.interpolate_x_derivative    
interpolate_x_derivative.argtypes  = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float]

interpolate_y_derivative           = interpolators_lib.interpolate_y_derivative
interpolate_y_derivative.argtypes  = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float]
    
interpolate_xy_derivative          = interpolators_lib.interpolate_xy_derivative  
interpolate_xy_derivative.argtypes = [POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float, c_float]
 
interpolate2                       = interpolators_lib.interpolate2
interpolate2.argtypes              = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8]
                
interpolate_x_derivative2          = interpolators_lib.interpolate_x_derivative2
interpolate_x_derivative2.argtypes = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float]

interpolate_y_derivative2          = interpolators_lib.interpolate_y_derivative2
interpolate_y_derivative2.argtypes = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float]

interpolate_xy_derivative2         = interpolators_lib.interpolate_xy_derivative2  
interpolate_y_derivative2.argtypes = [POINTER(_NumpyArray2D), POINTER(_NumpyArray1D), POINTER(_NumpyArray1D), POINTER(_Interpolator), c_int8, c_float, c_float]


#TODO check: https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
def interpolate_point(x: float, y: float, points: np.ndarray, rows: int, cols: int, mode: int = 0) -> float:
    interp = _Interpolator()
    interp.contents.data = points.data
    interp.contents.rows = rows
    interp.contents.cols = cols
    return interpolate_point(x, y, interp, mode)
