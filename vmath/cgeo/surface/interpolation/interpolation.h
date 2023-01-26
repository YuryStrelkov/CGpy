#ifndef __INTERPOLATION_H__
#define __INTERPOLATION_H__

#include <stdlib.h>
#include <cassert>
#include <stdlib.h>
#include <cassert>
#define TRUE  1
#define FALSE 0
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define DLL_EXPORT __declspec(dllexport)
#define I8  INT8
#define UI8 UINT8
#define I32 INT32
#define F32 float
#define STR char*
#define CSTR const char*
#define ARRF float*

#ifdef __cplusplus
extern "C"
{
#endif
	struct NumpyArray1D
	{
		F32* data;
		I32  size;
	};
	
	struct NumpyArray2D
	{
		F32** data;
		I32   rows;
		I32   cols;
	};

	struct Interpolator
	{
		NumpyArray2D* control_points;
		F32   width;
		F32   height;
		F32   x0;
		F32   y0;
		F32   z0;
	};

	DLL_EXPORT NumpyArray1D*  np_array_1d_new(I32 size, F32* data_ptr);
	DLL_EXPORT void           np_array_1d_del(NumpyArray1D* _array);
	DLL_EXPORT NumpyArray2D*  np_array_2d_new(I32 rows, I32 cols, F32** data_ptr);
	DLL_EXPORT void           np_array_2d_del(NumpyArray2D* _array);
	DLL_EXPORT Interpolator* interpolator_new(I32 rows, I32 cols, F32** data_ptr);
	DLL_EXPORT void          interpolator_del(Interpolator* _interpolator);

#define COLS(INTERPOLATOR) INTERPOLATOR->control_points->cols
#define ROWS(INTERPOLATOR) INTERPOLATOR->control_points->rows
#define WIDTH(INTERPOLATOR) INTERPOLATOR->width
#define HEIGHT(INTERPOLATOR) INTERPOLATOR->height
#define X_TO_INTERPOLATOR_LOCAL(X,INTERPOLATOR) ((X -INTERPOLATOR->x0) / INTERPOLATOR->width)
#define Y_TO_INTERPOLATOR_LOCAL(Y,INTERPOLATOR) ((Y -INTERPOLATOR->y0) / INTERPOLATOR->height)

	typedef F32(*InterplatorF)(F32, F32, const Interpolator*);
	UI8                               between(F32 x, F32 min, F32 max);
	InterplatorF       resolve_inerp_function(UI8 interpolation_method);
	F32 cpoint       (I32 row, I32 col, const Interpolator* interpolator);
	F32 cubic_poly   (F32 x,   F32 y, const F32* coefficients);
	F32 x_derivative (I32 row, I32 col, const Interpolator* interpolator);
	F32 y_derivative (I32 row, I32 col, const Interpolator* interpolator);
	F32 xy_derivative(I32 row, I32 col, const Interpolator* interpolator);
	void to_interpolator_space(F32 x, F32 y, const Interpolator* interpolator, F32* x_local, F32* y_local);

	F32  nearest                                (F32 x, F32 y, const Interpolator* interpolator);
	F32  bilinear                               (F32 x, F32 y, const Interpolator* interpolator);
	F32  bicubic                                (F32 x, F32 y, const Interpolator* interpolator);
	DLL_EXPORT F32  interpolate_pt              (F32 x, F32 y, const Interpolator* interpolator, const UI8 interp_f);
	DLL_EXPORT F32  interpolate_x_derivative_pt (F32 x, F32 y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta = 1e-6f);
	DLL_EXPORT F32  interpolate_y_derivative_pt (F32 x, F32 y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta = 1e-6f);
	DLL_EXPORT F32  interpolate_xy_derivative_pt(F32 x, F32 y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta_x = 1e-6f, const F32 delta_y = 1e-6f);
	DLL_EXPORT void interpolate                 (NumpyArray1D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f);
	DLL_EXPORT void interpolate_x_derivative    (NumpyArray1D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta = 1e-6f);
	DLL_EXPORT void interpolate_y_derivative    (NumpyArray1D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta = 1e-6f);
	DLL_EXPORT void interpolate_xy_derivative   (NumpyArray1D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta_x = 1e-6f, const F32 delta_y = 1e-6f);
	DLL_EXPORT void interpolate2                (NumpyArray2D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f);
	DLL_EXPORT void interpolate_x_derivative2   (NumpyArray2D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta = 1e-6f);
	DLL_EXPORT void interpolate_y_derivative2   (NumpyArray2D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta = 1e-6f);
	DLL_EXPORT void interpolate_xy_derivative2  (NumpyArray2D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta_x = 1e-6f, const F32 delta_y = 1e-6f);

#ifdef __cplusplus
}
#endif
#endif // __INTERPOLATION_H__