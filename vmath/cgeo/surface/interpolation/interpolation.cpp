#include "pch.h"
#include "interpolation.h"


DLL_EXPORT NumpyArray1D* np_array_1d_new(I32 size, F32* data_ptr)
{
	assert(data_ptr);
	NumpyArray1D* _array = (NumpyArray1D*)malloc(sizeof(NumpyArray1D));
	assert(_array);
	_array->data = data_ptr;
	_array->size = size;
	return _array;
};

DLL_EXPORT void np_array_1d_del(NumpyArray1D* _array)
{
	assert(_array);
	free(_array);
	_array = NULL;
};

DLL_EXPORT NumpyArray2D* np_array_2d_new(I32 rows, I32 cols, F32** data_ptr)
{
	assert(data_ptr);
	NumpyArray2D* _array = (NumpyArray2D*)malloc(sizeof(NumpyArray2D));
	assert(_array);
	_array->data = data_ptr;
	_array->rows = rows;
	_array->cols = cols;
	return _array;
};

DLL_EXPORT void np_array_2d_del(NumpyArray2D* _array)
{
	assert(_array);
	free(_array);
	_array = NULL;
};

DLL_EXPORT Interpolator* interpolator_new(I32 rows, I32 cols, F32** data_ptr)
{
	assert(data_ptr);
	Interpolator* _interpolator = (Interpolator*)malloc(sizeof(Interpolator));
	assert(_interpolator);
	_interpolator->control_points = np_array_2d_new(rows, cols, data_ptr);
	_interpolator->height = 1.0f;
	_interpolator->width = 1.0f;
	_interpolator->x0 = 0.0f;
	_interpolator->y0 = 0.0f;
	_interpolator->z0 = 0.0f;
	return _interpolator;
};

DLL_EXPORT void interpolator_del(Interpolator* _interpolator)
{
	assert(_interpolator);
	free(_interpolator);
	_interpolator = NULL;
};

DLL_EXPORT void          interpolator_del(NumpyArray2D* interpolator)
{
	if (interpolator == NULL) return;
	if (interpolator->data)
	{
		free(interpolator->data);
	}
	free(interpolator);
	interpolator = NULL;
}

UI8 between(F32 x, F32 min, F32 max) 
{
	if (x < min) return FALSE;
	if (x > max) return FALSE;
	return TRUE;
}
InterplatorF  resolve_inerp_function(UI8 interpolation_method) 
{
	if (interpolation_method == 0)
	{
		return nearest;
	}
	if (interpolation_method == 1)
	{
		return bilinear;
	}
	if (interpolation_method == 2)
	{
		return bicubic;
	}
	return nearest;
}
F32 cpoint       (I32 row, I32 col, const Interpolator* interpolator)
{
	return interpolator->control_points->data[row][col];
}
F32 cubic_poly   (F32 x, F32 y, const F32* coefficients)
{
	F32 x2 = x * x;
	F32 x3 = x2 * x;
	F32 y2 = y * y;
	F32 y3 = y2 * y;
	return (coefficients[0]  + coefficients[1]  * y + coefficients[2]  * y2 + coefficients[3]  * y3) +
		   (coefficients[4]  + coefficients[5]  * y + coefficients[6]  * y2 + coefficients[7]  * y3) * x +
		   (coefficients[8]  + coefficients[9]  * y + coefficients[10] * y2 + coefficients[11] * y3) * x2 +
		   (coefficients[12] + coefficients[13] * y + coefficients[14] * y2 + coefficients[15] * y3) * x3;
}
F32 x_derivative (I32 row, I32 col, const Interpolator* interpolator)
{
	I32 col_prew = MAX(col - 1, 0);
	I32 col_next = MIN(col + 1, COLS(interpolator)- 1);
	return (cpoint(row, col_next, interpolator) - cpoint(row, col_prew, interpolator)) * 0.5f;
}
F32 y_derivative (I32 row, I32 col, const Interpolator* interpolator)
{
	I32 row_prew = MAX(row - 1, 0);
	I32 row_next = MIN(row + 1, ROWS(interpolator) - 1);
	return (cpoint(row_next, col, interpolator) - cpoint(row_prew, col, interpolator)) * 0.5f;
}
F32 xy_derivative(I32 row, I32 col, const Interpolator* interpolator)
{
	I32 row1 = MIN(row + 1, ROWS(interpolator) - 1);
	I32 row0 = MAX(0, row - 1);

	I32 col1 = MIN(col + 1, COLS(interpolator) - 1);
	I32 col0 = MAX(0, col - 1);

	return (cpoint(row1, col1, interpolator) - cpoint(row1, col0, interpolator)) * 0.25f -
		   (cpoint(row0, col1, interpolator) - cpoint(row0, col0, interpolator)) * 0.25f;
}

void to_interpolator_space(F32 x, F32 y, const Interpolator* interpolator, F32* x_local, F32* y_local) 
{
	*x_local = X_TO_INTERPOLATOR_LOCAL(x, interpolator);
	*y_local = Y_TO_INTERPOLATOR_LOCAL(y, interpolator);
}

F32  nearest (F32 x, F32 y, const Interpolator* interpolator)
{
	F32 x_local, y_local;

	to_interpolator_space(x, y, interpolator, &x_local, &y_local);

	if (!between(x_local, 0.0f, 1.0f))return 0.0f;
	if (!between(y_local, 0.0f, 1.0f))return 0.0f;

	I32 col, row, color = 0;

	F32 tx = x * (COLS(interpolator));
	F32 ty = y * (ROWS(interpolator));

	col = min(I32(tx), COLS(interpolator) - 1);
	row = min(I32(ty), ROWS(interpolator) - 1);

	return cpoint(row, col, interpolator) + interpolator->z0;
};
F32  bilinear(F32 x, F32 y, const Interpolator* interpolator)
{
	F32 x_local, y_local;

	to_interpolator_space(x, y, interpolator, &x_local, &y_local);

	if (!between(x_local, 0.0f, 1.0f))return 0.0f;
	if (!between(y_local, 0.0f, 1.0f))return 0.0f;

	I32 col, row, col1, row1, color = 0;
	F32 tx, ty;

	col = (I32)(x * (COLS(interpolator) - 1));
	row = (I32)(y * (ROWS(interpolator) - 1));

	col1 = MIN(col + 1, COLS(interpolator) - 1);
	row1 = MIN(row + 1, ROWS(interpolator) - 1);

	F32 dx = 1.0f / (COLS(interpolator) - 1);
	F32 dy = 1.0f / (ROWS(interpolator) - 1);

	tx = MIN(1.0f, (x_local - col * dx) / dx);
	ty = MIN(1.0f, (y_local - row * dy) / dy);

	F32 q00 = cpoint(row,   col, interpolator);
	F32 q01 = cpoint(row,  col1, interpolator);
	F32 q10 = cpoint(row1,  col, interpolator);
	F32 q11 = cpoint(row1, col1, interpolator);
	return q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11) + interpolator->z0;
}
F32  bicubic (F32 x, F32 y, const Interpolator* interpolator)
{
	F32 x_local, y_local;

	to_interpolator_space(x, y, interpolator, &x_local, &y_local);

	if (!between(x_local, 0.0f, 1.0f))return 0.0f;
	if (!between(y_local, 0.0f, 1.0f))return 0.0f;

	I32 col, row, col1, row1, color = 0, index;
	F32 tx, ty;

	col = (I32)(x * (COLS(interpolator) - 1));
	row = (I32)(y * (ROWS(interpolator) - 1));

	col1 = MIN(col + 1, COLS(interpolator) - 1);
	row1 = MIN(row + 1, ROWS(interpolator) - 1);

	F32 dx = 1.0f / (COLS(interpolator) - 1);
	F32 dy = 1.0f / (ROWS(interpolator) - 1);

	tx = MIN(1.0f, (x_local - col * dx) / dx);
	ty = MIN(1.0f, (y_local - row * dy) / dy);

	F32* b = (F32*)malloc(sizeof(F32) * 16);
	F32* c = (F32*)malloc(sizeof(F32) * 16);
    
	assert(c);
	assert(b);

	b[0] = cpoint(row, col, interpolator);
	b[1] = cpoint(row, col1, interpolator);
	b[2] = cpoint(row1, col, interpolator);
	b[3] = cpoint(row1, col1, interpolator);

	b[4] = x_derivative(row, col, interpolator);
	b[5] = x_derivative(row, col1, interpolator);
	b[6] = x_derivative(row1, col, interpolator);
	b[7] = x_derivative(row1, col1, interpolator);

	b[8] = y_derivative(row, col, interpolator);
	b[9] = y_derivative(row, col1, interpolator);
	b[10] = y_derivative(row1, col, interpolator);
	b[11] = y_derivative(row1, col1, interpolator);

	b[12] = xy_derivative(row, col, interpolator);
	b[13] = xy_derivative(row, col1, interpolator);
	b[14] = xy_derivative(row1, col, interpolator);
	b[15] = xy_derivative(row1, col1, interpolator);

	for (index = 0; index < 16; index++) c[index] = 0.0f;

	c[0] = 1.0f * b[0];
	c[1] = 1.0f * b[8];
	c[2] = -3.0f * b[0] + 3.0f * b[2] - 2.0f * b[8] - 1.0f * b[10];
	c[3] = 2.0f * b[0] - 2.0f * b[2] + 1.0f * b[8] + 1.0f * b[10];
	c[4] = 1.0f * b[4];
	c[5] = 1.0f * b[12];
	c[6] = -3.0f * b[4] + 3.0f * b[6] - 2.0f * b[12] - 1.0f * b[14];
	c[7] = 2.0f * b[4] - 2.0f * b[6] + 1.0f * b[12] + 1.0f * b[14];
	c[8] = -3.0f * b[0] + 3.0f * b[1] - 2.0f * b[4] - 1.0f * b[5];
	c[9] = -3.0f * b[8] + 3.0f * b[9] - 2.0f * b[12] - 1.0f * b[13];
	c[10] = 9.0f * b[0] - 9.0f * b[1] - 9.0f * b[2] + 9.0f * b[3] + 6.0f * b[4] + 3.0f * b[5] - 6.0f * b[6] - 3.0f * b[7] +
		6.0f * b[8] - 6.0f * b[9] + 3.0f * b[10] - 3.0f * b[11] + 4.0f * b[12] + 2.0f * b[13] + 2.0f * b[14] + 1.0f * b[15];
	c[11] = -6.0f * b[0] + 6.0f * b[1] + 6.0f * b[2] - 6.0f * b[3] - 4.0f * b[4] - 2.0f * b[5] + 4.0f * b[6] + 2.0f * b[7] -
		3.0f * b[8] + 3.0f * b[9] - 3.0f * b[10] + 3.0f * b[11] - 2.0f * b[12] - 1.0f * b[13] - 2.0f * b[14] - 1.0f * b[15];
	c[12] = 2.0f * b[0] - 2.0f * b[1] + 1.0f * b[4] + 1.0f * b[5];
	c[13] = 2.0f * b[8] - 2.0f * b[9] + 1.0f * b[12] + 1.0f * b[13];
	c[14] = -6.0f * b[0] + 6.0f * b[1] + 6.0f * b[2] - 6.0f * b[3] - 3.0f * b[4] - 3.0f * b[5] + 3.0f * b[6] + 3.0f * b[7] -
		4.0f * b[8] + 4.0f * b[9] - 2.0f * b[10] + 2.0f * b[11] - 2.0f * b[12] - 2.0f * b[13] - 1.0f * b[14] - 1.0f * b[15];
	c[15] = 4.0f * b[0] - 4.0f * b[1] - 4.0f * b[2] + 4.0f * b[3] + 2.0f * b[4] + 2.0f * b[5] - 2.0f * b[6] - 2.0f * b[7] +
		2.0f * b[8] - 2.0f * b[9] + 2.0f * b[10] - 2.0f * b[11] + 1.0f * b[12] + 1.0f * b[13] + 1.0f * b[14] + 1.0f * b[15];
	
	F32 res = cubic_poly(tx, ty, c) + interpolator->z0;
	
	free(c);
	free(b);

	return res;

}

DLL_EXPORT F32  interpolate_pt              (F32 x, F32 y, const Interpolator* interpolator, const UI8 interp_f)
{
	return resolve_inerp_function(interp_f)(x, y, interpolator);
}
DLL_EXPORT F32  interpolate_x_derivative_pt (F32 x, F32 y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	return (interp_func(x + 0.5f * delta, y, interpolator) - interp_func(x - 0.5f * delta, y, interpolator)) / delta * 0.5f;
}
DLL_EXPORT F32  interpolate_y_derivative_pt (F32 x, F32 y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	return (interp_func(x, y + 0.5f * delta, interpolator) - interp_func(x, y - 0.5f * delta, interpolator)) / delta * 0.5f;
}
DLL_EXPORT F32  interpolate_xy_derivative_pt(F32 x, F32 y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta_x, const F32 delta_y)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	return (interp_func(x + 0.5f * delta_x, y + 0.5f * delta_y, interpolator) - interp_func(x + 0.5f * delta_x, y - 0.5f * delta_y, interpolator)) / delta_y * 0.25f - 
		   (interp_func(x - 0.5f * delta_x, y + 0.5f * delta_y, interpolator) - interp_func(x - 0.5f * delta_x, y - 0.5f * delta_y, interpolator)) / delta_y * 0.25f;
}

DLL_EXPORT void interpolate              (NumpyArray1D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	I32 n_points = MIN(MIN(result->size, x->size), y->size);
	for (I32 index = 0; index < n_points; index++)
	{
		result->data[index] = interp_func(x->data[index], y->data[index], interpolator);
	}
}
DLL_EXPORT void interpolate_x_derivative (NumpyArray1D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	I32 n_points = MIN(MIN(result->size, x->size), y->size);
	for (I32 index = 0; index < n_points; index++)
	{
		result->data[index] = (interp_func(x->data[index] + 0.5f * delta, y->data[index], interpolator) - 
							   interp_func(x->data[index] - 0.5f * delta, y->data[index], interpolator)) / delta * 0.5f;
	}
}
DLL_EXPORT void interpolate_y_derivative (NumpyArray1D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	I32 n_points = MIN(MIN(result->size, x->size), y->size);
	for (I32 index = 0; index < n_points; index++)
	{
		result->data[index] = (interp_func(x->data[index], y->data[index] + 0.5f * delta, interpolator) -
							   interp_func(x->data[index], y->data[index] - 0.5f * delta, interpolator)) / delta * 0.5f;
	}
}
DLL_EXPORT void interpolate_xy_derivative(NumpyArray1D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta_x, const F32 delta_y)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	I32 n_points = MIN(MIN(result->size, x->size), y->size);
	for (I32 index = 0; index < n_points; index++)
	{
		InterplatorF interp_func = resolve_inerp_function(interp_f);
		result->data[index] = (interp_func(x->data[index] + 0.5f * delta_x, y->data[index] + 0.5f * delta_y, interpolator) - 
							   interp_func(x->data[index] + 0.5f * delta_x, y->data[index] - 0.5f * delta_y, interpolator)) / delta_y * 0.25f -
						  	  (interp_func(x->data[index] - 0.5f * delta_x, y->data[index] + 0.5f * delta_y, interpolator) - 
							   interp_func(x->data[index] - 0.5f * delta_x, y->data[index] - 0.5f * delta_y, interpolator)) / delta_y * 0.25f;
	}
}

DLL_EXPORT void interpolate2              (NumpyArray2D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	I32 rows = MIN(y->size, result->rows);
	I32 cols = MIN(x->size, result->cols);
	I32 row, col;
	for (I32 index = 0; index <rows * cols; index++)
	{
		row = index / cols;
		col = index % cols;
		result->data[row][col] = interp_func(x->data[col], y->data[row], interpolator);
	}
}
DLL_EXPORT void interpolate_x_derivative2 (NumpyArray2D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	I32 rows = MIN(y->size, result->rows);
	I32 cols = MIN(x->size, result->cols);
	I32 row, col;
	for (I32 index = 0; index < rows * cols; index++)
	{
		row = index / cols;
		col = index % cols;
		result->data[row][col] = (interp_func(x->data[col] + 0.5f * delta, y->data[row], interpolator) -
							      interp_func(x->data[col] - 0.5f * delta, y->data[row], interpolator)) / delta * 0.5f;
	}
}
DLL_EXPORT void interpolate_y_derivative2 (NumpyArray2D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	I32 rows = MIN(y->size, result->rows);
	I32 cols = MIN(x->size, result->cols);
	I32 row, col;
	for (I32 index = 0; index < rows * cols; index++)
	{
		row = index / cols;
		col = index % cols;
		result->data[row][col] = (interp_func(x->data[col], y->data[row] + 0.5f * delta, interpolator) -
			                      interp_func(x->data[col], y->data[row] - 0.5f * delta, interpolator)) / delta * 0.5f;
	}
}
DLL_EXPORT void interpolate_xy_derivative2(NumpyArray2D* result, const NumpyArray1D* x, const NumpyArray1D* y, const Interpolator* interpolator, const UI8 interp_f, const F32 delta_x, const F32 delta_y)
{
	InterplatorF interp_func = resolve_inerp_function(interp_f);
	I32 rows = MIN(y->size, result->rows);
	I32 cols = MIN(x->size, result->cols);
	I32 row, col;
	for (I32 index = 0; index < rows * cols; index++)
	{
		row = index / cols;
		col = index % cols;
		result->data[row][col] = (interp_func(x->data[col] + 0.5f * delta_x, y->data[row] + 0.5f * delta_y, interpolator) -
								  interp_func(x->data[col] + 0.5f * delta_x, y->data[row] - 0.5f * delta_y, interpolator)) / delta_y * 0.25f -
								 (interp_func(x->data[col] - 0.5f * delta_x, y->data[row] + 0.5f * delta_y, interpolator) -
								  interp_func(x->data[col] - 0.5f * delta_x, y->data[row] - 0.5f * delta_y, interpolator)) / delta_y * 0.25f;
	}
}