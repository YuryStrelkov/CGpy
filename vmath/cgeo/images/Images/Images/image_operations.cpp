#include "pch.h"
#include "image_operations.h"
#pragma once

UINT8 between(float x, float min, float max)
{
	if (x < min) return FALSE;
	if (x > max) return FALSE;
	return TRUE;
};

UINT32 red(UINT8 r)
{
	return (UINT32)r;
};

UINT32 green(UINT8 r)
{
	return red(r) << 8;
};

UINT32 blue(UINT8 r)
{
	return red(r) << 16;
};

UINT32 alpha(UINT8 r)
{
	return red(r) << 24;
};

UINT32 rgb(UINT8 r, UINT8 g, UINT8 b)
{
	return red(r) | green(g) | blue(b);
};

UINT32 rgba(UINT8 r, UINT8 g, UINT8 b, UINT8 a)
{
	return red(r) | green(g) | blue(b) | alpha(a);
};

UINT32 redf(float r)
{
	return (UINT32)r;
};

UINT32 greenf(float r)
{
	return redf(r) << 8;
};

UINT32 bluef(float r)
{
	return redf(r) << 16;
};

UINT32 alphaf(float r)
{
	return redf(r) << 24;
};

UINT32 rgbf(float r, float g, float b)
{
	return redf(r) | greenf(g) | bluef(b);
};

UINT32 rgbaf(float r, float g, float b, float a)
{
	return redf(r) | greenf(g) | bluef(b) | alphaf(a);
};

UINT8 uv_to_pix_local(float x, float y, UINT32 cols, UINT32 rows,
	                  UINT32& col, UINT32& row, UINT32& col1, UINT32& row1, float& tx, float& ty)
{
	if (!between(x, 0.0, 1.0))return FALSE;
	if (!between(y, 0.0, 1.0))return FALSE;

	col = (UINT32)(x * (cols - 1));
	row = (UINT32)(y * (rows - 1));

	col1 = min(col + 1, cols - 1);
	row1 = min(row + 1, rows - 1);

	float dx = 1.0 / (cols - 1.0);
	float dy = 1.0 / (rows - 1.0);

	tx = (x - dx * col) / dx;
	ty = (y - dy * row) / dy;

	return TRUE;
}

float x_derivative(UINT32 row, UINT32 col, UINT8* image, UINT32 rows, UINT32 cols, UINT8 bpp, UINT8 bpp_sift)
{
	UINT32 col_prew = max(col - 1, 0);
	UINT32 col_next = min(col + 1, cols - 1);
	return (image[(row * cols + col_next) * bpp + bpp_sift] -
		image[(row * cols + col_prew) * bpp + bpp_sift]) * 0.5;
};

float y_derivative(UINT32 row, UINT32 col, UINT8* image, UINT32 rows, UINT32 cols, UINT8 bpp, UINT8 bpp_sift)
{
	UINT32 row_prew = max(row - 1, 0);
	UINT32 row_next = min(row + 1, rows - 1);
	return (image[(row_next * cols + col) * bpp + bpp_sift] -
		image[(row_prew * cols + col) * bpp + bpp_sift]) * 0.5;
};

float xy_derivative(UINT32 row, UINT32 col, UINT8* image, UINT32 rows, UINT32 cols, UINT8 bpp, UINT8 bpp_sift)
{
	UINT32 row1 = min(row + 1, rows - 1);
	UINT32 row0 = max(0, row - 1);

	UINT32 col1 = min(col + 1, cols - 1);
	UINT32 col0 = max(0, col - 1);

	return (image[(row1 * cols + col1) * bpp + bpp_sift] -
		image[(row1 * cols + col0) * bpp + bpp_sift]) * 0.25 -
		(image[(row0 * cols + col1) * bpp + bpp_sift] -
			image[(row0 * cols + col0) * bpp + bpp_sift]) * 0.25;
}

float  cubic_poly(float x, float y, float* coefficients)
{
	float x2 = x * x;
	float x3 = x2 * x;
	float y2 = y * y;
	float y3 = y2 * y;
	return (coefficients[0] + coefficients[1] * y + coefficients[2] * y2 + coefficients[3] * y3) +
		(coefficients[4] + coefficients[5] * y + coefficients[6] * y2 + coefficients[7] * y3) * x +
		(coefficients[8] + coefficients[9] * y + coefficients[10] * y2 + coefficients[11] * y3) * x2 +
		(coefficients[12] + coefficients[13] * y + coefficients[14] * y2 + coefficients[15] * y3) * x3;
};

DLL_EXPORT UINT32 nearest32(float x, float y, Image* image)
{
	UINT32 col, row, col1, row1;
	float tx, ty;

	if (!uv_to_pix_local(x, y, image->rows, image->cols, col, row, col1, row1, tx, ty))return 0;

	row = ty < 0.5 ? row : row1;
	col = tx < 0.5 ? col : col1;

	col *= image->bpp;
	row *= image->bpp;

	if (image->bpp = 1)
	{
		return  rgb(image->data[col + row * image->cols], 0, 0);
	}
	if (image->bpp = 3)
	{
		return rgb(image->data[col + row * image->cols],
			image->data[col + row * image->cols + 1],
			image->data[col + row * image->cols + 2]);
	}
	if (image->bpp = 4)
	{
		return rgba(image->data[col + row * image->cols],
			image->data[col + row * image->cols + 1],
			image->data[col + row * image->cols + 2],
			image->data[col + row * image->cols + 3]);
	}
	return 0;
}

DLL_EXPORT UINT32 bilinear32(float x, float y, Image* image)
{
	UINT32 col, row, col1, row1, color = 0;
	float tx, ty;
	if (!uv_to_pix_local(x, y, image->rows, image->cols, col, row, col1, row1, tx, ty))return 0;

	float q00;
	float q01;
	float q10;
	float q11;

	for (UINT8 layer = 0; layer < image->bpp; layer++)
	{
		q00 = (float)(image->data[(col + row * image->cols) * image->bpp + layer]);
		q01 = (float)(image->data[(col1 + row * image->cols) * image->bpp + layer]);
		q10 = (float)(image->data[(col + row1 * image->cols) * image->bpp + layer]);
		q11 = (float)(image->data[(col1 + row1 * image->cols) * image->bpp + layer]);
		color |= redf(q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)) << (layer * 8);
	}

	return color;
}

DLL_EXPORT UINT32 bicubic32(float x, float y, Image* image)
{
	UINT32 col, row, col1, row1;

	float tx, ty;

	if (!uv_to_pix_local(x, y, image->rows, image->cols, col, row, col1, row1, tx, ty))return 0;

	float* b = new float[16];

	float* c = new float[16];

	UINT32 row, col, index, color = 0;

	for (UINT8 layer = 0; layer < image->bpp; layer++)
	{
		b[0] = image->data[(row * image->cols + col) * image->bpp + layer];
		b[4] = image->data[(row * image->cols + col1) * image->bpp + layer];
		b[8] = image->data[(row1 * image->cols + col) * image->bpp + layer];
		b[12] = image->data[(row1 * image->cols + col1) * image->bpp + layer];
		b[1] = x_derivative(row, col, image->data, image->rows, image->cols, image->bpp, layer);
		b[5] = x_derivative(row, col1, image->data, image->rows, image->cols, image->bpp, layer);
		b[9] = x_derivative(row1, col, image->data, image->rows, image->cols, image->bpp, layer);
		b[13] = x_derivative(row1, col1, image->data, image->rows, image->cols, image->bpp, layer);
		b[2] = y_derivative(row, col, image->data, image->rows, image->cols, image->bpp, layer);
		b[6] = y_derivative(row, col1, image->data, image->rows, image->cols, image->bpp, layer);
		b[10] = y_derivative(row1, col, image->data, image->rows, image->cols, image->bpp, layer);
		b[14] = y_derivative(row1, col1, image->data, image->rows, image->cols, image->bpp, layer);
		b[3] = y_derivative(row, col, image->data, image->rows, image->cols, image->bpp, layer);
		b[7] = xy_derivative(row, col1, image->data, image->rows, image->cols, image->bpp, layer);
		b[11] = xy_derivative(row1, col, image->data, image->rows, image->cols, image->bpp, layer);
		b[15] = xy_derivative(row1, col1, image->data, image->rows, image->cols, image->bpp, layer);
		for (index = 0; index < 16; index++)
		{
			c[index] = 0.0;
		}
		for (index = 0; index < 256; index++)
		{
			row = index / 16;
			col = index % 16;
			//c[row] += bicubic_poly_coefficients[row * 16 + col] * b[col];
		}
		color |= redf(cubic_poly(tx, ty, c)) << (layer * 8);
	}

	delete c;
	delete b;

	return color;
}

DLL_EXPORT void rescale(Image*src, Image* dst, UINT8 interpolation_method)
{
	UINT32 row, col, color;
	
	float x_col, y_row;
	
	interplator interp;

	if (interpolation_method == 0)
	{
		interp = &nearest32;
	}
	else if (interpolation_method == 1)
	{
		interp = &bilinear32;
	}
	else if (interpolation_method == 2)
	{
		interp = &bicubic32;
	}
	else
	{
		interp = &nearest32;
	}
	UINT32 depth, index;
	for (index = 0; index < dst->rows * dst->cols; index++)
	{
		col = index % dst->cols;
		row = index / dst->cols;
		x_col = col * 1.0 / dst->cols;
		y_row = row * 1.0 / dst->rows;
		color = interp(x_col, y_row, src);
		for (depth = 0; depth < dst->bpp; depth++)
		{
			dst->data[index * dst->bpp + depth] = (UINT8)(color | (255 << (8 * depth))) >> (8 * depth);
		}
	}
}