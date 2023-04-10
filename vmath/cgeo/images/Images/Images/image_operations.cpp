#include "pch.h"
#include <math.h>
#include "image_operations.h"

CSTR            get_filename_ext(CSTR filename) {
	CSTR dot = strrchr(filename, '.');
	if (!dot || dot == filename) return "";
	return dot + 1;
}

DLL_EXPORT I8  chek_image_format(CSTR src)
{
	CSTR ext = get_filename_ext(src);
	if (strcmp(ext, "png"))return TRUE;
	if (strcmp(ext, "jpg"))return TRUE;
	if (strcmp(ext, "bmp"))return TRUE;
	if (strcmp(ext, "tga"))return TRUE;
	if (strcmp(ext, "hdr"))return TRUE;
	return FALSE;
}

DLL_EXPORT Image*      image_new(I32 w, I32 h, I8 bpp)
{
	return new Image(w, h, bpp);
}
				     
DLL_EXPORT void        image_del(Image* image)
{
	if (image == nullptr) return;
	delete image;
	image = nullptr;
}

DLL_EXPORT Image*     image_load(CSTR src)
{	
	CSTR ext = get_filename_ext(src);
	if (!chek_image_format(ext))return image_new(0, 0, 0);
	int w, h, bpp;
	if (!stbi_info(src, &w, &h, &bpp))return image_new(0, 0, 0);
	UI8* pixels = stbi_load(src, &w, &h, &bpp, 0);
	assert(pixels);
	Image* image = new Image(w, h, bpp, pixels);
	assert(image);
	return image;
}

DLL_EXPORT I8         image_save(const Image* image, CSTR dst)
{
	CSTR ext = get_filename_ext(dst);
	if (!chek_image_format(ext))
	{
		return FALSE;
	}
	if (strcmp(ext, "png"))
	{
		int stride_in_bytes = image->cols() * image->bpp();
		while (stride_in_bytes % 4 != 0)
		{
			stride_in_bytes++;
		}
		if (stbi_write_png(dst, image->cols(), image->rows(), image->bpp(), image->data(), stride_in_bytes))
		{
			throw FALSE;
		}
		return TRUE;
	}
	if (strcmp(ext, "jpg"))
	{
		if (!stbi_write_jpg(dst, image->cols(), image->rows(), image->bpp(), image->data(), 100))
		{
			throw FALSE;
		}
		return TRUE;
	}
	if (strcmp(ext, "bmp"))
	{
		if (!stbi_write_bmp(dst, image->cols(), image->rows(), image->bpp(), image->data()))
		{
			throw FALSE;
		}
		return TRUE;
	}
	if (strcmp(ext, "tga"))
	{
		if (!stbi_write_tga(dst, image->cols(), image->rows(), image->bpp(), image->data()))
		{
			throw FALSE;
		}
		return TRUE;
	}
	if (strcmp(ext, "hdr"))
	{
		I32 fdata_size = image->bytes_size();
		F32* fdata = (float*)malloc(fdata_size * sizeof(F32));
		I32 i;

#pragma omp parallel for
		for (i = 0; i < fdata_size; i++)
		{
			fdata[i] = ((F32)image->data()[i]) / 255.0f;
		}
		if (!stbi_write_hdr(dst, image->cols(), image->rows(), image->bpp(), fdata))
		{
			free(fdata);
			throw FALSE;
		}
		free(fdata);
		return TRUE;
	}
}

UI8 between(F32 x, F32 MIN, F32 MAX)
{
	if (x < MIN) return FALSE;
	if (x > MAX) return FALSE;
	return TRUE;
};

I32 red(UI8 r)
{
	return (I32)(255 & r);
};

I32 green(UI8 r)
{
	return red(r) << 8;
};

I32 blue(UI8 r)
{
	return red(r) << 16;
};

I32 alpha(UI8 r)
{
	return red(r) << 24;
};

I32 rgb(UI8 r, UI8 g, UI8 b)
{
	return red(r) | green(g) | blue(b);
};

I32 rgba(UI8 r, UI8 g, UI8 b, UI8 a)
{
	return red(r) | green(g) | blue(b) | alpha(a);
};

I32 redf(F32 r)
{
	return (I32)r;
};

I32 greenf(F32 r)
{
	return redf(r) << 8;
};

I32 bluef(F32 r)
{
	return redf(r) << 16;
};

I32 alphaf(F32 r)
{
	return redf(r) << 24;
};

I32 rgbf(F32 r, F32 g, F32 b)
{
	return redf(r) | greenf(g) | bluef(b);
};

I32 rgbaf(F32 r, F32 g, F32 b, F32 a)
{
	return redf(r) | greenf(g) | bluef(b) | alphaf(a);
};

UI8 channel      (I32 row, I32 col, const Image* image, UI8 layer)
{	
	return image->data()[(col + row * image->cols()) * image->bpp() + layer];
};
			     
I32 channelI32   (I32 row, I32 col, const Image* image, UI8 layer)
{
	return  I32(channel(row, col, image, layer));
};
			     
F32 channelF32   (I32 row, I32 col, const Image* image, UI8 layer)
{
	return  F32(channel(row, col, image, layer));
};
				 
F32 x_derivative (I32 row, I32 col, const Image* image, UI8 layer)
{
	I32 col_prew = MAX(col - 1, 0);
	I32 col_next = MIN(col + 1, image->cols() - 1);
	return (channelF32(row, col_next, image, layer) - channelF32(row, col_prew, image, layer)) * 0.5f;
};
				 
F32 y_derivative (I32 row, I32 col, const Image* image, UI8 layer)
{
	I32 row_prew = MAX(row - 1, 0);
	I32 row_next = MIN(row + 1, image->rows() - 1);
	return (channelF32(row_next, col, image, layer) - channelF32(row_prew, col, image, layer)) * 0.5f;
};

F32 xy_derivative(I32 row, I32 col, const Image* image, UI8 layer)
{
	I32 row1 = MIN(row + 1, image->rows() - 1);
	I32 row0 = MAX(0, row - 1);

	I32 col1 = MIN(col + 1, image->cols() - 1);
	I32 col0 = MAX(0, col - 1);

	return (channelF32(row1, col1, image, layer) - channelF32(row1, col0, image, layer)) * 0.25f -
		   (channelF32(row0, col1, image, layer) - channelF32(row0, col0, image, layer)) * 0.25f;
}

F32  cubic_poly(F32 x, F32 y, const F32* coefficients)
{
	F32 x2 = x * x;
	F32 x3 = x2 * x;
	F32 y2 = y * y;
	F32 y3 = y2 * y;
	return (coefficients[0]  + coefficients[1]  * y + coefficients[2]  * y2 + coefficients[3]  * y3) +
		   (coefficients[4]  + coefficients[5]  * y + coefficients[6]  * y2 + coefficients[7]  * y3) * x +
		   (coefficients[8]  + coefficients[9]  * y + coefficients[10] * y2 + coefficients[11] * y3) * x2 +
		   (coefficients[12] + coefficients[13] * y + coefficients[14] * y2 + coefficients[15] * y3) * x3;
};

DLL_EXPORT void image_clear_color(Image* image, I32 color) 
{
	I32 index, layer;

	for (index = 0; index < image->pixels_count(); index++)
	{
		for (layer = 0; layer < image->bpp(); layer++)
		{
			image->data()[index * image->bpp() + layer] = (color & (255 << layer * 8)) >> (8 * layer);
		}
	}
}

DLL_EXPORT I32 nearest32(F32 x, F32 y, const Image* image)
{
	if (!between(x, 0.0f, 1.0f))return 0;
	if (!between(y, 0.0f, 1.0f))return 0;
	
	I32 col, row, color = 0;
	
	F32 tx = x * (image->cols());
	F32 ty = y * (image->rows());

	col = min(I32(tx), image->cols() - 1);
	row = min(I32(ty), image->rows() - 1);

	for (I8 layer = 0; layer < image->bpp(); layer++)
	{
		color |= channel(row, col, image, layer) << (layer * 8);
	}
	return color;
};

DLL_EXPORT I32 bilinear32(F32 x, F32 y, const Image* image)
{
	if (!between(x, 0.0f, 1.0f))return 0;
	if (!between(y, 0.0f, 1.0f))return 0;

	I32 col, row, col1, row1, color = 0;
	F32 tx, ty;
	
	col = (I32)(x * (image->cols() - 1));
	row = (I32)(y * (image->rows() - 1));

	col1 = MIN(col + 1, image->cols() - 1);
	row1 = MIN(row + 1, image->rows() - 1);

	F32 dx = 1.0f / (image->cols() - 1);
	F32 dy = 1.0f / (image->rows() - 1);

	tx = MIN(1.0f, (x - col * dx) / dx);
	ty = MIN(1.0f, (y - row * dy) / dy);
	
	F32 q00, q01, q10, q11;

	for (I8 layer = 0; layer < image->bpp(); layer++)
	{
		q00 = channelF32(row, col, image, layer);
		q01 = channelF32(row, col1, image, layer);
		q10 = channelF32(row1, col, image, layer);
		q11 = channelF32(row1, col1, image, layer);
		color |= (I32(q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)) << (layer * 8));
	}
	return color;
}

DLL_EXPORT I32 bicubic32(F32 x, F32 y, const Image* image)
{
	if (!between(x, 0.0f, 1.0f))return 0;
	if (!between(y, 0.0f, 1.0f))return 0;

	I32 col, row, col1, row1, color = 0, index;
	F32 tx, ty;

	col = (I32)(x * (image->cols() - 1));
	row = (I32)(y * (image->rows() - 1));

	col1 = MIN(col + 1, image->cols() - 1);
	row1 = MIN(row + 1, image->rows() - 1);

	F32 dx = 1.0f / (image->cols() - 1);
	F32 dy = 1.0f / (image->rows() - 1);

	tx = MIN(1.0f, (x - col * dx) / dx);
	ty = MIN(1.0f, (y - row * dy) / dy);

	F32* b = (F32*)malloc(sizeof(F32) * 16);

	F32* c = (F32*)malloc(sizeof(F32) * 16);

	for (I8 layer = 0; layer < image->bpp(); layer++)
	{
		b[0 ] = channelF32(row,   col, image, layer);
		b[1 ] = channelF32(row,  col1, image, layer);
		b[2 ] = channelF32(row1,  col, image, layer);
		b[3 ] = channelF32(row1, col1, image, layer);
	
		b[4 ] = x_derivative(row,   col, image, layer);
		b[5]  = x_derivative(row,  col1, image, layer);
		b[6]  = x_derivative(row1,  col, image, layer);
		b[7]  = x_derivative(row1, col1, image, layer);
		
		b[8 ] = y_derivative (row,   col, image, layer);
		b[9 ] = y_derivative (row,  col1, image, layer);
		b[10] = y_derivative (row1,  col, image, layer);
		b[11] = y_derivative (row1, col1, image, layer);
		
		b[12] = xy_derivative(row,   col, image, layer);
		b[13] = xy_derivative(row,  col1, image, layer);
		b[14] = xy_derivative(row1,  col, image, layer);
		b[15] = xy_derivative(row1, col1, image, layer);

		for (index = 0; index < 16;  index++) c[index] = 0.0f;
		
		c[0]  =         b[0];
		c[1]  =         b[8];
		c[2]  = -3.0f * b[0] + 3.0f * b[2] - 2.0f * b[8] -        b[10];
		c[3]  =  2.0f * b[0] - 2.0f * b[2] +        b[8] +        b[10];
		c[4]  =         b[4];
		c[5]  =         b[12];
		c[6]  = -3.0f * b[4] + 3.0f * b[6] - 2.0f * b[12] -        b[14];
		c[7]  =  2.0f * b[4] - 2.0f * b[6] +        b[12] +        b[14];
		c[8]  = -3.0f * b[0] + 3.0f * b[1] - 2.0f * b[4]  -        b[5];
		c[9]  = -3.0f * b[8] + 3.0f * b[9] - 2.0f * b[12] -        b[13];
		c[10] =  9.0f * b[0] - 9.0f * b[1] - 9.0f * b[2]  + 9.0f * b[3]  + 6.0f * b[4]  + 3.0f * b[5]   - 6.0f * b[6] - 3.0f * b[7] +
			     6.0f * b[8] - 6.0f * b[9] + 3.0f * b[10] - 3.0f * b[11] + 4.0f * b[12] + 2.0f * b[13] + 2.0f * b[14] +        b[15];
		c[11] = -6.0f * b[0] + 6.0f * b[1] + 6.0f * b[2]  - 6.0f * b[3]  - 4.0f * b[4]  - 2.0f * b[5]  + 4.0f * b[6]  + 2.0f * b[7] - 
			     3.0f * b[8] + 3.0f * b[9] - 3.0f * b[10] + 3.0f * b[11] - 2.0f * b[12] -        b[13] - 2.0f * b[14] -        b[15];
		c[12] =  2.0f * b[0] - 2.0f * b[1] +        b[4]  +        b[5];
		c[13] =  2.0f * b[8] - 2.0f * b[9] +        b[12] +        b[13];
		c[14] = -6.0f * b[0] + 6.0f * b[1] + 6.0f * b[2]  - 6.0f * b[3]  - 3.0f * b[4]  - 3.0f * b[5]  + 3.0f * b[6]  + 3.0f * b[7] -
			     4.0f * b[8] + 4.0f * b[9] - 2.0f * b[10] + 2.0f * b[11] - 2.0f * b[12] - 2.0f * b[13] -        b[14] -        b[15];
		c[15] =  4.0f * b[0] - 4.0f * b[1] - 4.0f * b[2]  + 4.0f * b[3]  + 2.0f * b[4]  + 2.0f * b[5]  - 2.0f * b[6]  - 2.0f * b[7] +
			     2.0f * b[8] - 2.0f * b[9] + 2.0f * b[10] - 2.0f * b[11] +        b[12] +        b[13] +        b[14] +        b[15];

		color |= redf(MAX(MIN(cubic_poly(tx, ty, c), 255.0f), 0.0f)) << (layer * 8);
	}

	free(c);

	free(b);

	return color;
}

Interplator  resolve_inerp_function(UI8 interpolation_method)
{
	if (interpolation_method == 0)
	{
		return nearest32;
	}
	if (interpolation_method == 1)
	{
		return bilinear32;
	}
	if (interpolation_method == 2)
	{
		return bicubic32;
	}
	return nearest32;
}

void expand_bounds(const Mat3& tm, const Image& src, I32& width, I32& height)
{
	F32 x0 = -0.5f * src.cols();
	F32 x1 =  0.5f * src.cols();
	F32 y0 = -0.5f * src.rows();
	F32 y1 =  0.5f * src.rows();

	F32 x, y;
	F32 x_min =  1e12f, y_min =  1e12f;
	F32 x_max = -1e12f, y_max = -1e12f;

	transform_point(x, y, x0, y0, tm);
	x_min = MIN(x, x_min);
	x_max = MAX(x, x_max);
	y_min = MIN(y, y_min);
	y_max = MAX(y, y_max);

	transform_point(x, y, x1, y0, tm);
	x_min = MIN(x, x_min);
	x_max = MAX(x, x_max);
	y_min = MIN(y, y_min);
	y_max = MAX(y, y_max);

	transform_point(x, y, x0, y1, tm);
	x_min = MIN(x, x_min);
	x_max = MAX(x, x_max);
	y_min = MIN(y, y_min);
	y_max = MAX(y, y_max);

	transform_point(x, y, x1, y1, tm);
	x_min = MIN(x, x_min);
	x_max = MAX(x, x_max);
	y_min = MIN(y, y_min);
	y_max = MAX(y, y_max);
	
	width  = I32((x_max - x_min));
	height = I32((y_max - y_min));
}

DLL_EXPORT Image* transform(const F32* transform, const Image* src, const  UI8 interp_f, const UI8 expand)
{
	const Mat3* transform_m = (const Mat3*)transform;

	Image* dst;
	F32 src_scl_x = scl_x(*transform_m);
	F32 src_scl_y = scl_y(*transform_m);
	if (!expand)
	{
		dst = image_new(I32(src->cols() * src_scl_x), I32(src->rows() * src_scl_y), src->bpp());
	}
	else 
	{
		I32 w, h;
		expand_bounds(*transform_m, *src, w, h);
		dst = image_new(w, h, src->bpp());
	}

	F32 i_src_scl_x = 1.0f / src_scl_x / src_scl_x;
	F32 i_src_scl_y = 1.0f / src_scl_y / src_scl_y;
	F32 src_aspect  = (src->cols() * 1.0f) / src->rows();
	F32 scl_aspect  = src_scl_x / src_scl_y;
	F32 sx          = (dst->cols() * 1.0f) / src->cols();
	F32 sy          = (dst->rows() * 1.0f) / src->rows();

	F32	du = 1.0f / (dst->rows() - 1);
	F32 dv = 1.0f / (dst->cols() - 1);

	Interplator interpolator = resolve_inerp_function(interp_f);

#pragma omp parallel for
	for (I32 index = 0; index < dst->rows() * dst->cols(); index++)
	{
		F32 u = ((index / dst->cols()) * du * 2.0f - 1.0f) * sy;
		F32 v = ((index % dst->cols()) * dv * 2.0f - 1.0f) * sx * src_aspect;
		
		F32 ut = (transform_m->m00 * u * i_src_scl_x * scl_aspect + transform_m->m01 * v * i_src_scl_y              + transform_m->m02);
		F32 vt = (transform_m->m10 * u * i_src_scl_x              + transform_m->m11 * v * i_src_scl_y / scl_aspect + transform_m->m12);

		vt /= src_aspect;

 		I32 color = interpolator((vt + 1.0f) * 0.5f,  (ut + 1.0f) * 0.5f, src);

		for (I32 layer = 0; layer < dst->bpp(); layer++)
		{
			dst->data()[index * dst->bpp() + layer] = (color & (255 << layer * 8)) >> (8 * layer);
		}
	}
	return dst;
}

DLL_EXPORT Image* rescale(const F32* transform, const Image* src, const  UI8 interp_f, const UI8 expand)
{
	const Mat3* transform_m = (const Mat3*)transform;

	float s_x = scl_x(*transform_m);
	float s_y = scl_y(*transform_m);

	Image* dst = image_new(I32(src->cols() * s_x), I32(src->rows() * s_y), src->bpp());

	F32 aspect_1 = s_y / s_x;
	F32 aspect_2 = s_x / s_y;

	if (!expand)
	{
		s_x = 1.0f;
		s_y = 1.0f;
	}
	else 
	{
		if (aspect_1 > aspect_2)
		{
			s_x = 1.0f;
			s_y = aspect_1;
		}
		else 
		{
			s_x = aspect_2;
			s_y = 1.0f;
		}
	}

	Interplator interpolator = resolve_inerp_function(interp_f);

	F32 dv = 1.0f / (dst->cols() - 1),
		du = 1.0f / (dst->rows() - 1);

#pragma omp parallel for
	for (I32 index = 0; index < dst->pixels_count(); index++)
	{
		float vt = ((index % dst->cols()) * dv * 2.0f - 1.0f) * s_x;
		float ut = ((index / dst->cols()) * du * 2.0f - 1.0f) * s_y;

		I32 color = interpolator((vt + 1.0f) * 0.5f, (ut + 1.0f) * 0.5f, src);

		for (int layer = 0; layer < dst->bpp(); layer++)
		{
			dst->data()[index * dst->bpp() + layer] = (color & (255 << layer * 8)) >> (8 * layer);
		}
	}
	return dst;
};

DLL_EXPORT I32  get_uv(F32 u, F32 v, const Image* img, UI8 interp_f)
{
	return resolve_inerp_function(interp_f)(v, u, img);
}

DLL_EXPORT I32 get_pix(I32 row, I32 col, const Image* img)
{
	if (row < 0) return -1;
	if (col < 0) return -1;
	if (row >= img->rows()) return -1;
	if (col >= img->cols()) return -1;
	I32 color = 0, index = (row * img->cols() + col) * img->bpp();
	for (I8 layer = 0; layer < img->bpp(); layer++)
	{
		color |= img->data()[index + layer] << (layer * 8);
	}
	return color;
}

DLL_EXPORT I8  set_pix(I32 color, I32 row, I32 col, Image* img)
{
	if (row < 0) return FALSE;
	if (col < 0) return FALSE;
	if (row >= img->rows()) return FALSE;
	if (col >= img->cols()) return FALSE;
	I32 index = (row * img->cols() + col) * img->bpp();
	for (I8 layer = 0; layer < img->bpp(); layer++)
	{
		img->data()[index + layer] = (color & (255 << layer * 8)) >> (layer * 8);
	}
	return TRUE;
}
