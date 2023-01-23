#pragma once
#define TRUE 1
#define FALSE 0
#define min(a, b) a < b ? a : b
#define max(a, b) a > b ? a : b
#define DLL_EXPORT __declspec(dllexport)

typedef struct Image
{
	UINT8* data;
	UINT8  bpp;
	UINT32 rows;
	UINT32 cols;
};

Image* image_new(UINT32 rows, UINT32 cols, UINT8 bpp)
{

}

typedef UINT32(*interplator)(float, float, Image*);

UINT8 between(float x, float min, float max);

UINT32 red(UINT8 r);

UINT32 green(UINT8 r);

UINT32 blue(UINT8 r);

UINT32 alpha(UINT8 r);

UINT32 rgb(UINT8 r, UINT8 g, UINT8 b);

UINT32 rgba(UINT8 r, UINT8 g, UINT8 b, UINT8 a);

UINT32 redf(float r);

UINT32 greenf(float r);

UINT32 bluef(float r);

UINT32 alphaf(float r);

UINT32 rgbf(float r, float g, float b);

UINT32 rgbaf(float r, float g, float b, float a);

UINT8 uv_to_pix_local(float x, float y, UINT32 cols, UINT32 rows,
	                  UINT32& col, UINT32& row, UINT32& col1, UINT32& row1, float& tx, float& ty);

float cubic_poly(float x, float y, float* coefficients);

float x_derivative(UINT32 row, UINT32 col, UINT8* image, UINT32 rows, UINT32 cols, UINT8 bpp, UINT8 bpp_sift);

float y_derivative(UINT32 row, UINT32 col, UINT8* image, UINT32 rows, UINT32 cols, UINT8 bpp, UINT8 bpp_sift);

float xy_derivative(UINT32 row, UINT32 col, UINT8* image, UINT32 rows, UINT32 cols, UINT8 bpp, UINT8 bpp_sift);

DLL_EXPORT UINT32 nearest32 (float x, float y, Image* image);

DLL_EXPORT UINT32 bilinear32(float x, float y, Image* image);

DLL_EXPORT UINT32 bicubic32 (float x, float y, Image* image);

DLL_EXPORT void   rescale(Image* src, Image* dst, UINT8 interpolation_method);

DLL_EXPORT Image* crop   (Image* src, UINT32 x0, UINT32 y0, UINT32 x1, UINT32 y1);

DLL_EXPORT Image* crop_uv(Image* src, float x0, float y0, float x1, float y1, UINT8 interpolation_method);

DLL_EXPORT Image* rotate (Image* src, float x0, float y0, float angle, UINT8 interpolation_method);
