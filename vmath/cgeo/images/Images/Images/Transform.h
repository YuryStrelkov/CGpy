#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include <math.h>

struct Mat3x3
{
	float m00, m01, m02;
	float m10, m11, m12;
	float m20, m21, m22;
};

float scl_x(const Mat3x3& transform);
float scl_y(const Mat3x3& transform);

float pos_x(const Mat3x3& transform);
float pos_y(const Mat3x3& transform);

float ang_z(const Mat3x3& transform);

Mat3x3 make_transform(const float& x0, const float& y0, const float& sx, const float& sy, float& angle);

void transform_point(float& xt, float& yt, const float& x, const float& y, const Mat3x3& transform);

void inv_transform_point(float& xt, float& yt, const float& x, const float& y, const Mat3x3& transform);

void decompose_transform(float& x0, float& y0, float& sx, float& sy, float& angle, const Mat3x3& transform);

void transform_pt(float&      xt,    float&      yt, 
	              const float&  x,    const float&  y,
	              const float& x0,    const float& y0,
	              const float& sx,    const float& sy, 
			      const float& sin_a, const float& cos_a);

void inv_transform_pt(float&      xt,    float&      yt, 
	                  const float&  x,    const float&  y,
	                  const float& x0,    const float& y0,
	                  const float& sx,    const float& sy, 
			          const float& sin_a, const float& cos_a);

#endif // __TRANSFORM_H__
