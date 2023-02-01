#include "pch.h"
#include "Transform.h"

float scl_x(const Mat3x3& transform)
{
	return sqrtf(transform.m00 * transform.m00 +
		     transform.m10 * transform.m10);
}
float scl_y(const Mat3x3& transform)
{
	return sqrtf(transform.m01 * transform.m01 +
		        transform.m11 * transform.m11);
}

float pos_x(const Mat3x3& transform)
{
	return transform.m02;
}
float pos_y(const Mat3x3& transform)
{
	return transform.m12;
}

float ang_z(const Mat3x3& transform)
{
	if (fabsf(transform.m00) > fabs(transform.m10))
	{
		return acosf(transform.m00 / scl_x(transform));
	}
	return acosf(transform.m10 / scl_x(transform));
}

Mat3x3 make_transform(const float& x0, const float& y0, const float& sx, const float& sy, const float& angle)
{
	float sin_a = sinf(angle);
	float cos_a = cosf(angle);
	return { cos_a * sx, -sin_a * sy,   x0,
			 sin_a * sx,  cos_a * sy,   y0,
			 0.0f,              0.0f,   1.0f };
}

void transform_point(float& xt, float& yt, const float& x, const float& y, const Mat3x3& transform)
{
	xt = transform.m00 * x + transform.m01 * y + transform.m02;
	yt = transform.m10 * x + transform.m11 * y + transform.m12;
}

void inv_transform_point(float& xt, float& yt, const float& x, const float& y, const Mat3x3& transform)
{
	xt = x - transform.m02;
	yt = y - transform.m12;

	float sx = scl_x(transform);
	float sy = scl_y(transform);

	sx = 1.0f / sx / sx;
	sy = 1.0f / sy / sy;

	xt = (transform.m00 * x + transform.m10 * y) * sx;
	yt = (transform.m01 * x + transform.m11 * y) * sy;
}

void decompose_transform(float& x0, float& y0, float& sx, float& sy, float& angle, const Mat3x3& transform)
{
	x0 = pos_x(transform);
	y0 = pos_y(transform);

	sx = scl_x(transform);
	sy = scl_y(transform);

	angle = ang_z(transform);
}

void transform_pt(float& xt, float& yt,
				  const float& x, const float& y,
				  const float& x0, const float& y0,
				  const float& sx, const float& sy,
				  const float& sin_a, const float& cos_a)
{
	xt = cos_a * sx * x - sin_a * sy * y + x0;
	yt = sin_a * sx * x + cos_a * sy * y + y0;
}

void inv_transform_pt(float& xt, float& yt,
	const float&  x,    const float&  y,
	const float& x0,    const float& y0,
	const float& sx,    const float& sy,
	const float& sin_a, const float& cos_a)
{
	xt = x - x0;
	yt = y - y0;
	float _sx = 1.0f / sx / sx;
	float _sy = 1.0f / sy / sy;
	xt = cos_a * _sx * x + sin_a * _sy * y;
	yt = -sin_a * _sx * x + cos_a * _sy * y;
}
