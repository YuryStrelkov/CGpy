#include "pch.h"
#include "Transform.h"

F32 scl_x(const Mat3& transform)
{
	return sqrtf(transform.m00 * transform.m00 +
		     transform.m10 * transform.m10);
}
F32 scl_y(const Mat3& transform)
{
	return sqrtf(transform.m01 * transform.m01 +
		        transform.m11 * transform.m11);
}

F32 pos_x(const Mat3& transform)
{
	return transform.m02;
}
F32 pos_y(const Mat3& transform)
{
	return transform.m12;
}

F32 ang_z(const Mat3& transform)
{
	if (fabsf(transform.m00) > fabs(transform.m10))
	{
		return acosf(transform.m00 / scl_x(transform));
	}
	return acosf(transform.m10 / scl_x(transform));
}

Mat3 make_transform(const F32& x0, const F32& y0, const F32& sx, const F32& sy, const F32& angle)
{
	F32 sin_a = sinf(angle);
	F32 cos_a = cosf(angle);
	return { cos_a * sx, -sin_a * sy,   x0,
			 sin_a * sx,  cos_a * sy,   y0,
			 0.0f,              0.0f,   1.0f };
}

void transform_point(F32& xt, F32& yt, const F32& x, const F32& y, const Mat3& transform)
{
	xt = transform.m00 * x + transform.m01 * y + transform.m02;
	yt = transform.m10 * x + transform.m11 * y + transform.m12;
}

void inv_transform_point(F32& xt, F32& yt, const F32& x, const F32& y, const Mat3& transform)
{
	xt = x - transform.m02;
	yt = y - transform.m12;

	F32 sx = scl_x(transform);
	F32 sy = scl_y(transform);

	sx = 1.0f / sx / sx;
	sy = 1.0f / sy / sy;

	xt = (transform.m00 * x + transform.m10 * y) * sx;
	yt = (transform.m01 * x + transform.m11 * y) * sy;
}

void decompose_transform(F32& x0, F32& y0, F32& sx, F32& sy, F32& angle, const Mat3& transform)
{
	x0 = pos_x(transform);
	y0 = pos_y(transform);

	sx = scl_x(transform);
	sy = scl_y(transform);

	angle = ang_z(transform);
}

void transform_pt(F32& xt, F32& yt,
				  const F32& x, const F32& y,
				  const F32& x0, const F32& y0,
				  const F32& sx, const F32& sy,
				  const F32& sin_a, const F32& cos_a)
{
	xt = cos_a * sx * x - sin_a * sy * y + x0;
	yt = sin_a * sx * x + cos_a * sy * y + y0;
}

void inv_transform_pt(F32& xt, F32& yt,
	const F32&  x,    const F32&  y,
	const F32& x0,    const F32& y0,
	const F32& sx,    const F32& sy,
	const F32& sin_a, const F32& cos_a)
{
	xt = x - x0;
	yt = y - y0;
	F32 _sx = 1.0f / sx / sx;
	F32 _sy = 1.0f / sy / sy;
	xt = cos_a * _sx * x + sin_a * _sy * y;
	yt = -sin_a * _sx * x + cos_a * _sy * y;
}
