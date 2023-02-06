#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__
#include "common.h"
#include <math.h>

F32 scl_x(const Mat3& transform);
F32 scl_y(const Mat3& transform);

F32 pos_x(const Mat3& transform);
F32 pos_y(const Mat3& transform);

F32 ang_z(const Mat3& transform);

Mat3 make_transform     (const F32& x0, const F32& y0, const F32& sx, const F32& sy, const F32& angle);
					    
void transform_point    (F32& xt, F32& yt, const F32& x, const F32& y, const Mat3& transform);

void inv_transform_point(F32& xt, F32& yt, const F32& x, const F32& y, const Mat3& transform);

void decompose_transform(F32& x0, F32& y0, F32& sx, F32& sy, F32& angle, const Mat3& transform);

void transform_pt(F32&          xt, F32&          yt, 
	              const F32&     x, const F32&     y,
	              const F32&    x0, const F32&    y0,
	              const F32&    sx, const F32&    sy, 
			      const F32& sin_a, const F32& cos_a);

void inv_transform_pt(F32&          xt, F32&          yt, 
	                  const F32&     x, const F32&     y,
	                  const F32&    x0, const F32&    y0,
	                  const F32&    sx, const F32&    sy, 
			          const F32& sin_a, const F32& cos_a);

#endif // __TRANSFORM_H__
