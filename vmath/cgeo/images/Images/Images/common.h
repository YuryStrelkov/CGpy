#ifndef __COMMON_H__
#define __COMMON_H__
#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <list>

#define PI 3.1415926535897f
#define TRUE  1
#define FALSE 0
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLAMP(value, bmin, bmax) (MAX(MIN(value, bmax), 0))

#define DLL_EXPORT __declspec(dllexport)
#define I8  INT8
#define UI8 UINT8
#define I32 INT32
#define F32 float
#define STR char*
#define CSTR const char*

#define VERTICES_ATTRIBUTE     (1 << 0 )
#define NORMALS_ATTRIBUTE      (1 << 1 )
#define UVS_0_ATTRIBUTE        (1 << 2 )
#define TANGENTS_ATTRIBUTE     (1 << 3 )
#define BITANGENTS_ATTRIBUTE   (1 << 4 )
#define TRIANGLES_ATTRIBUTE    (1 << 5 )
#define UVS_1_ATTRIBUTE        (1 << 6 )
#define UVS_2_ATTRIBUTE        (1 << 7 )
#define UVS_3_ATTRIBUTE        (1 << 8 )
#define VERTEX_COLOR_ATTRIBUTE (1 << 9 )
#define MODEL_START_ATTRIBUTE  (1 << 10)
#define MODEL_END_ATTRIBUTE    (1 << 11)

struct Color
{
	union
	{
		union
		{
			struct
			{
				UI8 r;
				UI8 g;
				UI8 b;
				UI8 a;
			};
			I32 color;
		};
		UI8  channels[4];
	};

	const UI8& operator[](const int& index)const
	{
		assert(index < 0);
		assert(index >= 4);
		return channels[index];
	}

	UI8& operator[](const int& index)
	{
		assert(index < 0);
		assert(index >= 4);
		return channels[index];
	}

	Color() :r(255), g(255), b(255), a(255) {};

	Color(const UI8& r, const UI8& g, const UI8& b, const UI8& a)
	{
		this->r = r;
		this->g = g;
		this->b = b;
		this->a = a;
	}

	Color(const I32& rgba)
	{
		this->color = rgba;
	}

	Color(const Color& color)
	{
		this->r = color.r;
		this->g = color.g;
		this->b = color.b;
		this->a = color.a;
	}
	UI8* ptr()
	{
		return (UI8*)this;
	}
	friend std::ostream& operator<<(std::ostream& stream, const Color& arg);
};

struct Image
{
private:
	UI8* _data;
	UI8  _bpp ;
	I32  _rows;
	I32  _cols;
public:
	const UI8* data        ()const{return _data;}
	      UI8* data        ()     {return _data;}
	const UI8& bpp         ()const{return _bpp ;}
	const I32& rows        ()const{return _rows;}
	const I32& cols        ()const{return _cols;}
	const I32& bytes_size  ()const{return cols() * rows() * bpp(); }
	const I32& pixels_count()const{return cols() * rows(); }

	Color operator()(const I32& _index)const
	{
		int index = _index;
		Color c;
		if (bpp() == 1)
		{
			c.r = _data[index];
			return c;
		};
		if (bpp() == 3)
		{
			index *= bpp();
			c.r = _data[index];
			c.g = _data[index + 1];
			c.b = _data[index + 2];
			return c;
		};
		if (bpp() == 4)
		{
			index *= bpp();
			c.r = _data[index];
			c.g = _data[index + 1];
			c.b = _data[index + 2];
			c.a = _data[index + 3];
			return c;
		};
		return c;
	}

	void  operator()(const I32& _index, const Color& color)
	{
		int index = _index;
		if (bpp() == 1)
		{
			_data[index] = color.r;
			return;

		};
		if (bpp() == 3)
		{
			index *= bpp();
			_data[index] =     color.r;
			_data[index + 1] = color.g;
			_data[index + 2] = color.b;
			return;
		};
		if (bpp() == 4)
		{
			index *= bpp();
			_data[index]     = color.r;
			_data[index + 1] = color.g;
			_data[index + 2] = color.b;
			_data[index + 3] = color.a;
			return;
		};
	}

	Color operator()(const I32& row, const I32& col)const
	{
		return (*this)(row * _cols + col);
	}

	void  operator()(const I32& row, const I32& col, const Color& color)
	{
		(*this)(row * _cols + col, color);
	}

	Image(const  I32& cols, const I32& rows, const  UI8& bpp )
	{
		_data = rows * cols * bpp == 0 ? NULL : (UI8*)malloc(sizeof(UI8) * rows * cols * bpp);
		assert(_data);
		_bpp  = bpp ;
		_rows = rows;
		_cols = cols;
	}

	Image(const  I32& cols, const I32& rows, const  UI8& bpp, UI8* data)
	{
		assert(data);
		_data = data;
		_bpp = bpp;
		_rows = rows;
		_cols = cols;
	}

	~Image()
	{
		if (_data == NULL)return;
		free(_data);
		_data = 0;
	}
};

struct IVec2
{
	union
	{
		struct
		{
			I32 x;
			I32 y;
		};
		I32 data[2];
	};

	const I32& operator[](const int& index)const
	{
		assert(index < 0);
		assert(index >= 2);
		return data[index];
	}

	I32& operator[](const int& index)
	{
		assert(index < 0);
		assert(index >= 2);
		return data[index];
	}

	IVec2() :x(0), y(0){};

	IVec2(const I32& x, const I32& y)
	{
		this->x = x;
		this->y = y;
	}

	IVec2(const IVec2& src)
	{
		this->x = src.x;
		this->y = src.y;
	}
	F32* ptr()
	{
		return (F32*)this;
	}
	friend std::ostream& operator<<(std::ostream& stream, const IVec2& arg);
};

struct IVec3
{
	union
	{
		struct
		{
			I32 x;
			I32 y;
			I32 z;
		};
		I32 data[3];
	};

	const I32& operator[](const int& index)const
	{
		assert(index < 0);
		assert(index >= 3);
		return data[index];
	}

	I32& operator[](const int& index)
	{
		assert(index < 0);
		assert(index >= 3);
		return data[index];
	}

	IVec3() :x(0), y(0), z(0) {};

	IVec3(const I32& x, const I32& y, const I32& z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	IVec3(const IVec3& src)
	{
		this->x = src.x;
		this->y = src.y;
		this->z = src.z;
	}
	I32* ptr()
	{
		return (I32*)this;
	}
	friend std::ostream& operator<<(std::ostream& stream, const IVec3& arg);
};

struct Vec2
{
	union
	{
		struct
		{
			F32 x;
			F32 y;
		};
		F32 data[2];
	};

	const float& operator[](const int& index)const
	{
		assert(index < 0);
		assert(index >= 2);
		return data[index];
	}

	float& operator[](const int& index)
	{
		assert(index < 0);
		assert(index >= 2);
		return data[index];
	}

	Vec2() :x(0), y(0) {};

	Vec2(const  F32& x, const  F32& y)
	{
		this->x = x;
		this->y = y;
	}

	Vec2(const Vec2& src)
	{
		this->x = src.x;
		this->y = src.y;
	}

	F32* ptr()
	{
		return (F32*)this;
	}
	friend std::ostream& operator<<(std::ostream& stream, const Vec2& arg);
};

struct Vec3
{
	union
	{
		struct
		{
			F32 x;
			F32 y;
			F32 z;
		};
		F32 data[3];
	};

	const float& operator[](const int& index)const
	{
		assert(index < 0);
		assert(index >= 3);
		return data[index];
	}

	float& operator[](const int& index)
	{
		assert(index < 0);
		assert(index >= 3);
		return data[index];
	}

	Vec3() :x(0), y(0), z(0) {};

	Vec3(const F32& x, const F32& y, const F32& z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	Vec3(const Vec3& src)
	{
		this->x = src.x;
		this->y = src.y;
		this->z = src.z;
	}
	F32* ptr()
	{
		return (F32*)this;
	}
	friend std::ostream& operator<<(std::ostream& stream, const Vec3& arg);
};

struct Vec4
{
	union
	{
		struct
		{
			F32 x;
			F32 y;
			F32 z;
			F32 w;
		};
		F32 data[4];
	};

	const float& operator[](const int& index)const
	{
		assert(index < 0);
		assert(index >= 4);
		return data[index];
	}

	float& operator[](const int& index)
	{
		assert(index < 0);
		assert(index >= 4);
		return data[index];
	}

	Vec4() : x(0), y(0), z(0), w(0) {};

	Vec4(const F32& x, const F32& y, const F32& z, const F32& w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	Vec4(const Vec4& src)
	{
		this->x = src.x;
		this->y = src.y;
		this->z = src.z;
		this->w = src.w;
	}
	F32* ptr()
	{
		return (F32*)this;
	}
	friend std::ostream& operator<<(std::ostream& stream, const Vec4& arg);
};


struct Mat3
{
	union
	{
		struct
		{
			F32 m00; F32 m01; F32 m02;
			F32 m10; F32 m11; F32 m12;
			F32 m20; F32 m21; F32 m22;
		};
		F32 data[9];
	};

	const float& operator[](const int& index)const
	{
		assert(index < 0);
		assert(index >= 9);
		return data[index];
	}

	float& operator[](const int& index)
	{
		assert(index < 0);
		assert(index >= 9);
		return data[index];
	}

	Mat3() : m00(0), m01(0), m02(0),
		     m10(0), m11(0), m12(0),
		     m20(0), m21(0), m22(0)
		     {};

	Mat3(const F32& m00, const F32& m01, const F32& m02,
		 const F32& m10, const F32& m11, const F32& m12,
		 const F32& m20, const F32& m21, const F32& m22)
	{
		this->m00 = m00; this->m01 = m01; this->m02 = m02; 
		this->m10 = m10; this->m11 = m11; this->m12 = m12; 
		this->m20 = m20; this->m21 = m21; this->m22 = m22; 
	}

	Mat3(const Mat3& src)
	{
		this->m00 = src.m00; this->m01 = src.m01; this->m02 = src.m02;
		this->m10 = src.m10; this->m11 = src.m11; this->m12 = src.m12;
		this->m20 = src.m20; this->m21 = src.m21; this->m22 = src.m22;
	}
	F32* ptr()
	{
		return (F32*)this;
	}
	friend std::ostream& operator<<(std::ostream& stream, const Mat3& arg);
};

struct Mat4
{
	union
	{
		struct
		{
			F32 m00; F32 m01; F32 m02; F32 m03;
			F32 m10; F32 m11; F32 m12; F32 m13;
			F32 m20; F32 m21; F32 m22; F32 m23;
			F32 m30; F32 m31; F32 m32; F32 m33;
		};
		F32 data[16];
	};

	const float& operator[](const int& index)const
	{
		assert(index < 0);
		assert(index >= 16);
		return data[index];
	}

	float& operator[](const int& index)
	{
		assert(index < 0);
		assert(index >= 16);
		return data[index];
	}

	Mat4(): m00(0), m01(0), m02(0), m03(0),
			m10(0), m11(0), m12(0), m13(0),
			m20(0), m21(0), m22(0), m23(0),
			m30(0), m31(0), m32(0), m33(0){};

	Mat4(const F32& m00, const F32& m01, const F32& m02, const F32& m03,
		 const F32& m10, const F32& m11, const F32& m12, const F32& m13,
		 const F32& m20, const F32& m21, const F32& m22, const F32& m23,
		 const F32& m30, const F32& m31, const F32& m32, const F32& m33)
	{
		this->m00 = m00; this->m01 = m01; this->m02 = m02; this->m03 = m03;
		this->m10 = m10; this->m11 = m11; this->m12 = m12; this->m13 = m13;
		this->m20 = m20; this->m21 = m21; this->m22 = m22; this->m23 = m23;
		this->m30 = m30; this->m31 = m31; this->m32 = m32; this->m33 = m33;
	}

	Mat4(const Mat4& src)
	{
		this->m00 = src.m00; this->m01 = src.m01; this->m02 = src.m02; this->m03 = src.m03;
		this->m10 = src.m10; this->m11 = src.m11; this->m12 = src.m12; this->m13 = src.m13;
		this->m20 = src.m20; this->m21 = src.m21; this->m22 = src.m22; this->m23 = src.m23;
		this->m30 = src.m30; this->m31 = src.m31; this->m32 = src.m32; this->m33 = src.m33;
	}
	F32* ptr()
	{
		return (F32*)this;
	}
	friend std::ostream& operator<<(std::ostream& stream, const Mat4& arg);
};

#endif // __COMMON_H__
