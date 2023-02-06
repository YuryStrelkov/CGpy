#pragma once
#include "common.h"
#include "RawMesh.h"

namespace Rendering 
{
#define INSIDE 0
#define LEFT   1
#define RIGHT  2
#define BOTTOM 4
#define TOP    8

	struct Vertex
	{
		Vec3  position;
		Vec3  normal;
		Vec2  uv;
		Color color;
		Vertex() :position(0, 0, 0), normal(0, 0, 0), uv(0, 0), color(255, 0, 0, 0) {}
		Vertex(const Vec3& _position, const Vec3& _normal, const Vec2& _uv, const Color& _color) :position(_position), normal(_normal), uv(_uv), color(_color) {}
		Vertex(const Vertex& src) :position(src.position), normal(src.normal), uv(src.uv), color(src.color) {}
		friend std::ostream& operator<<(std::ostream& stream, const Vertex& arg);
	};

	struct ScreenVertex
	{
		Vec2  position;
		F32   depth;
		Vec3  normal;
		Vec2  uv;
		Color color;
		ScreenVertex() :position(0, 0), depth(1e12f), normal(0, 0, 0), uv(0, 0), color(255, 0, 0, 0) {}
		ScreenVertex(const Vec2& _position, const float& _depth, const Vec3& _normal, const Vec2& _uv, const Color& _color) :position(_position), depth(_depth), normal(_normal), uv(_uv), color(_color) {}
		ScreenVertex(const ScreenVertex& src) :position(src.position), depth(src.depth), normal(src.normal), uv(src.uv), color(src.color) {}
		friend std::ostream& operator<<(std::ostream& stream, const ScreenVertex& arg);
	};

	struct Triangle
	{
		Vertex v1;
		Vertex v2;
		Vertex v3;
		Triangle() :v1(), v2(), v3() {}
		Triangle(const Vertex& _v1, const Vertex& _v2, const Vertex& _v3) :v1(_v1), v2(_v2), v3(_v3) {}
		Triangle(const Triangle& triangle) :v1(triangle.v1), v2(triangle.v2), v3(triangle.v3) {}
		friend std::ostream& operator<<(std::ostream& stream, const Triangle& arg);
	};

	struct RenderBuffer
	{
	private:
		Image* _image;
		F32*   _depth;
		F32    _z_far;
		F32    _z_near;
		bool   _image_own = false;
		void         set_color(const I32& index, const F32& depth, const Color& c)
		{
			if (depth > _z_far)return;
			if (depth < _z_near)return;
			// if (index < 0)return;
			// if (index >= rows() * cols())return;
			if (_depth[index] < depth)return;
			_depth[index] = depth;
			image()(index, c);
		}
		void         set_color(const I32& index, const Color& c)
		{
			// if (index < 0)return;
			// if (index >= rows() * cols())return;
			image()(index, c);
		}

		bool validate_pix_coords(const I32& row, const I32& col)const 
		{
			if (row < 0)return false;
			if (col < 0)return false;
			if (row >= rows())return false;
			if (col >= cols())return false;
			return true;
		}

	public:
		const Image& image     ()const { return *_image; }
		      Image& image     ()      { return *_image; }
		const I32    rows      ()const { return _image->rows();}
		const I32    cols      ()const { return _image->cols();}
		const F32&   z_far     ()const { return _z_far;}
		const F32&   z_near    ()const { return _z_far;}
		F32*         depth     ()const { return _depth;}
		void         set_z_far (const F32& value)
		{
			_z_far = value;
			if (_z_far < _z_near) std::swap(_z_far, _z_near);
			for (I32 index = 0; index < _image->pixels_count(); index++)_depth[index] = _z_far;
		}
		void         set_z_near(const F32& value)
		{
			_z_near = value;
			if (_z_far < _z_near) std::swap(_z_far, _z_near);
			for (I32 index = 0; index < _image->pixels_count(); index++)_depth[index] = _z_far;
		}

		void         set_color (const I32& row, const I32& col, const F32& depth, const Color& c)
		{
			if (!validate_pix_coords(row, col)) return;
			set_color(col + row * cols(), depth, c);
		}
		void         set_color (const I32& row, const I32& col, const Color& c)
		{
			if (!validate_pix_coords(row, col)) return;
			set_color(col + row * cols(), c);
		}

		~RenderBuffer() 
		{
			if(_image && _image_own)delete  _image;
			if(_depth)delete[]_depth;
		}
		RenderBuffer(const I32& cols, const I32& rows, const I8& bpp)
		{
			_image  = new Image(cols, rows, bpp);
			assert(_image);
			_depth  = new float[cols * rows];
			assert(_depth);
			_z_far  = 1e12f;
			_z_near = -0.01f;
			_image_own = true;
			for (I32 index = 0; index < _image->pixels_count(); index++) _depth[index] = _z_far;
		}
		RenderBuffer(Image* image)
		{
			_image = image;
			assert(_image);
			_depth = new float[cols() * rows()];
			assert(_depth);
			_z_far = 1e12f;
			_z_near = -0.01f;
			_image_own = false;
			for (I32 index = 0; index < _image->pixels_count(); index++) _depth[index] = _z_far;
		}

	};

	RenderBuffer* render_buffer_new         (Image* image);
	RenderBuffer* render_buffer_new_empty   (const I32& cols, const I32& rows, const I8& bpp);
	void          render_buffer_del         (RenderBuffer* render_buffer);
	Vertex        _lerp_vertex              (const Vertex& p1, const Vertex& p2, const F32& t);
	ScreenVertex  _lerp_screen_vertex       (const ScreenVertex& p1, const ScreenVertex& p2, const F32& t);
	I8            _position_code            (const float x, const float y);
	I8            _vector_line_cast         (Vec2& p1, Vec2& p2);
	I8            _vertex_line_cast         (Vertex& p1, Vertex& p2);
	void          _cast_triangle            (const Triangle& src, Vertex* tris_casted, I8& tris_number);
	void          _transfrom_vertex         (const Mat4& transform, Vertex& vertex);
	Vec2          _pos_to_src               (const Image& screen, const Vec3& pos);
	ScreenVertex  _vert_to_screen           (const Image& screen, const Vertex& vertex);
	void          _render_triangle          (RenderBuffer& renderBuffer, const Vertex& v_1, const Vertex& v_2, const Vertex& v_3);
	void          _render_triangle_wireframe(RenderBuffer& renderBuffer, const Vertex& v_1, const Vertex& v_2, const Vertex& v_3);
	void          draw_line                 (RenderBuffer& renderBuffer, const Vec2& p0, const Vec2& p1, const I32 color);
	void          render_triangle           (RenderBuffer& renderBuffer, Triangle triangle);
	void          render_triangle_wireframe (RenderBuffer& renderBuffer, Triangle triangle);
	void          render_triangle           (RenderBuffer& renderBuffer, const Mat4& transform, Triangle triangle);
	void          render_triangle_wireframe (RenderBuffer& renderBuffer, const Mat4& transform, Triangle triangle);
	void          render_mesh               (RenderBuffer& renderBuffer, const Mat4& transform, const RawMesh& mesh);
	void          render_mesh_wireframe     (RenderBuffer& renderBuffer, const Mat4& transform, const RawMesh& mesh);
}


