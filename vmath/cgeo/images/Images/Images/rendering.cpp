#include "pch.h"
#include "rendering.h"

using namespace Rendering;
namespace Rendering
{
    std::ostream& operator<<(std::ostream& stream, const Vertex& arg)
    {
        stream << "{\n";
        stream << "\t\"position\": " << arg.position << ",\n";
        stream << "\t\"normal\"  : " << arg.normal << ",\n";
        stream << "\t\"uv\"      : " << arg.uv << ",\n";
        stream << "\t\"color\"   : " << arg.color << "\n}";
        return stream;
    }

    std::ostream& operator<<(std::ostream& stream, const ScreenVertex& arg)
    {
        stream << "{\n";
        stream << "\t\"position\": " << arg.position << ",\n";
        stream << "\t\"depth\"   : " << arg.depth << ",\n";
        stream << "\t\"normal\"  : " << arg.normal << ",\n";
        stream << "\t\"uv\"      : " << arg.uv << ",\n";
        stream << "\t\"color\"   : " << arg.color << "\n}";
        return stream;
    }

    std::ostream& operator<<(std::ostream& stream, const Triangle& arg)
    {
        stream << "{\n";
        stream << "\"v1\":\n" << arg.v1 << ",\n";
        stream << "\"v2\":\n" << arg.v2 << ",\n";
        stream << "\"v3\":\n" << arg.v3 << ",\n";
        stream << "\n}\n";
        return stream;
    }
}

RenderBuffer* Rendering::render_buffer_new         (Image* image)
{
    return new RenderBuffer(image);
}
RenderBuffer* Rendering::render_buffer_new_empty   (const I32& cols, const I32& rows, const I8& bpp)
{
    return new RenderBuffer(cols, rows, bpp);
}
void          Rendering::render_buffer_del         (RenderBuffer* render_buffer)
{
   if(render_buffer) delete render_buffer;
}
Vertex        Rendering::_lerp_vertex              (const Vertex& p1, const Vertex& p2, const F32& t)
{
    Vertex v;
    v.position = {
       p1.position.x + (p2.position.x - p1.position.x) * t,
       p1.position.y + (p2.position.y - p1.position.y) * t,
       p1.position.z + (p2.position.z - p1.position.z) * t
    };
    v.normal = {
       p1.normal.x + (p2.normal.x - p1.normal.x) * t,
       p1.normal.y + (p2.normal.y - p1.normal.y) * t,
       p1.normal.z + (p2.normal.z - p1.normal.z) * t
    };
    v.uv = {
       p1.uv.x + (p2.uv.x - p1.uv.x) * t,
       p1.uv.y + (p2.uv.y - p1.uv.y) * t
    };
    v.color = {
       (UI8)(p1.color.r + (p2.color.r - p1.color.r) * t),
       (UI8)(p1.color.g + (p2.color.g - p1.color.g) * t),
       (UI8)(p1.color.b + (p2.color.b - p1.color.b) * t),
       (UI8)(p1.color.a + (p2.color.a - p1.color.a) * t)
    };
    return v;
}
ScreenVertex  Rendering::_lerp_screen_vertex       (const ScreenVertex& p1, const ScreenVertex& p2, const F32& t)
{
    ScreenVertex v;
    v.position = {
       p1.position.x + (p2.position.x - p1.position.x) * t,
       p1.position.y + (p2.position.y - p1.position.y) * t,
    };

    v.depth = p1.depth + (p2.depth - p1.depth) * t;

    v.normal = {
       p1.normal.x + (p2.normal.x - p1.normal.x) * t,
       p1.normal.y + (p2.normal.y - p1.normal.y) * t,
       p1.normal.z + (p2.normal.z - p1.normal.z) * t
    };
    v.uv = {
       p1.uv.x + (p2.uv.x - p1.uv.x) * t,
       p1.uv.y + (p2.uv.y - p1.uv.y) * t
    };
    v.color = {
       (UI8)(p1.color.r + (p2.color.r - p1.color.r) * t),
       (UI8)(p1.color.g + (p2.color.g - p1.color.g) * t),
       (UI8)(p1.color.b + (p2.color.b - p1.color.b) * t),
       (UI8)(p1.color.a + (p2.color.a - p1.color.a) * t)
    };
    return v;
}
I8            Rendering::_position_code            (const float x, const float y)
{
    I8 code = 0;
    if (x < -1.0f) code |= LEFT;
    if (x > 1.0f)  code |= RIGHT;
    if (y < -1.0f) code |= BOTTOM;
    if (y > 1.0f)  code |= TOP;
    return code;
}
I8            Rendering::_vector_line_cast         (Vec2& p1,   Vec2& p2)
{
    I8 code1 = _position_code(p1.x, p1.y);
    I8 code2 = _position_code(p2.x, p2.y);
    if (code1 | code2 == 0) return   1; // <- all inside
    if (code1 & code2 != 0) return  -1; // <- all outside
    I8 code_out;
    F32 x, y;
    Vec2 p1_(p1.x, p1.y);
    Vec2 p2_(p2.x, p2.y);
    while (true)
    {
        if (code1 | code2 == 0) break; // <- vertices processed
        code_out = code1 != 0 ? code1 : code2;
        if (code_out & TOP)
        {
            x = p1.x + (p2.x - p1.x) * (1.0f - p1.y) / (p2.y - p1.y);
            y = 1.0f;
        }
        if (code_out & BOTTOM)
        {
            x = p1.x + (p2.x - p1.x) * (-1.0f - p1.y) / (p2.y - p1.y);
            y = -1.0f;
        }
        if (code_out & RIGHT)
        {
            y = p1.y + (p2.y - p1.y) * (1.0f - p1.x) / (p2.x - p1.x);
            x = 1.0f;
        }
        if (code_out & LEFT)
        {
            y = p1.y + (p2.y - p1.y) * (-1.0f - p1.x) / (p2.x - p1.x);
            x = -1.0f;
        }
        if (code_out == code1)
        {
            p1_.x = x;
            p1_.y = y;
            code1 = _position_code(x, y);
            continue;
        }
        p2_.x = x;
        p2_.y = y;
        code2 = _position_code(x, y);
    }
    p1 = p1_;
    p2 = p2_;
    return 0;
}
I8            Rendering::_vertex_line_cast         (Vertex& p1, Vertex& p2)
{
    I8 code1 = _position_code(p1.position.x, p1.position.y);
    I8 code2 = _position_code(p2.position.x, p2.position.y);
    if ((code1 | code2) == 0) return   1; // <- all inside
    if ((code1 & code2) != 0) return  -1; // <- all outside
    I8 code_out;
    F32 x, y, t;
    Vertex p1_(p1);
    Vertex p2_(p2);
    while (true)
    {
        if ((code1 | code2) == 0) break;
        code_out = code1 != 0 ? code1 : code2;
        if (code_out & TOP)
        {
            t = (1.0f - p1.position.y) / (p2.position.y - p1.position.y);
            x = p1.position.x + (p2.position.x - p1.position.x) * t;
            y = 1.0f;
        }
        if (code_out & BOTTOM)
        {
            t = (-1.0f - p1.position.y) / (p2.position.y - p1.position.y);
            x = p1.position.x + (p2.position.x - p1.position.x) * t;
            y = -1.0f;
        }
        if (code_out & RIGHT)
        {
            t = (1.0f - p1.position.x) / (p2.position.x - p1.position.x);
            y = p1.position.y + (p2.position.y - p1.position.y) * t;
            x = 1.0f;
        }
        if (code_out & LEFT)
        {
            t = (-1.0f - p1.position.x) / (p2.position.x - p1.position.x);
            y = p1.position.y + (p2.position.y - p1.position.y) * t;
            x = -1.0f;
        }
        if (code_out == code1)
        {
            p1_.position.x = x;
            p1_.position.y = y;
            p1_.position.z = p1.position.z + (p2.position.z - p1.position.z) * t;

            p1_.uv.x = p1.uv.x + (p2.uv.x - p1.uv.x) * t;
            p1_.uv.y = p1.uv.y + (p2.uv.y - p1.uv.y) * t;

            p1_.normal.x = p1.normal.x + (p2.normal.x - p1.normal.x) * t;
            p1_.normal.y = p1.normal.y + (p2.normal.y - p1.normal.y) * t;
            p1_.normal.z = p1.normal.z + (p2.normal.z - p1.normal.z) * t;

            p1_.color.r = (UI8)(p1.color.r + (p2.color.r - p1.color.r) * t);
            p1_.color.g = (UI8)(p1.color.g + (p2.color.g - p1.color.g) * t);
            p1_.color.b = (UI8)(p1.color.b + (p2.color.b - p1.color.b) * t);
            p1_.color.a = (UI8)(p1.color.a + (p2.color.a - p1.color.a) * t);

            code1 = _position_code(x, y);
            continue;
        }
        p2_.position.x = x;
        p2_.position.y = y;
        p2_.position.z = p1.position.z - (p2.position.z - p1.position.z) * t;

        p2_.uv.x = p1.uv.x - (p2.uv.x - p1.uv.x) * t;
        p2_.uv.y = p1.uv.y - (p2.uv.y - p1.uv.y) * t;

        p2_.normal.x = p1.normal.x - (p2.normal.x - p1.normal.x) * t;
        p2_.normal.y = p1.normal.y - (p2.normal.y - p1.normal.y) * t;
        p2_.normal.z = p1.normal.z - (p2.normal.z - p1.normal.z) * t;

        p1_.color.r = (UI8)(p1.color.r - (p2.color.r - p1.color.r) * t);
        p1_.color.g = (UI8)(p1.color.g - (p2.color.g - p1.color.g) * t);
        p1_.color.b = (UI8)(p1.color.b - (p2.color.b - p1.color.b) * t);
        p1_.color.a = (UI8)(p1.color.a - (p2.color.a - p1.color.a) * t);
        code2 = _position_code(x, y);
    }
    p1 = p1_;
    p2 = p2_;
    return 0; // <- vertices processed
}
void          Rendering::_cast_triangle            (const Triangle& src, Vertex* tris_casted, I8& tris_number)
{
    Triangle _processed = src;
    I8 flag;
    tris_number = 0;
    Vertex p1 = src.v1;
    Vertex p2 = src.v2;
    flag = _vertex_line_cast(p1, p2);
    if (flag == 1)
    {
        tris_casted[tris_number] = p1; tris_number++;
    }
    if (flag == 0)
    {
        tris_casted[tris_number] = p1; tris_number++;
        tris_casted[tris_number] = p2; tris_number++;
    }

    p1 = src.v2;
    p2 = src.v3;
    flag = _vertex_line_cast(p1, p2);
    if (flag == 1)
    {
        tris_casted[tris_number] = p1; tris_number++;
    }
    if (flag == 0)
    {
        tris_casted[tris_number] = p1; tris_number++;
        tris_casted[tris_number] = p2; tris_number++;
    }

    p1 = src.v3;
    p2 = src.v1;
    flag = _vertex_line_cast(p1, p2);
    if (flag == 1)
    {
        tris_casted[tris_number] = p1; tris_number++;
    }
    if (flag == 0)
    {
        tris_casted[tris_number] = p1; tris_number++;
        tris_casted[tris_number] = p2; tris_number++;
    }
}
void          Rendering::_transfrom_vertex         (const Mat4& transform, Vertex& vertex)
{
    vertex.position = Vec3(transform.m00 * vertex.position.x + transform.m01 * vertex.position.y + transform.m02 * vertex.position.z + transform.m02,
                           transform.m10 * vertex.position.x + transform.m11 * vertex.position.y + transform.m12 * vertex.position.z + transform.m12,
                           transform.m20 * vertex.position.x + transform.m21 * vertex.position.y + transform.m12 * vertex.position.z + transform.m12);
    vertex.normal = Vec3(transform.m00 * vertex.normal.x + transform.m01 * vertex.normal.y + transform.m02 * vertex.normal.z,
                         transform.m10 * vertex.normal.x + transform.m11 * vertex.normal.y + transform.m12 * vertex.normal.z,
                         transform.m20 * vertex.normal.x + transform.m21 * vertex.normal.y + transform.m12 * vertex.normal.z);
    F32 nl = sqrt(vertex.normal.x * vertex.normal.x + vertex.normal.y * vertex.normal.y + vertex.normal.z * vertex.normal.z);
    vertex.normal.x /= nl;
    vertex.normal.y /= nl;
    vertex.normal.z /= nl;
}
Vec2          Rendering::_pos_to_src               (const Image& screen, const Vec3& pos)
{
    return  Vec2((pos.x + 1.0f) * 0.5f * screen.cols(),  (pos.y + 1.0f) * 0.5f * screen.rows());
}
ScreenVertex  Rendering::_vert_to_screen           (const Image& screen, const Vertex& vertex)
{
    return { _pos_to_src(screen, vertex.position), vertex.position.z, vertex.normal, vertex.uv, vertex.color };
}
void          Rendering::_render_triangle          (RenderBuffer& renderBuffer, const Vertex& v_1, const Vertex& v_2, const Vertex& v_3)
{
    ScreenVertex t0 = _vert_to_screen(renderBuffer.image(), v_1);
    ScreenVertex t1 = _vert_to_screen(renderBuffer.image(), v_2);
    ScreenVertex t2 = _vert_to_screen(renderBuffer.image(), v_3);

    if(t0.position.y == t1.position.y && t0.position.y == t2.position.y) return; 
    if(t0.position.y > t1.position.y) std::swap(t0, t1);
    if(t0.position.y > t2.position.y) std::swap(t0, t2);
    if(t1.position.y > t2.position.y) std::swap(t1, t2);

    I32 total_height = round(t2.position.y - t0.position.y);
    I32 segment_height, index_row, index_col;
    F32 alpha;
    F32 beta;
    F32 phi;
    bool second_half;
    for (index_row = 0; index_row < total_height; index_row++)
    {
        second_half    = index_row   > t1.position.y - t0.position.y || t1.position.y == t0.position.y;
        segment_height = second_half ? round(t2.position.y - t1.position.y) : round(t1.position.y - t0.position.y);

        if (segment_height == 0)continue;

        alpha = F32(index_row) / total_height;

        beta  = F32(index_row - (second_half ? t1.position.y - t0.position.y : 0.0f)) / segment_height; // be careful: with above conditions no division by zero here

        ScreenVertex _A = _lerp_screen_vertex(t0, t2, alpha);
        ScreenVertex _B = second_half ? _lerp_screen_vertex(t1, t2, beta) : _lerp_screen_vertex(t0, t1, beta);

        if (_A.position.x > _B.position.x) std::swap(_A, _B);
      
        for (index_col = _A.position.x; index_col <= _B.position.x; index_col++)
        {
            F32 phi = _B.position.x == _A.position.x ? 1.0f : (index_col - _A.position.x) / (_B.position.x - _A.position.x);
            ScreenVertex P = _lerp_screen_vertex(_A, _B, phi);
            renderBuffer.set_color(round(P.position.y), round(P.position.x), P.depth,  P.color);
        }
    }
}
void          Rendering::_render_triangle_wireframe(RenderBuffer& renderBuffer, const Vertex& v_1, const Vertex& v_2, const Vertex& v_3)
{
    ScreenVertex t0 = _vert_to_screen(renderBuffer.image(), v_1);
    ScreenVertex t1 = _vert_to_screen(renderBuffer.image(), v_2);
    ScreenVertex t2 = _vert_to_screen(renderBuffer.image(), v_3);
    draw_line(renderBuffer, t0.position, t1.position, 0);
    draw_line(renderBuffer, t1.position, t2.position, 0);
    draw_line(renderBuffer, t2.position, t0.position, 0);
}
void          Rendering::draw_line                 (RenderBuffer& renderBuffer, const Vec2& _p0, const Vec2& _p1, const I32 color)
{
    IVec2 p0(round(_p0.x), round(_p0.y));
    IVec2 p1(round(_p1.x), round(_p1.y));
    bool steep = false;

    if (std::abs(p0.x - p1.x) < std::abs(p0.y - p1.y))
    {
        std::swap(p0.x, p0.y);
        std::swap(p1.x, p1.y);
        steep = true;
    }
    if (p0.x > p1.x) std::swap(p0, p1);
    for (int x = p0.x; x <= p1.x; x++) {
        float t = (x - p0.x) / (float)(p1.x - p0.x);
        int y = p0.y * (1. - t) + p1.y * t;
        if (steep)
        {
            renderBuffer.set_color(x, y, color);
            continue;
        }
        renderBuffer.set_color(y, x, color);
    }
    return;
}
void          Rendering::render_triangle           (RenderBuffer& renderBuffer, Triangle triangle)
{
    std::cout << triangle;
    if (triangle.v1.normal.z < 0 && triangle.v2.normal.z < 0 && triangle.v3.normal.z < 0)  return;
    Vertex _triangles[6];
    I8 tris_number;
    _cast_triangle(triangle, _triangles, tris_number);
    if (tris_number == 0)return;
    for (I8 index = 1; index < tris_number - 1; index += 1)
    {
        _render_triangle(renderBuffer, _triangles[0], _triangles[index], _triangles[index + 1]);
    }
}
void          Rendering::render_triangle_wireframe (RenderBuffer& renderBuffer, Triangle triangle)
{
    if (triangle.v1.normal.z < 0 && triangle.v2.normal.z < 0 && triangle.v3.normal.z < 0)  return;
    Vertex _triangles[6];
    I8 tris_number;
    _cast_triangle(triangle, _triangles, tris_number);
    if (tris_number == 0)return;
    for (I8 index = 1; index < tris_number - 1; index += 1)
    {
        _render_triangle_wireframe(renderBuffer, _triangles[0], _triangles[index], _triangles[index + 1]);
    }
}
void          Rendering::render_triangle           (RenderBuffer& renderBuffer, const Mat4& transform, Triangle triangle)
{
    _transfrom_vertex(transform, triangle.v1);
    _transfrom_vertex(transform, triangle.v2);
    _transfrom_vertex(transform, triangle.v3);
    render_triangle(renderBuffer, triangle);
}
void          Rendering::render_triangle_wireframe (RenderBuffer& renderBuffer, const Mat4& transform, Triangle triangle)
{
    _transfrom_vertex(transform, triangle.v1);
    _transfrom_vertex(transform, triangle.v2);
    _transfrom_vertex(transform, triangle.v3);
    render_triangle_wireframe(renderBuffer, triangle);
}
void          Rendering::render_mesh               (RenderBuffer& renderBuffer, const Mat4& transform, const RawMesh& mesh)
{
    for (I32 index = 0; index < mesh.IndecesNumber(); index += 3)
    {
        render_triangle(renderBuffer, transform, Triangle(Vertex(mesh.GetPosition(index),     mesh.GetNormal(index),     mesh.GetUv(index),     mesh.GetColor(index)),
                                                          Vertex(mesh.GetPosition(index + 1), mesh.GetNormal(index + 1), mesh.GetUv(index + 1), mesh.GetColor(index + 1)),
                                                          Vertex(mesh.GetPosition(index + 2), mesh.GetNormal(index + 2), mesh.GetUv(index + 2), mesh.GetColor(index + 2))));
    }
}
void          Rendering::render_mesh_wireframe     (RenderBuffer& renderBuffer, const Mat4& transform, const RawMesh& mesh)
{
    for (I32 index = 0; index < mesh.IndecesNumber(); index += 3)
    {
        render_triangle_wireframe(renderBuffer, transform, Triangle(Vertex(mesh.GetPosition(index),     mesh.GetNormal(index),     mesh.GetUv(index),     mesh.GetColor(index)),
                                                                    Vertex(mesh.GetPosition(index + 1), mesh.GetNormal(index + 1), mesh.GetUv(index + 1), mesh.GetColor(index + 1)),
                                                                    Vertex(mesh.GetPosition(index + 2), mesh.GetNormal(index + 2), mesh.GetUv(index + 2), mesh.GetColor(index + 2))));
    }
}

