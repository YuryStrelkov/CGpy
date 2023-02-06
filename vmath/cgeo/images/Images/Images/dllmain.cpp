// dllmain.cpp : Определяет точку входа для приложения DLL.
#include "pch.h"
#include "image_operations.h"
#include "rendering.h"

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

int main() 
{
    F32 angle = -0.25f * PI;
    Mat3 transfrom = make_transform(0.0f, 0.0f, 2.0f, 2.0f, angle);
    Rendering::RenderBuffer renderBuffer(512, 512, 3);
    Rendering::Vertex v1(Vec3( 1.2f,  0.8f, 0.0f), Vec3(), Vec2(), Color(255,   0,   0, 0));
    Rendering::Vertex v3(Vec3( -1.2f,  0.0f, 0.0f), Vec3(), Vec2(), Color(  0, 255,   0, 0));
    Rendering::Vertex v2(Vec3( 0.0f,  -1.2f, 0.0f), Vec3(), Vec2(), Color(  0,   0, 255, 0));
    Rendering::Triangle tris(v1, v2, v3);
    Rendering::render_triangle(renderBuffer, tris);
    Rendering::render_triangle_wireframe(renderBuffer, tris);
    image_save(&renderBuffer.image(), "E:\\GitHub\\CGpy\\vmath\\cgeo\\images\\test_images\\iceland_transformed.png");

    std::cout << "1234";
}