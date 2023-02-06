#include "pch.h"
#include "common.h"

std::ostream& operator<<(std::ostream& stream, const Color& arg)
{
	stream << "{ \"r\": " << I32(arg.r) << ", \"g\": " << I32(arg.g) << ", \"b\": " << I32(arg.b) << ", \"a\": " << I32(arg.a) << "}";
	return stream;
}
std::ostream& operator<<(std::ostream& stream, const IVec2& arg)
{
	stream << "{ \"x\": " << arg.x << ", \"y\": " << arg.y << "}";
	return stream;
}
std::ostream& operator<<(std::ostream& stream, const IVec3& arg)
{
	stream << "{ \"x\": " << arg.x << ", \"y\": " << arg.y << ", \"z\": " << arg.z << "}";
	return stream;
}
std::ostream& operator<<(std::ostream& stream, const Vec2 & arg)
{
	stream << "{ \"x\": " << arg.x << ", \"y\": " << arg.y << "}";
	return stream;
}
std::ostream& operator<<(std::ostream& stream, const Vec3 & arg)
{
	stream << "{ \"x\": " << arg.x << ", \"y\": " << arg.y << ", \"z\": " << arg.z << "}";
	return stream;
}
std::ostream& operator<<(std::ostream& stream, const Vec4 & arg)
{
	stream << "{ \"x\": " << arg.x << ", \"y\": " << arg.y << ", \"z\": " << arg.z << ", \"w\": " << arg.w << "}";
	return stream;
}
std::ostream& operator<<(std::ostream& stream, const Mat3 & arg)
{
	stream << "{\n";
	stream << "\t\"m00\": " << arg.m00 << ", \"m01\": " << arg.m01 << ", \"m02\": " << arg.m02 << ",\n";
	stream << "\t\"m10\": " << arg.m10 << ", \"m11\": " << arg.m11 << ", \"m12\": " << arg.m12 << ",\n";
	stream << "\t\"m20\": " << arg.m20 << ", \"m21\": " << arg.m21 << ", \"m22\": " << arg.m22 << "\n}";
	return stream;
}
std::ostream& operator<<(std::ostream& stream, const Mat4 & arg)
{
	stream << "{\n";
	stream << "\t\"m00\": " << arg.m00 << ", \"m01\": " << arg.m01 << ", \"m02\": " << arg.m02 << ", \"m03\": " << arg.m03 << ",\n";
	stream << "\t\"m10\": " << arg.m10 << ", \"m11\": " << arg.m11 << ", \"m12\": " << arg.m12 << ", \"m13\": " << arg.m13 << ",\n";
	stream << "\t\"m20\": " << arg.m20 << ", \"m21\": " << arg.m21 << ", \"m22\": " << arg.m22 << ", \"m23\": " << arg.m23 << ",\n";
	stream << "\t\"m30\": " << arg.m30 << ", \"m31\": " << arg.m31 << ", \"m32\": " << arg.m32 << ", \"m33\": " << arg.m33 << "\n}";
	return stream;
}