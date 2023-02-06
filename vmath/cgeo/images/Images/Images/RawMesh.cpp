#include "pch.h"
#include "RawMesh.h"


const Vec3 & RawMesh::GetPosition(const int i)const { return positions.at(i); }
const Vec2 & RawMesh::GetUv      (const int i)const { return uvs.at(i); }
const Vec3 & RawMesh::GetNormal  (const int i)const { return normals.at(i); }
const Vec4 & RawMesh::GetTangent (const int i)const { return tangents.at(i); }
const Color& RawMesh::GetColor   (const int i)const { return colors.at(i % colors.size()); }
const int  & RawMesh::GetIndex   (const int i)const { return indeces.at(i); }

const std::vector <Vec3> & RawMesh::GetPositions()const { return positions; }
const std::vector <Vec2> & RawMesh::GetUvs      ()const { return uvs; }
const std::vector <Vec3> & RawMesh::GetNormals  ()const { return normals; }
const std::vector <Vec4> & RawMesh::GetTangents ()const { return tangents; }
const std::vector <Color>& RawMesh::GetColors   () const{ return colors; }
const std::vector <int>  & RawMesh::GetIndeces  ()const { return indeces; }

void RawMesh::TransformMesh(const Mat4& transform)
{
	if (positions.size() != 0)
	{
		Vec3 v;

		Vec3 origin = Vec3(transform.m03, transform.m13, transform.m23);

		for (int i = 0; i < positions.size(); i++)
		{
			v = positions.at(i);// transform& positions.at(i);
			positions.at(i) = Vec3(v.x * transform.m00 + v.y * transform.m01 + v.z * transform.m02 + origin.x,
								   v.x * transform.m10 + v.y * transform.m11 + v.z * transform.m12 + origin.y,
								   v.x * transform.m20 + v.y * transform.m21 + v.z * transform.m22 + origin.z);
		}
	}
	if (normals.size() != 0)
	{
		Vec3 v;

		for (int i = 0; i < normals.size(); i++)
		{
			v = normals.at(i);// transform& positions.at(i);
			normals.at(i) = Vec3(v.x * transform.m00 + v.y * transform.m01 + v.z * transform.m02,
								 v.x * transform.m10 + v.y * transform.m11 + v.z * transform.m12,
								 v.x * transform.m20 + v.y * transform.m21 + v.z * transform.m22);
		}
	}
	
	if (tangents.size() != 0)
	{
		Vec4 v;

		for (int i = 0; i < tangents.size(); i++)
		{
			v = tangents.at(i);// transform& positions.at(i);
			tangents.at(i) = Vec4(v.x * transform.m00 + v.y * transform.m01 + v.z * transform.m02,
								  v.x * transform.m10 + v.y * transform.m11 + v.z * transform.m12,
								  v.x * transform.m20 + v.y * transform.m21 + v.z * transform.m22, 1.0f);
		}
	}
}

void RawMesh::SetPositions (const std::vector<Vec3>& v)
{
	SetPositions(v, 0);
}
void RawMesh::SetUVs       (const std::vector<Vec2>& v) 
{
	SetUVs(v, 0);
}
void RawMesh::SetNormals   (const std::vector<Vec3>& v) 
{
	SetNormals(v, 0);
}
void RawMesh::SetTangents  (const std::vector<Vec4>& v) 
{
	SetTangents(v, 0);
}
void RawMesh::SetColors    (const std::vector<Color>& v)
{
	SetColors(v, 0);
}
void RawMesh::SetIndeces   (const std::vector<int>& v) 
{
	SetIndeces(v, 0);
}

void RawMesh::SetPositions(const std::vector<Vec3>& v, const int offset)
{
	if (offset + v.size() > VerticesNumber())
	{
		positions.resize(offset + v.size(), Vec3());
	}
	for (int i = offset; i < offset + v.size(); i++)
	{
		positions.at(i) = v[i - offset];
	}
}
void RawMesh::SetUVs      (const std::vector<Vec2>& v, const int offset)
{
	if (offset + v.size() > VerticesNumber())
	{
		uvs.resize(offset + v.size(), Vec2());
	}
	for (int i = offset; i < offset + v.size(); i++)
	{
		uvs.at(i) = v[i - offset];
	}
}
void RawMesh::SetNormals  (const std::vector<Vec3>& v, const int offset)
{
	//normals.insert(normals.begin() + offset, &v[0], &v[count]);
	if (offset + v.size() > VerticesNumber())
	{
		normals.resize(offset + v.size(), Vec3());
	}
	for (int i = offset; i < offset + v.size(); i++)
	{
		normals.at(i) = v[i - offset];
	}
}
void RawMesh::SetTangents (const std::vector<Vec4>& v, const int offset)
{
	//tangents.insert(tangents.begin() + offset, &v[0], &v[count]);
	if (offset + v.size() > VerticesNumber())
	{
		tangents.resize(offset + v.size(), Vec4());
	}
	for (int i = offset; i < offset + v.size(); i++)
	{
		tangents.at(i) = v[i - offset];
	}
}
void RawMesh::SetColors   (const std::vector<Color>& v, const int offset)
{
	///colors.insert(colors.begin() + offset, &v[0], &v[count]);
	if (offset + v.size() > VerticesNumber())
	{
		colors.resize(offset + v.size(), Color());
	}
	for (int i = offset; i < offset + v.size(); i++)
	{
		colors.at(i) = v[i - offset];
	}
}
void RawMesh::SetIndeces  (const std::vector<int>& v, const int offset, const int indeces_shift)
{
	if (offset + v.size() > IndecesNumber())
	{
		indeces.resize(offset + v.size(), -1);
	}
	for (int i = offset; i < offset + v.size(); i++)
	{
		indeces.at(i) = v[i - offset] + indeces_shift;
	}
}

void RawMesh::SetPosition(const int id, const Vec3& v)
{
	if (id  >= positions.size())
	{
		positions.push_back(v);
		return;
	}
	positions.at(id) = v;
}
void RawMesh::SetUV      (const int id, const Vec2& v)
{
	if (id >= uvs.size())
	{
		uvs.push_back(v);
		return;
	}
	uvs.at(id) = v;
}
void RawMesh::SetNormal  (const int id, const Vec3& v)
{
	if (id >= normals.size())
	{
		normals.push_back(v);
		return;
	}
	normals.at(id) = v;
}
void RawMesh::SetTangent (const int id, const Vec4& v)
{
	if (id >= tangents.size())
	{
		tangents.push_back(v);
		return;
	}
	tangents.at(id) = v;
}
void RawMesh::SetColor   (const int id, const Color& v)
{
	if (id >= colors.size())
	{
		colors.push_back(v);
		return;
	}
	colors.at(id) = v;
}
void RawMesh::SetIndex   (const int id, const int i) 
{
	if (id >= indeces.size())
	{
		indeces.push_back(i);
		return;
	}
	indeces.at(id) = i;
}

void RawMesh::AddPosition(const Vec3& v)
{
	positions.push_back(v);
}
void RawMesh::AddUV      (const Vec2& v)
{
	uvs.push_back(v);
}
void RawMesh::AddNormal  (const Vec3& v)
{
	normals.push_back(v);
}
void RawMesh::AddTangent (const Vec4& v)
{
	tangents.push_back(v);
}
void RawMesh::AddColor   (const Color & v)
{
	colors.push_back(v);
}
void RawMesh::AddIndex   (const int i)
{
	indeces.push_back(i);
}

int RawMesh::VerticesNumber() const { return positions.size();}
int RawMesh::IndecesNumber () const { return indeces.size(); }
int RawMesh::AttribyteMask () const 
{
	int attribyteMask = 0;
	
	if (positions.size() != 0)
	{
		attribyteMask |= VERTICES_ATTRIBUTE;
	}
	
	if (normals.size() != 0)
	{
		attribyteMask |= NORMALS_ATTRIBUTE;
	}
	
	if (uvs.size() != 0)
	{
		attribyteMask |= UVS_0_ATTRIBUTE;
	}
	
	if (tangents.size() != 0)
	{
		attribyteMask |= TANGENTS_ATTRIBUTE;
	}
	
	if (colors.size() != 0)
	{
		attribyteMask |= VERTEX_COLOR_ATTRIBUTE;
	}

	if (indeces.size() != 0)
	{
		attribyteMask |= TRIANGLES_ATTRIBUTE;
	}
	return attribyteMask; 
}

void RawMesh::MergeMeshes(const RawMesh& other)
{
	if (AttribyteMask() != 0)
	{
		if (AttribyteMask() != other.AttribyteMask())
		{
			return;
		}
	}
	int start_indeces_offset  = IndecesNumber();
	int start_vertices_offset = VerticesNumber();

	if ((other.indeces.size() != 0))
	{
		SetIndeces(other.indeces, start_indeces_offset, start_vertices_offset);
	}
	if ((other.normals.size() != 0))
	{
		SetNormals(other.normals, start_vertices_offset);
	}
	if ((other.tangents.size() != 0))
	{
		SetTangents(other.tangents, start_vertices_offset);
	}
	if ((other.uvs.size() != 0))
	{
		SetUVs(other.uvs, start_vertices_offset);
	}
	if ((other.colors.size() != 0))
	{
		SetColors(other.colors, start_vertices_offset);
	}

	if ((other.positions.size() != 0))
	{
		SetPositions(other.positions, start_vertices_offset);
	}
}
RawMesh::RawMesh(const int n_v, const int n_i, const int attribyteMask)
{
 	if ((attribyteMask & VERTICES_ATTRIBUTE) == VERTICES_ATTRIBUTE)
	{
		positions.resize(n_v);// = new std::vector<Vec3>(verticesNumber);
	}
	if ((attribyteMask & TANGENTS_ATTRIBUTE) == TANGENTS_ATTRIBUTE)
	{
		tangents.resize(n_v);// = new std::vector<Vec4>(verticesNumber);
	}
	if ((attribyteMask & VERTEX_COLOR_ATTRIBUTE) == VERTEX_COLOR_ATTRIBUTE)
	{
		colors.resize(n_v);// = new std::vector<Vec3>(verticesNumber);
	}
	if ((attribyteMask & NORMALS_ATTRIBUTE) == NORMALS_ATTRIBUTE)
	{
		normals.resize(n_v);// = new std::vector<Vec3>(verticesNumber);
	}
	if ((attribyteMask & UVS_0_ATTRIBUTE) == UVS_0_ATTRIBUTE)
	{
		uvs.resize(n_v);// = new std::vector<Vec2>(verticesNumber);
	}
	if (((attribyteMask & TRIANGLES_ATTRIBUTE) == TRIANGLES_ATTRIBUTE) || n_i!=0)
	{
		indeces.resize(n_i);// = new std::vector<int>(indecesNumber);
	}
}
RawMesh::RawMesh()
{
}
RawMesh::~RawMesh() 
{
}

bool LoadObjMesh(const std::string& path,       std::vector<RawMesh*>& mehes)
{
	if (mehes.size() != 0)
	{
		RawMesh* ptr;
		for (int i = 0; i < mehes.size(); i++)
		{
			ptr = mehes[i];
			delete ptr;
		}
		mehes.clear();
	}

	std::ifstream in;

	in.open(path, std::ifstream::in);

	if (!in.is_open())
	{
		return false;
	}
	std::string line;

	RawMesh* mesh = nullptr;
	IVec3 iv3;
	Vec3 v3;
	Vec2 v2;

	char trash;

	while (!in.eof())
	{
		std::getline(in, line);

		std::istringstream iss(line.c_str());

		if (!line.compare(0, 2, "o ")) {
			mesh = new RawMesh();
			mehes.push_back(mesh);
			continue;
		}

		if (!line.compare(0, 2, "v ")) {
			iss >> trash;
			for (int i = 0; i < 3; i++) iss >> v3[i];
			mesh->AddPosition(v3);
			continue;
		}
		if (!line.compare(0, 3, "vn ")) {
			iss >> trash >> trash;
			for (int i = 0; i < 3; i++) iss >> v3[i];
			mesh->AddNormal(v3);
			continue;
		}
		if (!line.compare(0, 3, "vt ")) {
			iss >> trash >> trash;
			for (int i = 0; i < 2; i++) iss >> v2[i];
			mesh->AddUV(v2);
			continue;
		}
		if (!line.compare(0, 2, "f ")) {
			iss >> trash;
			while (iss >> iv3[0] >> trash >> iv3[1] >> trash >> iv3[2])
			{
				for (int i = 0; i < 3; i++)
				{
					iv3[i]--; // in wavefront obj all indices start at 1, not zero
					mesh->AddIndex(iv3[i]);
				}
			}
			continue;
		}
	}
	in.close();
	return true;
}
bool SaveObjMesh(const std::string& path, const std::vector<RawMesh*>& mesh);