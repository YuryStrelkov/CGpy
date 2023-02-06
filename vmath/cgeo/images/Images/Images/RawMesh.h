#pragma once
#include "common.h"


typedef class RawMesh;

bool LoadObjMesh(const std::string& path,       std::vector<RawMesh*>& mesh);
bool SaveObjMesh(const std::string& path, const std::vector<RawMesh*>& mesh);

class RawMesh
{
	std::vector <Vec3>  positions;
	std::vector <Vec2>  uvs;
	std::vector <Vec3>  normals;
	std::vector <Vec4>  tangents;
	std::vector <Color> colors;
	std::vector <int>   indeces;
public:
	
	const Vec3 & GetPosition(const int i)const;
	const Vec2 & GetUv      (const int i)const;
	const Vec3 & GetNormal  (const int i)const;
	const Vec4 & GetTangent (const int i)const;
	const Color& GetColor   (const int i)const;
	const int  & GetIndex   (const int i)const;

	const std::vector <Vec3>&  GetPositions()const;
	const std::vector <Vec2>&  GetUvs()      const;
	const std::vector <Vec3>&  GetNormals()  const;
	const std::vector <Vec4>&  GetTangents() const;
	const std::vector <Color>& GetColors()   const;
	const std::vector <int>&   GetIndeces()  const;

	void TransformMesh(const Mat4 & transform);
	void SetPositions (const std::vector<Vec3>&  v);
	void SetUVs       (const std::vector<Vec2>&  v);
	void SetNormals   (const std::vector<Vec3>&  vt);
	void SetTangents  (const std::vector<Vec4>&  v);
	void SetColors    (const std::vector<Color>& v);
	void SetIndeces   (const std::vector<int>&   v);

	void SetPositions(const std::vector<Vec3>& v,  const int offset);
	void SetUVs      (const std::vector<Vec2>& v,  const int offset);
	void SetNormals  (const std::vector<Vec3>& v,  const int offset);
	void SetTangents (const std::vector<Vec4>& v,  const int offset);
	void SetColors   (const std::vector<Color>& v, const int offset);
	void SetIndeces  (const std::vector<int>&  v,  const int offset, const int indeces_shift = 0);

	void SetPosition(const int id, const Vec3& v);
	void SetUV      (const int id, const Vec2& v);
	void SetNormal  (const int id, const Vec3& v);
	void SetTangent (const int id, const Vec4& v);
	void SetColor   (const int id, const Color& v);
	void SetIndex   (const int id, const int i);

	void AddPosition(const Vec3& v);
	void AddUV      (const Vec2& v);
	void AddNormal  (const Vec3& v);
	void AddTangent (const Vec4& v);
	void AddColor   (const Color& v);
	void AddIndex   (const int i);

	void MergeMeshes(const RawMesh& other);

	// Mesh* ToMesh      ()const;
	int VerticesNumber()const;
	int IndecesNumber ()const;
	int AttribyteMask ()const;
	RawMesh           (const int n_v, const  int n_i, const  int attributeMask);
	RawMesh           ();
	~RawMesh          ();
};

