#pragma once

#include "Model3D.h"

namespace AlgGeom
{
	class DrawMesh: public Model3D
	{
	protected:
		Assimp::Importer _assimpImporter;

	protected:
		Model3D::Component* processMesh(aiMesh* mesh, const aiScene* scene, const std::string& folder);
		void processNode(aiNode* node, const aiScene* scene, const std::string& folder);

	public:
		DrawMesh();
		virtual ~DrawMesh();

		DrawMesh* loadModelOBJ(const std::string& path);
	};
}

