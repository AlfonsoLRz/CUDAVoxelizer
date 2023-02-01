#pragma once

#include "AABB.h"
#include "ApplicationState.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "RenderingShader.h"
#include "Texture.h"
#include "TextureList.h"
#include "VAO.h"

#define BINARY_EXTENSION ".bin"
#define FBX_EXTENSION ".fbx"
#define GLTF_EXTENSION ".gltf"
#define OBJ_EXTENSION ".obj"

namespace AlgGeom
{
	class DrawVoxelization;

	class Model3D
	{
		friend class GUI;

	protected:
		struct Material
		{
			vec4		_kdColor;
			vec3		_ksColor;
			float		_metallic, _roughnessK;
			Texture*	_kadTexture;
			bool		_useUniformColor;
			vec3		_pointColor;
			vec3		_lineColor;

			Material() : _kdColor(1.00, 0.81, 0.29, 1.0f), _ksColor(.5f), _kadTexture(nullptr), _useUniformColor(true), _metallic(.7f), _roughnessK(.3f), _pointColor(.0f), _lineColor(.0f) {}
		};

	public:
		struct Component
		{
			bool							_enabled;
			std::string						_name;

			std::vector<VAO::Vertex>		_vertices;
			std::vector<GLuint>				_indices[VAO::NUM_IBOS];
			VAO*							_vao;
			AABB							_aabb;

			Material						_material;

			float							_lineWidth, _pointSize;
			bool							_activeRendering[VAO::NUM_IBOS];

			Component(VAO* vao = nullptr) { 
				_enabled = true; _vao = vao; _pointSize = 3.0f; _lineWidth = 1.0f; 			
				for (int i = 0; i < VAO::NUM_IBOS; ++i) _activeRendering[i] = true;
			}
			~Component() { delete _vao; _vao = nullptr; }

			void completeTopology();
			void generateWireframe();
			void generatePointCloud();
		};

	public:
		struct MatrixRenderInformation
		{
			enum MatrixType { MODEL, VIEW, VIEW_PROJECTION };
			
			mat4				_matrix[VIEW_PROJECTION + 1];
			std::vector<mat4>	_heapMatrices[VIEW_PROJECTION + 1];

			MatrixRenderInformation();
			mat4 multiplyMatrix(MatrixType tMatrix, const mat4& matrix) { this->saveMatrix(tMatrix); return _matrix[tMatrix] *= matrix; }
			void saveMatrix(MatrixType tMatrix) { _heapMatrices[tMatrix].push_back(_matrix[tMatrix]); }
			void setMatrix(MatrixType tMatrix, const mat4& matrix) { this->saveMatrix(tMatrix); _matrix[tMatrix] = matrix; }
			void undoMatrix(MatrixType type);
		};

	protected:
		static std::string							CHECKER_PATTERN_PATH;
		static std::unordered_set<std::string>		USED_NAMES;

	protected:
		AABB										_aabb;
		std::vector<std::unique_ptr<Component>>		_components;
		mat4										_modelMatrix;
		std::string									_name;
		DrawVoxelization*							_voxelization;

	protected:
		void buildVao(Component* component);
		void loadModelBinaryFile(const std::string& path);
		void writeBinaryFile(const std::string& path);
		void writeVoxelizationObj(const std::string& path, vec3* translationVectors, vec3 scale, size_t numVoxels);
		void writeVoxelizationPly(const std::string& path, vec3* translationVectors, vec3 scale, size_t numVoxels);

	public:
		Model3D();
		virtual ~Model3D();

		bool belongsModel(Component* component);
		virtual void draw(RenderingShader* shader, MatrixRenderInformation* matrixInformation, ApplicationState* appState, GLuint primitive);
		AABB getAABB(bool applyModelMatrix = true) { return applyModelMatrix ? _aabb.dot(_modelMatrix) : _aabb; }
		mat4 getModelMatrix() { return _modelMatrix; }
		std::string getName() { return _name; }
		Model3D* moveGeometryToOrigin(const mat4& origMatrix = mat4(1.0f), float maxScale = FLT_MAX);
		Model3D* overrideModelName();
		Model3D* setModelMatrix(const mat4& modelMatrix) { _modelMatrix = modelMatrix; return this; }
		Model3D* setLineColor(const vec3& color);
		Model3D* setPointColor(const vec3& color);
		Model3D* setTriangleColor(const vec4& color);
		Model3D* setTopologyVisibility(VAO::IBO_slots topology, bool visible);
		AlgGeom::DrawVoxelization* voxelize(const uvec3& voxelizationDimensions, bool createOpenGLStructures = true);
		AlgGeom::DrawVoxelization* voxelize(const uvec3& voxelizationDimensions, float& responseTime, bool createOpenGLStructures = true);
	};

	class DrawVoxelization : public Model3D
	{
	protected:
		size_t	_numVoxels;
		vec3	_voxelLength;

	public:
		static Component* getVoxel();

	public:
		DrawVoxelization();
		virtual ~DrawVoxelization();

		virtual void draw(RenderingShader* shader, MatrixRenderInformation* matrixInformation, ApplicationState* appState, GLuint primitive);
		vec3 getVoxelScale() { return _voxelLength; }
		DrawVoxelization* loadVoxelization(vec3* translation, size_t numVoxels, vec3 voxelScale);
	};
}

