#include "stdafx.h"
#include "SceneContent.h"

#include "ChronoUtilities.h"
#include "DrawMesh.h"
#include "../FileManagement.h"
#include "RandomUtilities.h"


// ----------------------------- BUILD YOUR SCENARIO HERE -----------------------------------

void AlgGeom::SceneContent::buildScenario()
{
    vec2 minBoundaries = vec2(-1.5, -.5), maxBoundaries = vec2(-minBoundaries);

    // Triangle mesh
    auto model = (new DrawMesh())->loadModelOBJ("Assets/Models/Sticks&Snow.obj");
    model->moveGeometryToOrigin(model->getModelMatrix(), 10.0f)->setModelMatrix(glm::translate(model->getModelMatrix(), vec3(.0f, .0f, .5f)));
	this->addNewModel(model);
	this->addNewModel(model->voxelize(uvec3(8), true)->overrideModelName());

#if TEST_MS
	for (int voxelDims = 8; voxelDims <= 1024; voxelDims *= 2)
	{
		printf("Testing dimensionality %i\n", voxelDims);

		std::vector<float> msBuffer;
		float ms;

		for (int testId = 0; testId < 10; ++testId)
		{
			model->voxelize(uvec3(voxelDims), ms, false);
			msBuffer.push_back(ms);
		}

		const std::string file = "performance/" + std::to_string(voxelDims) + ".txt";
		FileStorageUtilities::writeString(file, msBuffer);
	}
#endif
}


// ------------------------------------------------------------------------------------------


AlgGeom::SceneContent::SceneContent()
{
}

AlgGeom::SceneContent::~SceneContent()
{
	_camera.clear();
	_model.clear();
}

void AlgGeom::SceneContent::addNewCamera(ApplicationState* appState)
{
	_camera.push_back(std::unique_ptr<Camera>(new Camera(appState->_viewportSize.x, appState->_viewportSize.y)));
}

void AlgGeom::SceneContent::addNewModel(Model3D* model)
{	
	_sceneAABB.update(model->getAABB());
	_model.push_back(std::unique_ptr<Model3D>(model));
}

AlgGeom::Model3D* AlgGeom::SceneContent::getModel(Model3D::Component* component)
{
	for (auto& model : _model)
	{
		if (model->belongsModel(component))
			return model.get();
	}

	return nullptr;
}
