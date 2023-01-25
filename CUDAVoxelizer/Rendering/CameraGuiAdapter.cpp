#include "stdafx.h"
#include "CameraGuiAdapter.h"

#include "GuiUtilities.h"

void AlgGeom::CameraGuiAdapter::renderGuiObject()
{
	bool updateMatrices = false;

	const char* projectionTitle[] = { "Perspective", "Orthographic" };

	updateMatrices |= ImGui::Combo("Camera Type", &_camera->_properties._cameraType, projectionTitle, IM_ARRAYSIZE(projectionTitle));

	GuiUtilities::leaveSpace(2);
	ImGui::Text("Current information");
	ImGui::Separator();
	GuiUtilities::leaveSpace(2);
	GuiUtilities::renderText(_camera->_properties._eye);

	GuiUtilities::leaveSpace(4);
	ImGui::Text("Camera");
	ImGui::Separator();
	GuiUtilities::leaveSpace(2);
	updateMatrices |= ImGui::InputFloat3("Eye", &_camera->_properties._eye[0]);
	updateMatrices |= ImGui::InputFloat3("Look at", &_camera->_properties._lookAt[0]);
	updateMatrices |= ImGui::InputFloat("Z near", &_camera->_properties._zNear);
	updateMatrices |= ImGui::InputFloat("Z far", &_camera->_properties._zFar);
	updateMatrices |= ImGui::InputFloat("FoV X", &_camera->_properties._fovX);
	updateMatrices |= ImGui::InputFloat("FoV Y", &_camera->_properties._fovY);
	if (_camera->_properties._cameraType == CameraProjection::ORTHOGRAPHIC)
	{
		updateMatrices |= ImGui::InputFloat2("Bottom Left Coordinates", &_camera->_properties._bottomLeftCorner[0]);
	}

	if (updateMatrices)
	{
		_camera->updateMatrices();
	}
}
