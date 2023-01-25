#pragma once

/**
	Copyright(C) 2023 Alfonso López Ruiz

	This program is free software : you can redistribute itand /or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program. If not, see <https://www.gnu.org/licenses/>.
**/

#include "CameraProjection.h"
#include "Model3D.h"

namespace AlgGeom
{
	class Camera
	{
		friend class CameraGuiAdapter;

	protected:
		Camera*								_backupCamera;	
		CameraProjection::CameraProperties	_properties;

	protected:
		void copyCameraAttributes(const Camera* camera);

	public:
		Camera(uint16_t width, uint16_t height, bool is2D = false);
		Camera(const Camera& camera);
		virtual ~Camera();
		void reset();
		void track(Model3D* model);

		Camera& operator=(const Camera& camera) = delete;

		mat4 getProjectionMatrix() { return _properties._projectionMatrix; }
		mat4 getViewMatrix() { return _properties._viewMatrix; }
		mat4 getViewProjectionMatrix() { return _properties._viewProjectionMatrix; }

		void saveCamera();
		void setBottomLeftCorner(const vec2& bottomLeft);
		void setCameraType(CameraProjection::Projection projection);
		void setFovX(float fovX);
		void setFovY(float fovY);
		void setLookAt(const vec3& position);
		void setPosition(const vec3& position);
		void setRaspect(uint16_t width, uint16_t height);
		void setUp(const vec3& up);
		void setZFar(float zfar);
		void setZNear(float znear);
		void updateMatrices();

		// Movements

		void boom(float speed);
		void crane(float speed);
		void dolly(float speed);
		void orbitXZ(float speed);
		void orbitY(float speed);
		void pan(float speed);
		void tilt(float speed);
		void truck(float speed);
		void zoom(float speed);
	};
}

