#include "stdafx.h"
#include "Camera.h"

#include "CameraProjection.h"


// Public methods

AlgGeom::Camera::Camera(uint16_t width, uint16_t height, bool is2D) : _backupCamera(nullptr)
{
	this->_properties._cameraType = CameraProjection::PERSPECTIVE;
	this->_properties._2d = is2D;

	this->_properties._eye = vec3(0.0f, 3.0f, 10.0f);
	this->_properties._lookAt = vec3(0.0f, 3.0f, 0.0f);
	this->_properties._up = vec3(0.0f, 1.0f, 0.0f);

	this->_properties._zNear = 0.1f;
	this->_properties._zFar = 100.0f;

	this->_properties._width = width;
	this->_properties._height = height;
	this->_properties._aspect = this->_properties.computeAspect();

	this->_properties._bottomLeftCorner = vec2(-2.0f * this->_properties._aspect, -2.0f);
	this->_properties._fovX = 80.0f * glm::pi<float>() / 180.0f;
	this->_properties._fovY = this->_properties.computeFovY();

	this->_properties.computeAxes(this->_properties._n, this->_properties._u, this->_properties._v);
	this->_properties.computeViewMatrix();
	this->_properties.computeProjectionMatrices(&this->_properties);

	this->saveCamera();
}

AlgGeom::Camera::Camera(const AlgGeom::Camera& camera) : _backupCamera(nullptr)
{
	this->copyCameraAttributes(&camera);
}

AlgGeom::Camera::~Camera()
{
	delete _backupCamera;
}

void AlgGeom::Camera::reset()
{
	this->copyCameraAttributes(_backupCamera);
}

void AlgGeom::Camera::track(Model3D* model)
{
	AABB aabb = model->getAABB();

	this->setLookAt(aabb.center());
	this->setPosition(aabb.min() + vec3(.0f, aabb.extent().y, 1.0f) - vec3(aabb.extent().x, .0f, .0) * (1 + (1.0f / std::max(aabb.size().x, std::max(aabb.size().y, aabb.size().z))) * 4.0f));
}

void AlgGeom::Camera::saveCamera()
{
	delete _backupCamera;
	_backupCamera = nullptr;

	_backupCamera = new AlgGeom::Camera(*this);
}

void AlgGeom::Camera::setBottomLeftCorner(const vec2& bottomLeft)
{
	this->_properties._bottomLeftCorner = bottomLeft;
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void AlgGeom::Camera::setCameraType(const AlgGeom::CameraProjection::Projection projection)
{
	this->_properties._cameraType = projection;
	this->_properties.computeViewMatrices();
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void AlgGeom::Camera::setFovX(const float fovX)
{
	this->_properties._fovX = fovX;
	this->_properties._fovY = this->_properties.computeFovY();
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void AlgGeom::Camera::setFovY(const float fovY)
{
	this->_properties._fovY = fovY;
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void AlgGeom::Camera::setLookAt(const vec3& position)
{
	this->_properties._lookAt = position;
	this->_properties.computeAxes(this->_properties._n, this->_properties._u, this->_properties._v);
	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::setPosition(const vec3& position)
{
	this->_properties._eye = position;
	this->_properties.computeAxes(this->_properties._n, this->_properties._u, this->_properties._v);
	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::setRaspect(const uint16_t width, const uint16_t height)
{
	this->_properties._width = width;
	this->_properties._height = height;
	this->_properties._aspect = this->_properties.computeAspect();
	this->_properties._bottomLeftCorner = vec2(this->_properties._bottomLeftCorner.y * this->_properties._aspect, this->_properties._bottomLeftCorner.y);
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void AlgGeom::Camera::setUp(const vec3& up)
{
	this->_properties._up = up;
	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::setZFar(const float zfar)
{
	this->_properties._zFar = zfar;
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void AlgGeom::Camera::setZNear(const float znear)
{
	this->_properties._zNear = znear;
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void AlgGeom::Camera::updateMatrices()
{
	this->_properties.computeViewMatrix();
	this->_properties.computeProjectionMatrices(&_properties);
}

// [Movements] 

void AlgGeom::Camera::boom(float speed)
{
	const glm::mat4 translationMatrix = glm::translate(mat4(1.0f), this->_properties._v * speed);			// Translation in y axis

	this->_properties._eye = vec3(translationMatrix * vec4(this->_properties._eye, 1.0f));
	this->_properties._lookAt = vec3(translationMatrix * vec4(this->_properties._lookAt, 1.0f));

	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::crane(float speed)
{
	boom(-speed);					// Implemented as another method to take advantage of nomenclature
}

void AlgGeom::Camera::dolly(float speed)
{
	if (this->_properties._2d) return;

	const mat4 translationMatrix = glm::translate(mat4(1.0f), -this->_properties._n * speed);			// Translation in z axis
	this->_properties._eye = vec3(translationMatrix * vec4(this->_properties._eye, 1.0f));
	this->_properties._lookAt = vec3(translationMatrix * vec4(this->_properties._lookAt, 1.0f));

	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::orbitXZ(float speed)
{
	if (this->_properties._2d) return;

	const mat4 rotationMatrix = glm::rotate(mat4(1.0f), speed, this->_properties._u);					// We will pass over the scene, x or z axis could be used

	this->_properties._eye = vec3(rotationMatrix * vec4(this->_properties._eye - this->_properties._lookAt, 1.0f)) + this->_properties._lookAt;
	this->_properties._u = vec3(rotationMatrix * vec4(this->_properties._u, 0.0f));
	this->_properties._v = vec3(rotationMatrix * vec4(this->_properties._v, 0.0f));
	this->_properties._n = vec3(rotationMatrix * vec4(this->_properties._n, 0.0f));
	this->_properties._up = glm::normalize(glm::cross(this->_properties._n, this->_properties._u));						// Free rotation => we can look down or up

	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::orbitY(float speed)
{
	if (this->_properties._2d) return;

	const mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), speed, glm::vec3(0.0, 1.0f, 0.0f));

	this->_properties._u = vec3(rotationMatrix * vec4(this->_properties._u, 0.0f));
	this->_properties._v = vec3(rotationMatrix * vec4(this->_properties._v, 0.0f));
	this->_properties._n = vec3(rotationMatrix * vec4(this->_properties._n, 0.0f));
	this->_properties._up = glm::normalize(glm::cross(this->_properties._n, this->_properties._u));								// This movement doesn't change UP, but it could occur as a result of previous operations
	this->_properties._eye = vec3(rotationMatrix * vec4(this->_properties._eye - this->_properties._lookAt, 1.0f)) + this->_properties._lookAt;

	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::pan(float speed)
{
	if (this->_properties._2d) return;

	const mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), speed, vec3(0.0f, 1.0f, 0.0f));

	// Up vector can change, not in the original position tho. Example: orbit XZ (rotated camera) + pan
	this->_properties._u = vec3(rotationMatrix * vec4(this->_properties._u, 0.0f));
	this->_properties._v = vec3(rotationMatrix * vec4(this->_properties._v, 0.0f));
	this->_properties._n = vec3(rotationMatrix * vec4(this->_properties._n, 0.0f));
	this->_properties._up = glm::normalize(glm::cross(this->_properties._n, this->_properties._u));
	this->_properties._lookAt = vec3(rotationMatrix * vec4(this->_properties._lookAt - this->_properties._eye, 1.0f)) + this->_properties._eye;

	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::tilt(float speed)
{
	if (this->_properties._2d) return;

	const mat4 rotationMatrix = glm::rotate(mat4(1.0f), speed, this->_properties._u);

	const vec3 n = glm::vec3(rotationMatrix * glm::vec4(this->_properties._n, 0.0f));
	float alpha = glm::acos(glm::dot(n, glm::vec3(0.0f, 1.0f, 0.0f)));

	if (alpha < speed || alpha >(glm::pi<float>() - speed))
	{
		return;
	}

	this->_properties._v = glm::vec3(rotationMatrix * glm::vec4(this->_properties._v, 0.0f));
	this->_properties._n = n;
	this->_properties._up = glm::normalize(glm::cross(this->_properties._n, this->_properties._u));											// It could change because of the rotation
	this->_properties._lookAt = glm::vec3(rotationMatrix * glm::vec4(this->_properties._lookAt - this->_properties._eye, 1.0f)) + this->_properties._eye;

	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::truck(float speed)
{
	const mat4 translationMatrix = glm::translate(mat4(1.0f), this->_properties._u * speed);				// Translation in x axis

	this->_properties._eye = vec3(translationMatrix * vec4(this->_properties._eye, 1.0f));
	this->_properties._lookAt = vec3(translationMatrix * vec4(this->_properties._lookAt, 1.0f));

	this->_properties.computeViewMatrices();
}

void AlgGeom::Camera::zoom(float speed)
{
	this->_properties.zoom(speed);
}

/// [Private methods]

void AlgGeom::Camera::copyCameraAttributes(const AlgGeom::Camera* camera)
{
	this->_properties = camera->_properties;

	if (camera->_backupCamera)
	{
		Camera* backupCamera = this->_backupCamera;
		this->_backupCamera = new AlgGeom::Camera(*camera->_backupCamera);

		delete backupCamera;
	}
}
