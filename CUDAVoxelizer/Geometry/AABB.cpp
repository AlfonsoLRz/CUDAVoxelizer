#include "stdafx.h"
#include "AABB.h"

// Public methods

AABB::AABB(const vec3& min, const vec3& max) : _max(max), _min(min)
{
}

AABB::AABB(const AABB& aabb) : _max(aabb._max), _min(aabb._min)
{
}

AABB::~AABB()
{
}

AABB& AABB::operator=(const AABB& aabb)
{
	_max = aabb._max;
	_min = aabb._min;

	return *this;
}

AABB AABB::dot(const mat4& matrix)
{
	return AABB(matrix * vec4(_min, 1.0f), matrix * vec4(_max, 1.0f));
}

void AABB::update(const AABB& aabb)
{
	this->update(aabb.max());
	this->update(aabb.min());
}

void AABB::update(const vec3& point)
{
	if (point.x < _min.x) { _min.x = point.x; }
	if (point.y < _min.y) { _min.y = point.y; }
	if (point.z < _min.z) { _min.z = point.z; }

	if (point.x > _max.x) { _max.x = point.x; }
	if (point.y > _max.y) { _max.y = point.y; }
	if (point.z > _max.z) { _max.z = point.z; }
}

std::ostream& operator<<(std::ostream& os, const AABB& aabb)
{
	os << "Maximum corner: " << aabb.max().x << ", " << aabb.max().y << ", " << aabb.max().z << "\n";
	os << "Minimum corner: " << aabb.min().x << ", " << aabb.min().y << ", " << aabb.min().z << "\n";

	return os;
}
