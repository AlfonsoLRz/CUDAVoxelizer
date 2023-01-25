#pragma once

class AABB
{
protected:
	vec3	_max, _min;		

public:
	AABB(const vec3& min = vec3(INFINITY), const vec3& max = vec3(-INFINITY));
	AABB(const AABB& aabb);
	virtual ~AABB();
	AABB& operator=(const AABB& aabb);

	vec3 center() const { return (_max + _min) / 2.0f; }
	AABB dot(const mat4& matrix);
	vec3 extent() const { return _max - center(); }
	vec3 max() const { return _max; }
	vec3 min() const { return _min; }
	vec3 size() const { return _max - _min; }

	void update(const AABB& aabb);
	void update(const vec3& point);

	friend std::ostream& operator<<(std::ostream& os, const AABB& aabb);
};

