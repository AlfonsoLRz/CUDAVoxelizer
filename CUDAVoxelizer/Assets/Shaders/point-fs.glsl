#version 450

uniform vec3 pointColor;

layout (location = 0) out vec4 fragmentColor;

void main() 
{
	vec2 centerPointv = gl_PointCoord - 0.5f;
	if (dot(centerPointv, centerPointv) > 0.25f)
		discard;								

	fragmentColor = vec4(pointColor, 1.0f);
}