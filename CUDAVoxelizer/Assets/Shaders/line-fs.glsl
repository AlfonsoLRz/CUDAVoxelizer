#version 450

uniform vec3 lineColor;

layout (location = 0) out vec4 fragmentColor;

void main()
{
	fragmentColor = vec4(lineColor, 1.0f);
}