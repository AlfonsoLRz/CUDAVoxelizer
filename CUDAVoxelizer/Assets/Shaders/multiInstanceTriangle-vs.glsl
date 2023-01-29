#version 450
 
layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTextCoord;
layout (location = 5) in vec3 vOffset;

uniform mat4 mModelView;
uniform mat4 mModelViewProj;
uniform float voxelLength;

out vec3 position;
out vec3 normal;
out vec2 textCoord;


void main ()
{
	position = vPosition;
	position = vec3(mModelView * vec4(position, 1.0f));
	gl_Position = mModelViewProj * vec4(position, 1.0f);
	normal = (mModelView * vec4(vNormal, .0f)).xyz;
	textCoord = vTextCoord;
}