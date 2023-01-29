#version 450

layout (location = 0) in vec3 vPosition;
layout (location = 5) in vec3 vOffset;

uniform mat4 mModelViewProj;	
uniform vec3 globalScale;

subroutine vec3 instanceType();
subroutine uniform instanceType instanceUniform;


// ------------------------------------
// ---------- MULTI_INSTANCE ----------
// ------------------------------------

subroutine(instanceType)
vec3 multiInstanceUniform()
{
	return vPosition * globalScale + vOffset;
}

subroutine(instanceType)
vec3 singleInstanceUniform()
{
	return vPosition;
}


void main() 
{
	gl_Position = mModelViewProj * vec4(instanceUniform(), 1.0f);
}