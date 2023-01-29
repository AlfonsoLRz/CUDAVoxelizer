#version 450
 
layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTextCoord;
layout (location = 5) in vec3 vOffset;

uniform mat4 mModelView;
uniform mat4 mModelViewProj;
uniform vec3 globalScale;

subroutine vec3 instanceType();
subroutine uniform instanceType instanceUniform;

out vec3 position;
out vec3 normal;
out vec2 textCoord;


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

void main ()
{
	vec3 instancePosition = instanceUniform();
	position = vec3(mModelView * vec4(instancePosition, 1.0f));
	gl_Position = mModelViewProj * vec4(instancePosition, 1.0f);
	normal = (mModelView * vec4(vNormal, .0f)).xyz;
	textCoord = vTextCoord;
}