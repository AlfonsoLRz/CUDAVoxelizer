#version 450

// ------------ Constraints ------------
#define CUTOFF .8f
#define EPSILON 0.00001f
#define PI 3.14159265359f

// ------------------------------------
// ---------- GEOMETRY ----------------
// ------------------------------------
in vec3 position;
in vec3 normal;
in vec2 textCoord;

// ------------------------------------
// ---------- LIGHTING ---------------
// ------------------------------------
uniform vec3 lightPosition;
uniform vec3 Ia;
uniform vec3 Id;
uniform vec3 Is;
uniform float gamma;

// ------------------------------------
// ---------- MATERIALS ---------------
// ------------------------------------
uniform sampler2D texKdSampler;

uniform vec4 Kd;
uniform vec3 Ks;
uniform float metallic;
uniform float roughnessK;

subroutine vec4 kadTextureType();
subroutine uniform kadTextureType kadUniform;

layout(location = 0) out vec4 fragmentColor;

// ------------------------------------
// ---------- LIGHTING ---------------
// ------------------------------------

vec3 fresnelSchlick(float cos_theta, vec3 F0)
{
	return F0 + (1.0f - F0) * pow(clamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
}

float distributionGGX(vec3 n, vec3 h, float shininess)
{
	float a = shininess * shininess;
	float a2 = a * a;
	float dotNH = max(dot(n, h), 0.0);
	float dotNH2 = dotNH * dotNH;

	float num = a2;
	float denom = (dotNH2 * (a2 - 1.0f) + 1.0f);
	denom = PI * denom * denom;

	return num / denom;
}

float geometrySchlickGGX(float dotNV, float shininess)
{
	float r = (shininess + 1.0f);
	float k = (r * r) / 8.0f;

	float num = dotNV;
	float denom = dotNV * (1.0f - k) + k;

	return num / denom;
}

float geometrySmith(float dotNV, float dotNL, float shininess)
{
	float ggx2 = geometrySchlickGGX(dotNV, shininess);
	float ggx1 = geometrySchlickGGX(dotNL, shininess);

	return ggx1 * ggx2;
}

vec3 getDiffuseAndSpecular(vec3 fragKad, vec3 fragKs, vec3 fragNormal, float metallic, float shininess)
{
	const vec3 n = normalize(fragNormal);
	const vec3 l = normalize(lightPosition - position);
	const vec3 v = normalize(-position);
	const vec3 h = normalize(v + l);						// Halfway vector

	const float dotLN = clamp(dot(l, n), -1.0f, 1.0f);      // Prevents Nan values from acos
	const float dotHN = dot(h, n);
	const float dotHV = dot(h, v);
	const float dotNV = dot(n, v);

	vec3 F0 = vec3(.04);
	F0 = mix(F0, fragKad, metallic);

	// BRDF
	float NDF = distributionGGX(n, h, shininess);
	float G = geometrySmith(dotNV, dotLN, shininess);
	vec3 F = fresnelSchlick(max(dotHV, .0f), F0);

	vec3 kS = F;
	vec3 kD = vec3(1.0f) - kS;
	kD *= 1.0f - metallic;

	vec3 numerator = NDF * G * F;
	float denominator = 4.0f * max(dotNV, 0.0f) * max(dotLN, 0.0f) + 0.0001f;
	vec3 specularFactor = clamp(numerator / denominator, .0f, 1.0f);

	return (Id * (fragKad + Ia) / PI + fragKs * Is * specularFactor) * max(dotLN, 0.0f);
}

vec3 pointLight(vec3 fragKad, vec3 fragKs, vec3 fragNormal, float metallic, float shininess)
{
	return getDiffuseAndSpecular(fragKad, fragKs, fragNormal, metallic, shininess);
}

// ------------------------------------
// ---------- MATERIALS ---------------
// ------------------------------------

subroutine(kadTextureType)
vec4 getUniformColor()
{
	return Kd;
}

subroutine(kadTextureType)
vec4 getTextureColor()
{
	return texture(texKdSampler, textCoord);
}


void main ()
{
	const vec4 fragKad = kadUniform();
	const vec3 reflectionColor = pointLight(fragKad.rgb, Ks, normal, metallic, roughnessK);
	fragmentColor = vec4(pow(reflectionColor, vec3(1.0 / gamma)), fragKad.w);
}