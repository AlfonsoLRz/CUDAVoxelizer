#pragma once

#include "ShaderProgram.h"

namespace AlgGeom
{
	class RenderingShader: public ShaderProgram
	{
	public:
		RenderingShader();
		virtual ~RenderingShader();
		virtual void applyActiveSubroutines();
		virtual GLuint createShaderProgram(const char* filename);
	};
}

