#pragma once

#include "RenderingShader.h"
#include "Singleton.h"

namespace AlgGeom
{
	class ShaderProgramDB : public Singleton<ShaderProgramDB>
	{
		friend class Singleton<ShaderProgramDB>;

	private:
		static std::unordered_map<uint8_t, std::string>								RENDERING_SHADER_PATH;
		static std::unordered_map<uint8_t, std::unique_ptr<AlgGeom::RenderingShader>>	_renderingShader;

	public:
		enum RenderingShaderId
		{
			POINT_RENDERING,
			LINE_RENDERING,
			TRIANGLE_RENDERING,
		};

	private:
		ShaderProgramDB();

	public:
		virtual ~ShaderProgramDB();
		RenderingShader* getShader(RenderingShaderId shaderId);
	};
}

