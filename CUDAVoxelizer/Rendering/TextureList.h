#pragma once

#include "Singleton.h"
#include "Texture.h"

namespace AlgGeom
{
	class TextureList : public Singleton<TextureList>
	{
		friend class Singleton<TextureList>;

	private:
		struct ColorHash {
			std::size_t operator()(const vec4& color) const
			{
				glm::ivec4 iColor = color * 256.0f;
				return static_cast<size_t>(iColor.x + iColor.y * 256 + iColor.z * 512 + iColor.w + 1024);
			}
		};

		struct ColorEqual {
			bool operator()(const vec4& c1, const vec4& c2) const
			{
				return glm::distance(c1, c2) < glm::epsilon<float>();
			}
		};

	private:
		std::unordered_map<vec4, AlgGeom::Texture*, ColorHash, ColorEqual>	_colorTexture;
		std::unordered_map<std::string, AlgGeom::Texture*>					_imageTexture;

	private:
		TextureList();

	public:
		virtual ~TextureList();

		AlgGeom::Texture* getTexture(const vec4& color);
		AlgGeom::Texture* getTexture(const std::string& path);
		void saveTexture(const vec4& color, AlgGeom::Texture* texture);
		void saveTexture(const std::string& path, AlgGeom::Texture* texture);
	};
}

