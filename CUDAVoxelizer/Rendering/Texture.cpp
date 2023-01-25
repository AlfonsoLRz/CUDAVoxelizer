#include "stdafx.h"
#include "Texture.h"

// [Static variables initialization]

const GLuint AlgGeom::Texture::MAG_FILTER = GL_LINEAR;
const GLuint AlgGeom::Texture::MIN_FILTER = GL_LINEAR_MIPMAP_NEAREST;
const GLuint AlgGeom::Texture::WRAP_S = GL_MIRRORED_REPEAT;
const GLuint AlgGeom::Texture::WRAP_T = GL_MIRRORED_REPEAT;
const GLuint AlgGeom::Texture::WRAP_R = GL_MIRRORED_REPEAT;

/// [Public methods]

/** glTexImage2D:
* Target. Type of desired texture: GL_TEXTURE_2D, GL_PROXY_TEXTURE_2D...
* Level:  Level of Detail. Level 0 is the base image. Level n is n-reduced image with mipmap algorithm
* Internal Format:  Number of texture components
* Width
* Height
* Border: must be 0
* Format: Format of pixel information. GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_BGRA, GL_RED_INTEGER...
* Type: Type of data of pixel information. GL_UNSIGNED_BYTE, GL_BYTE, GL_UNSIGNED_SHORT, GL_SHORT, GL_UNSIGNED_INT, GL_INT, GL_FLOAT...
* Data: Image pixels
*/

AlgGeom::Texture::Texture(Image* image, const GLuint wrapS, const GLuint wrapT, const GLuint minFilter, const GLuint magFilter)
	: _id(-1), _filename(image->getFilename()), _color(.0f)
{
	unsigned char* bits = image->bits();
	if (image and !bits)
	{
		throw std::runtime_error("Failed to generate texture from image!: " + image->getFilename());
	}

	const unsigned int width = image->getWidth(), height = image->getHeight();
	const GLuint depthID[] = { GL_RED, GL_RED, GL_RGB, GL_RGBA };

	glGenTextures(1, &_id);
	glBindTexture(GL_TEXTURE_2D, _id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);

	glTexImage2D(GL_TEXTURE_2D, 0, depthID[image->getDepth() - 1], width, height, 0, depthID[image->getDepth() - 1], GL_UNSIGNED_BYTE, bits);
	glGenerateMipmap(GL_TEXTURE_2D);
}

AlgGeom::Texture::Texture(const vec4& color)
	: _id(-1), _filename(""), _color(color)
{
	const int width = 1, height = 1;
	const unsigned char image[] = { 
		static_cast<unsigned char>(255.0f * color.x), static_cast<unsigned char>(255.0f * color.y), 
		static_cast<unsigned char>(255.0f * color.z), static_cast<unsigned char>(255.0f * color.a) 
	};

	glGenTextures(1, &_id);
	glBindTexture(GL_TEXTURE_2D, _id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MIN_FILTER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, MAG_FILTER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, WRAP_S);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, WRAP_T);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
}

AlgGeom::Texture::~Texture()
{
	glDeleteTextures(1, &_id);
}

void AlgGeom::Texture::applyTexture(AlgGeom::ShaderProgram* shader, const GLint id, const std::string& shaderVariable)
{
	shader->setUniform(shaderVariable, id);
	glActiveTexture(GL_TEXTURE0 + id);
	glBindTexture(GL_TEXTURE_2D, _id);
}
