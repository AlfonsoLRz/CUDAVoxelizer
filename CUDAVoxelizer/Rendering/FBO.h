#pragma once

#include "Image.h"

namespace AlgGeom
{
	class FBO
	{
	protected:
		GLuint	_id;
		ivec2	_size;

	protected:
		void checkFBOstate();

	public:
		FBO(const uint16_t width, const uint16_t height);
		virtual ~FBO();

		virtual GLuint getId() const { return _id; }
		ivec2 getSize() const { return _size; }

		virtual void bindFBO() = 0;
		virtual void modifySize(const uint16_t width, const uint16_t height);

		void threadedWriteImage(std::vector<GLubyte>* pixels, const std::string& filename, const uint16_t width, const uint16_t height);
	};
}

