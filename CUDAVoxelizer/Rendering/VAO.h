#pragma once

#define RESTART_PRIMITIVE_INDEX 0xFFFFFFFF

namespace AlgGeom
{
	class VAO
	{
	public:
		enum VBO_slots
		{
			VBO_POSITION, VBO_NORMAL, VBO_TEXT_COORD, VBO_TANGENT, VBO_BITANGENT, NUM_VBOS
		};

		enum IBO_slots
		{
			IBO_POINT,
			IBO_LINE,
			IBO_TRIANGLE,
			NUM_IBOS
		};

		struct Vertex
		{
			vec3		_position, _normal;
			vec2		_textCoord;
		};

	private:
		GLuint				_vao;						
		std::vector<GLuint> _vbos;					
		std::vector<GLuint> _ibos;	

	private:
		void defineInterleavedVBO(GLuint vboId);
		void defineNonInterleaveVBO(GLuint vboId, size_t structSize, GLuint elementType, uint8_t slot);

	public:
		VAO(bool interleaved = true);
		virtual ~VAO();

		void drawObject(IBO_slots ibo, GLuint openGLPrimitive, GLuint numIndices);

		template<typename T>
		void setVBOData(VBO_slots vbo, T* geometryData, GLuint size, GLuint changeFrequency = GL_STATIC_DRAW);
		void setVBOData(const std::vector<Vertex>& vertices, GLuint changeFrequency = GL_STATIC_DRAW);
		void setIBOData(IBO_slots ibo, const std::vector<GLuint>& indices, GLuint changeFrequency = GL_STATIC_DRAW);
	};

	template<typename T>
	inline void VAO::setVBOData(VBO_slots vbo, T* geometryData, GLuint size, GLuint changeFrequency)
	{
		glBindVertexArray(_vao);
		glBindBuffer(GL_ARRAY_BUFFER, _vbos[vbo]);
		glBufferData(GL_ARRAY_BUFFER, size * sizeof(T), geometryData, changeFrequency);
	}
}

