#pragma once

#define RESTART_PRIMITIVE_INDEX 0xFFFFFFFF

namespace AlgGeom
{
	class VAO
	{
	public:
		enum VBO_slots
		{
			VBO_POSITION, VBO_NORMAL, VBO_TEXT_COORD, VBO_TANGENT, VBO_BITANGENT, 
			VBO_MULTI_POSITION,
			NUM_VBOS
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
		void drawObject(IBO_slots ibo, GLuint openGLPrimitive, GLuint numIndices, GLuint numInstances);
		template<typename T, typename Z>
		int defineMultiInstancingVBO(VBO_slots vbo, const T dataExample, const Z dataPrimitive, const GLuint openGLBasicType);

		template<typename T>
		void setVBOData(VBO_slots vbo, T* geometryData, GLuint size, GLuint changeFrequency = GL_STATIC_DRAW);
		void setVBOData(const std::vector<Vertex>& vertices, GLuint changeFrequency = GL_STATIC_DRAW);
		void setIBOData(IBO_slots ibo, const std::vector<GLuint>& indices, GLuint changeFrequency = GL_STATIC_DRAW);
	};

	template<typename T, typename Z>
	inline int VAO::defineMultiInstancingVBO(VBO_slots vbo, const T dataExample, const Z dataPrimitive, const GLuint openGLBasicType)
	{
		glBindVertexArray(_vao);
		glGenBuffers(1, &_vbos[vbo]);
		glBindBuffer(GL_ARRAY_BUFFER, _vbos[vbo]);
		glEnableVertexAttribArray(vbo);
		glVertexAttribPointer(vbo, sizeof(dataExample) / sizeof(dataPrimitive), openGLBasicType, GL_FALSE, sizeof(dataExample), (GLubyte*)nullptr);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glVertexAttribDivisor(vbo, 1);

		return vbo;
	}

	template<typename T>
	inline void VAO::setVBOData(VBO_slots vbo, T* geometryData, GLuint size, GLuint changeFrequency)
	{
		glBindVertexArray(_vao);
		glBindBuffer(GL_ARRAY_BUFFER, _vbos[vbo]);
		glBufferData(GL_ARRAY_BUFFER, size * sizeof(T), geometryData, changeFrequency);
	}
}

