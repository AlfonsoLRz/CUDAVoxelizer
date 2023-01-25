#pragma once

static void HandleError(cudaError_t err,
	const char* file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

class CUDAHandler
{
protected:
	void checkError(cudaError_t result);

public:
	CUDAHandler();
	virtual ~CUDAHandler();
	
	template<typename T>
	void initializeBufferGPU(int*& bufferPointer, T* buffer, size_t size);

	template<typename T>
	void sendBufferGPU(int*& bufferPointer, T* buffer, size_t size);

	void free(int*& bufferPointer) { cudaFree(bufferPointer); }
	void setDevice(uint8_t deviceIndex = UINT8_MAX);
};

template<typename T>
inline void CUDAHandler::initializeBufferGPU(int*& bufferPointer, T* buffer, size_t size)
{
	this->checkError(cudaMalloc((void**)&bufferPointer, size * sizeof(T)));
}

template<typename T>
inline void CUDAHandler::sendBufferGPU(int*& bufferPointer, T* buffer, size_t size)
{
	this->checkError(cudaMemcpy(bufferPointer, buffer, size * sizeof(T), cudaMemcpyHostToDevice));
}
