#pragma once

class CUDAHandler
{
public:
	CUDAHandler();
	virtual ~CUDAHandler();
	
	static void checkError(cudaError_t result);

	template<typename T>
	static void downloadBufferGPU(T*& bufferPointer, T* buffer, size_t size);

	template<typename T>
	static void free(T*& bufferPointer) { cudaFree(bufferPointer); }

	static size_t getNumBlocks(size_t size, size_t blockThreads) { return (size + blockThreads) / blockThreads; }

	template<typename T>
	static void initializeBufferGPU(T*& bufferPointer, size_t size, T* buffer = nullptr);

	static void setDevice(uint8_t deviceIndex = UINT8_MAX);

	static void startTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent);

	static float stopTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent);
};

template<typename T>
inline void CUDAHandler::downloadBufferGPU(T*& bufferPointer, T* buffer, size_t size)
{
	CUDAHandler::checkError(cudaMemcpy(buffer, bufferPointer, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template<typename T>
inline void CUDAHandler::initializeBufferGPU(T*& bufferPointer, size_t size, T* buffer)
{
	CUDAHandler::checkError(cudaMalloc((void**)&bufferPointer, size * sizeof(T)));
	if (buffer)
		CUDAHandler::checkError(cudaMemcpy(bufferPointer, buffer, size * sizeof(T), cudaMemcpyHostToDevice));
}