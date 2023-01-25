#include "stdafx.h"
#include "CUDAHandler.h"

// Public methods

CUDAHandler::CUDAHandler()
{
}

CUDAHandler::~CUDAHandler()
{
}

void CUDAHandler::setDevice(uint8_t deviceIndex)
{
    int numDevices, bestDevice = 0;
    checkError(cudaGetDeviceCount(&numDevices));

    if (deviceIndex == UINT8_MAX)
    {
        size_t bestScore = 0;
        for (int deviceIdx = 0; deviceIdx < numDevices; deviceIdx++)
        {
            int clockRate;
            int numProcessors;
            checkError(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, deviceIdx));
            checkError(cudaDeviceGetAttribute(&numProcessors, cudaDevAttrMultiProcessorCount, deviceIdx));

            size_t score = clockRate * numProcessors;
            if (score > bestScore)
            {
                bestDevice = deviceIdx;
                bestScore = score;
            }
        }

        if (bestScore == 0)
            throw std::runtime_error("CudaModule: No appropriate CUDA device found!");
    }
    else
    {
        bestDevice = glm::clamp(deviceIndex, static_cast<uint8_t>(0), static_cast<uint8_t>(numDevices));
    }

    checkError(cudaSetDevice(bestDevice));
}

// Protected methods

void CUDAHandler::checkError(cudaError_t result)
{
    if (result != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}