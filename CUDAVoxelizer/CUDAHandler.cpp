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

void CUDAHandler::startTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{
    checkError(cudaEventCreate(&startEvent));
    checkError(cudaEventCreate(&stopEvent));
    checkError(cudaEventRecord(startEvent, 0));
}

float CUDAHandler::stopTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{
    float ms;
    checkError(cudaEventRecord(stopEvent, 0));
    checkError(cudaEventSynchronize(stopEvent));
    checkError(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    return ms;
}

// Protected methods

void CUDAHandler::checkError(cudaError_t result)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}