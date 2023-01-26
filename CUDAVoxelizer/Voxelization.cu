#include "stdafx.h"
#include "Voxelization.cuh"

__device__ AABBGPU c_aabb;
__device__ uvec3 c_gridDimensions;


__device__ int clamp(int x, int a, int b)
{
	return max(a, min(b, x));
}

__device__ int getPositionIndex(const ivec3& position)
{
	return position.x * c_gridDimensions.y * c_gridDimensions.z + position.y * c_gridDimensions.z + position.z;
}

__global__ void voxelizeComponent(const size_t numFaces, const size_t numSamples, int* grid, const AlgGeom::VAO::Vertex* vertices, const unsigned* faceIndices, const vec2* noise)
{
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadID < numFaces * numSamples)
	{ 
		int faceIdx = (threadID / numSamples) * 4;
		int sampleIdx = threadID % numSamples;

		vec3 v1 = vertices[faceIndices[faceIdx + 0]]._position, v2 = vertices[faceIndices[faceIdx + 1]]._position, v3 = vertices[faceIndices[faceIdx + 2]]._position;
		vec3 u = v2 - v1, v = v3 - v1;
		vec2 noiseSample = noise[sampleIdx];

		if (noiseSample.x + noiseSample.y >= 1.0f)
			noiseSample = 1.0f - noiseSample;

		vec3 point = v1 + u * noiseSample.x + v * noiseSample.y;
		ivec3 voxelIndices = c_aabb.getVoxel(point);
		int positionIndex = getPositionIndex(voxelIndices);

		grid[positionIndex] = 1;
		//result[threadID] = vec3(faceIdx, sampleIdx, numFaces);
	}
}

__global__ void countOccupiedVoxels(const size_t numVoxels, int* grid, size_t* count)
{
	int cellIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (cellIdx < numVoxels)
		atomicAdd(count, grid[cellIdx]);
}

__global__ void generateVoxelTranslation(const size_t numVoxels, int* grid, size_t* count, vec3* translation)
{
	int cellIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (cellIdx < numVoxels && grid[cellIdx] > 0)
	{
		int translationIndex = atomicAdd(count, 1);
		int x = cellIdx / (c_gridDimensions.y * c_gridDimensions.z);
		int w = cellIdx % (c_gridDimensions.y * c_gridDimensions.z);
		int y = w / c_gridDimensions.z;
		int z = w % c_gridDimensions.z;

		translation[translationIndex] = c_aabb.min + c_aabb.stepLength * vec3(x, y, z) + c_aabb.stepLength / 2.0f;
	}
}