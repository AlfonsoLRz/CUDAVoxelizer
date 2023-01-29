#pragma once

#include "AABB.h"
#include "VAO.h"

#define BLOCK_SIZE 256

struct AABBGPU
{
	vec3 max, min, stepLength;

	__device__ AABBGPU() { };
	__host__ AABBGPU(const AABB& aabb, const uvec3& numSubdivisions)
	{
		max = aabb.max();
		min = aabb.min();
		stepLength = (max - min) / vec3(numSubdivisions);
	}

	__device__ ivec3 getVoxel(const vec3& position)
	{
		return ivec3((position - min) / stepLength);
	}

	__host__ vec3 getStepLength() { return stepLength; }
};

extern __device__ AABBGPU c_aabb;
extern __device__ uvec3 c_gridDimensions;

__device__ int clamp(int x, int a, int b);
__device__ int getPositionIndex(const ivec3& position);

__global__ void voxelizeComponent(const size_t numVertices, const size_t offsetVertices, const size_t offsetIndices, const size_t numSamples, int* grid, const AlgGeom::VAO::Vertex* vertices, const unsigned* faceIndices, const vec2* noise);
__global__ void countOccupiedVoxels(const size_t numVoxels, int* grid, size_t* count);
__global__ void generateVoxelTranslation(const size_t numVoxels, int* grid, size_t* count, vec3* translation);

