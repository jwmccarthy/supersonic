#pragma once

#include <cuda_runtime.h>

#include "CudaMath.cuh"
#include "RLConstants.cuh"

struct AABB
{
    float4 min;
    float4 max;
};

__device__ __forceinline__ AABB getCarAABB(ArenaMesh* arena, float4 pos, float4 rot)
{
    float4 aabbMin = ARENA_MAX;
    float4 aabbMax = ARENA_MIN;

    // Check world space car corners
    #pragma unroll 8
    for (int i = 0; i < 8; ++i)
    {
        float4 local = {
            CAR_OFFSETS.x + CAR_HALF_EX.x * ((i & 1) ? 1.f : -1.f),
            CAR_OFFSETS.y + CAR_HALF_EX.y * ((i & 2) ? 1.f : -1.f),
            CAR_OFFSETS.z + CAR_HALF_EX.z * ((i & 4) ? 1.f : -1.f), 0.0f
        };

        float4 world = vec3::add(pos, quat::toWorld(local, rot));
        aabbMin = vec3::min(aabbMin, world);
        aabbMax = vec3::max(aabbMax, world);
    }

    return { aabbMin, aabbMax };
}

__device__ __forceinline__ void carArenaBroadPhase(GameState* state, ArenaMesh* arena, Workspace* space, int carIdx)
{
    float4 pos = __ldg(&state->cars.position[carIdx]);
    float4 rot = __ldg(&state->cars.rotation[carIdx]);

    auto [aabbMin, aabbMax] = getCarAABB(arena, pos, rot);

    int3 cellMin = arena->getCellIdx(aabbMin);
    int3 cellMax = arena->getCellIdx(aabbMax);
    
    int numTris = 0;

    // Count total triangles in overlapping cells
    for (int x = cellMin.x; x <= cellMax.x; ++x)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        int cellIdx = arena->flatCellIdx(x, y, z);
        numTris += __ldg(&arena->triPre[cellIdx + 1]) - __ldg(&arena->triPre[cellIdx]);
    }

    // Store for narrow phase
    space->numTris[carIdx] = numTris;
    space->cellMin[carIdx] = make_int4(cellMin.x, cellMin.y, cellMin.z, 0);
    space->cellMax[carIdx] = make_int4(cellMax.x, cellMax.y, cellMax.z, 0);
}