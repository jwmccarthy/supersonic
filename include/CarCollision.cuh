#pragma once

#include <cuda_runtime.h>

#include "CudaMath.cuh"
#include "ArenaMesh.cuh"
#include "GameState.cuh"
#include "Workspace.cuh"
#include "RLConstants.cuh"

__device__ __forceinline__ float4 carAABBMin(ArenaMesh* arena, float4 pos, float4 rot)
{
    float4 aabbMin = ARENA_MAX;

    // Check world space car corners
    #pragma unroll 8
    for (int i = 0; i < 8; ++i)
    {
        float4 local = {
            CAR_OFFSETS.x + CAR_HALF_EX.x * ((i & 1) ? 1.f : -1.f),
            CAR_OFFSETS.y + CAR_HALF_EX.y * ((i & 2) ? 1.f : -1.f),
            CAR_OFFSETS.z + CAR_HALF_EX.z * ((i & 4) ? 1.f : -1.f)
        };

        float4 world = vec3::add(pos, quat::toWorld(local, rot));
        aabbMin = vec3::min(aabbMin, world);
    }

    return aabbMin;
}

__device__ __forceinline__ void carArenaBroadPhase(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int carIdx)
{
    float4 pos = __ldg(&state->cars.pos[carIdx]);
    float4 rot = __ldg(&state->cars.rot[carIdx]);

    // Get cell group flat index
    float4 aabbMin = carAABBMin(arena, pos, rot);
    int3 cellMin = arena->getCellIdx(aabbMin);
    int  cellIdx = arena->getFlatIdx(cellMin.x, cellMin.y, cellMin.z);

    // Count total triangles in overlapping cells
    int numTris = __ldg(&arena->triPre[cellIdx + 1]) - __ldg(&arena->triPre[cellIdx]);

    // Store for narrow phase
    space->cellIdx[carIdx] = cellIdx;
    space->numTris[carIdx] = numTris;
}

__device__ __forceinline__ void carArenaNarrowPhase(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int pairIdx)
{

}
