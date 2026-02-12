#include "StateReset.cuh"
#include "CudaUtils.cuh"
#include "CarCollision.cuh"
#include "CudaKernels.cuh"

__global__ void resetKernel(GameState* state)
{
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->sims) return;

    resetToKickoff(state, simIdx);
}

__global__ void carArenaBroadPhaseKernel(
    GameState* state, 
    ArenaMesh* arena, 
    Workspace* space)
{
    int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (carIdx >= state->nCar) return;

    carArenaBroadPhase(state, arena, space, carIdx);
}

__global__ void carArenaNarrowPhaseKernel(
    GameState* state, 
    ArenaMesh* arena, 
    Workspace* space,
    int nPairs)
{
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= nPairs) return;

    // Binary search for car index
    // int carIdx = binarySearch();
}
