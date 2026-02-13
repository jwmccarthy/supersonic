#include <cub/device/device_scan.cuh>

#include "CudaCommon.cuh"
#include "GameState.cuh"
#include "CudaKernels.cuh"
#include "RLEnvironment.cuh"

using ds = cub::DeviceScan;

RLEnvironment::RLEnvironment(int sims, int numB, int numO, int seed)
    : h_state(sims, numB, numO, seed)
    , h_arena()
    , h_space()
{
    int n = h_state.nCar;

    // Physics state & arena mesh
    cudaMallocCpy(d_state, &h_state);
    cudaMallocCpy(d_arena, &h_arena);

    // Intermediate algo results
    cudaMallocSOA(h_space, {n, n, n + 1});
    cudaMallocCpy(d_space, &h_space);

    // Initialize prefix sum space
    ds::ExclusiveSum(nullptr, cubBytes, h_space.numTris, h_space.triPrfx, n + 1);
    CUDA_CHECK(cudaMalloc(&d_cubBuf, cubBytes));
}

void RLEnvironment::step()
{
    int n = h_state.nCar;

    int blockSize = 256;
    int gridSize = (h_state.nCar + blockSize - 1) / blockSize;

    // Obtain triangle candidate counts per car
    carArenaBroadPhaseKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_space);

    // Prefix sum over candidate counts to size narrow phase kernel launch
    ds::ExclusiveSum(d_cubBuf, cubBytes, h_space.numTris, h_space.triPrfx, n + 1);
    cudaMemcpy(&h_nPairs, h_space.triPrfx + n, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_nPairs > 0)
    {
        gridSize = (h_nPairs + blockSize - 1) / blockSize;

        // Perform SAT over car-triangle candidate pairs
        carArenaNarrowPhaseKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_space, h_nPairs);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void RLEnvironment::reset()
{
    int blockSize = 32;
    int gridSize = (h_state.sims + blockSize - 1) / blockSize;

    // Random kickoff positions and ball at center
    resetKernel<<<gridSize, blockSize>>>(d_state);

    CUDA_CHECK(cudaDeviceSynchronize());
}
