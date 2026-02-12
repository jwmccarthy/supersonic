#include "CudaCommon.cuh"
#include "GameState.cuh"
#include "CudaKernels.cuh"
#include "RLEnvironment.cuh"

RLEnvironment::RLEnvironment(int sims, int numB, int numO, int seed)
    : m_state(sims, numB, numO, seed)
    , m_arena()
{
    // Physics state & arena mesh
    cudaMallocCpy(d_state, &m_state);
    cudaMallocCpy(d_arena, &m_arena);

    // Intermediate algo results
    cudaMallocSOA(m_space, m_state.nCar);
    cudaMallocCpy(d_space, &m_space);
}

void RLEnvironment::step()
{
    int blockSize = 256;
    int gridSize = (m_state.nCar + blockSize - 1) / blockSize;


}

void RLEnvironment::reset()
{
    int blockSize = 32;
    int gridSize = (m_state.sims + blockSize - 1) / blockSize;

    resetKernel<<<gridSize, blockSize>>>(d_state);
    CUDA_CHECK(cudaDeviceSynchronize());
}