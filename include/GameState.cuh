#pragma once

#include <cuda_runtime.h>

#include "CudaCommon.cuh"

struct Ball
{
    float4* pos;
    float4* vel;
    float4* ang;
};

struct Cars
{
    float4* pos;
    float4* vel;
    float4* ang;
    float4* rot;
};

struct GameState
{
    int sims;  // Total simulations
    int numB;  // Number of blue cars per sim
    int numO;  // Number of orange cars per sim
    int nCar;  // Number of cars across all sims
    int seed;  // Pseudo-random seed

    Ball ball{};
    Cars cars{};

    GameState(int sims, int numB, int numO, int seed)
        : sims(sims), numB(numB), numO(numO), seed(seed)
        , nCar(sims * (numB + numO))
    {
        cudaMallocSOA(ball, sims);
        cudaMallocSOA(cars, nCar);
    }

    ~GameState()
    {
        cudaFreeSOA(ball);
        cudaFreeSOA(cars);
    }
};
