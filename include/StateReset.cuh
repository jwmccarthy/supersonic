#pragma once

#include <cuda_runtime.h>

#include "CudaCommon.cuh"
#include "CudaUtils.cuh"
#include "GameState.cuh"
#include "RLConstants.cuh"

__device__ __forceinline__ void resetBall(Ball* ball, int simIdx)
{
    ball->pos[simIdx] = { 0, 0, BALL_REST_Z, 0 };
    ball->vel[simIdx] = { 0, 0, 0, 0 };
    ball->ang[simIdx] = { 0, 0, 0, 0 };
}

__device__ __forceinline__ void resetCar(Cars* cars, int carIdx, int locIdx, bool invert)
{
    CarSpawn loc = KICKOFF_LOCATIONS[locIdx % 5];

    // Reflect location based on team
    float x   = invert ? -loc.x   : loc.x;
    float y   = invert ? -loc.y   : loc.y;
    float yaw = invert ? -loc.yaw : loc.yaw;;

    cars->pos[carIdx] = { x, y, loc.z, 0 };
    cars->vel[carIdx] = { 0, 0, 0, 0 };
    cars->ang[carIdx] = { 0, 0, 0, 0 };
    cars->rot[carIdx] = { 0, 0, sinf(yaw / 2), cosf(yaw / 2) };
}

__device__ __forceinline__ void resetToKickoff(GameState* state, int simIdx)
{
    const int sims = state->sims;
    const int numB = state->numB;
    const int numO = state->numO;
    const int nCar = numB + numO;

    // Pseudorandom kickoff permutation
    const int  permIdx = hash(simIdx ^ sims) % 120;
    const int* carLocs = KICKOFF_PERMUTATIONS[permIdx];

    // Ball back to center field
    resetBall(&state->ball, simIdx);

    #pragma unroll 2
    for (int team = 0; team < 2; team++)
    {
        // Invert orange positions
        const bool invert = team;
        const int numCars = team ? numO : numB;
        
        for (int i = 0; i < numCars; i++)
        {
            const int locIdx = carLocs[i];
            const int carIdx = simIdx * nCar + (team * numB + i);

            // Place cars at kickoff positions
            resetCar(&state->cars, carIdx, locIdx, invert);
        }
    }
}
