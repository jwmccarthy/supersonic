#pragma once

#pragma once

#include <cuda_runtime.h>

#include "GameState.cuh"
#include "ArenaMesh.cuh"
#include "Workspace.cuh"

class RLEnvironment
{
private:
    // Game physics state
    GameState  h_state;
    GameState* d_state;

    // Static arena mesh
    ArenaMesh  h_arena;
    ArenaMesh* d_arena;

    // Intermediate outputs
    Workspace  h_space;
    Workspace* d_space;

    // Prefix sum buffer
    int    h_nPairs;
    void*  d_cubBuf;
    size_t cubBytes = 0;

public:
    RLEnvironment(int sims, int numB, int numO, int seed);

    // Gym API
    void step();
    void reset();
};
