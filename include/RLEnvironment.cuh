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
    GameState  m_state;
    GameState* d_state;

    // Static arena mesh
    ArenaMesh  m_arena;
    ArenaMesh* d_arena;

    // Intermediate outputs
    Workspace  m_space;
    Workspace* d_space;

public:
    RLEnvironment(int sims, int numB, int numO, int seed);

    // Gym API
    void step();
    void reset();
};
