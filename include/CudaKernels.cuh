#pragma once

#include "GameState.cuh"
#include "ArenaMesh.cuh"
#include "Workspace.cuh"

__global__ void resetKernel(GameState* state);

__global__ void carArenaBroadPhaseKernel(
    GameState* state, 
    ArenaMesh* arena, 
    Workspace* space);

__global__ void carArenaNarrowPhaseKernel(
    GameState* state, 
    ArenaMesh* arena, 
    Workspace* space,
    int nPairs);
