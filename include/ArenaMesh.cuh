#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#include "CudaMath.cuh"
#include "RLConstants.cuh"

#define MESH_PATH "./assets/pitch.obj"

struct Mesh
{
    std::vector<int4>   tris;
    std::vector<float4> verts;
    std::vector<float4> norms;
    std::vector<float4> aabbMin;
    std::vector<float4> aabbMax;
};

struct Grid
{
    std::vector<int> triIdx;
    std::vector<int> triPre;
};

struct ArenaMesh
{
    // Mesh dims
    int nTris;
    int nVerts;
    int nCells;

    // Device mesh arrays
    float4* verts;
    float4* norms;
    int4*   tris;

    // Tri bounding boxes
    float4* aabbMin;
    float4* aabbMax;

    // Broadphase grid
    int* triIdx;
    int* triPre;

    ArenaMesh();

    // Mesh & grid construction
    Mesh loadMeshObj();
    Grid buildBroadphaseGrid(Mesh& m);

    // Indexing helpers
    __host__ __device__ __forceinline__ int3 getCellIdx(float4 p)
    {
        // Convert world position to cell coordinates
        int x = (int)((p.x - ARENA_MIN.x) / CELL_SIZE.x);
        int y = (int)((p.y - ARENA_MIN.y) / CELL_SIZE.y);
        int z = (int)((p.z - ARENA_MIN.z) / CELL_SIZE.z);

        // Clamp to grid bounds
        x = max(0, min(x, GRID_DIMS.x - 1));
        y = max(0, min(y, GRID_DIMS.y - 1));
        z = max(0, min(z, GRID_DIMS.z - 1));

        return { x, y, z };
    }

    __host__ __device__ __forceinline__ int getFlatIdx(int x, int y, int z)
    {
        return x + y * GRID_DIMS.x + z * GRID_DIMS.x * GRID_DIMS.y;
    }
};

inline float4 getTriNormal(float4 v1, float4 v2, float4 v3)
{
    // Get edge vectors
    float4 e1 = vec3::sub(v2, v1);
    float4 e2 = vec3::sub(v3, v1);

    return vec3::norm(vec3::cross(e1, e2));
}
