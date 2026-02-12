#include "CudaCommon.cuh"
#include "ArenaMesh.cuh"

Mesh ArenaMesh::loadMeshObj()
{
    // File stream
    std::string line;
    std::ifstream file(MESH_PATH);

    if (!file.is_open())
    {
        std::cerr << "ERROR: Failed to open mesh file: " << MESH_PATH << "\n";
    }

    // Output vectors
    std::vector<float4> verts;
    std::vector<int4> tris;

    // Read in vertices and triangles
    while (std::getline(file, line))
    {
        char type;
        std::istringstream s(line);
        if (!(s >> type)) continue;

        if (type == 'v')
        {
            // Vertex world locations
            float x, y, z;
            s >> y >> x >> z;
            verts.push_back({ x, y, z, 0 });
        }
        else if (type == 'f')
        {
            // Triangle vertex indices
            int x, y, z;
            s >> x >> y >> z;
            tris.push_back({ x - 1, y - 1, z - 1, 0 });
        }
    }

    // Set mesh dimensions
    nVerts = verts.size();
    nTris = tris.size();

    // Initialize tri normal array
    std::vector<float4> norms(nTris);

    // Initialize pre-computed AABB
    std::vector<float4> aabbMin(nTris);
    std::vector<float4> aabbMax(nTris);

    return { tris, verts, norms, aabbMin, aabbMax };
}

Grid ArenaMesh::buildBroadphaseGrid(Mesh& m)
{
    // Number of cells in grid
    nCells = vec3::prod(GRID_DIMS);

    // Tri accumulators for grid cells
    std::vector<std::vector<int>> cells(nCells);

    for (int i = 0; i < m.tris.size(); ++i)
    {
        int4 tri = m.tris[i];

        // Get vertices via index
        float4 v0 = m.verts[tri.x];
        float4 v1 = m.verts[tri.y];
        float4 v2 = m.verts[tri.z];

        // Get cell index for each
        int3 cX = getCellIdx(v0);
        int3 cY = getCellIdx(v1);
        int3 cZ = getCellIdx(v2);

        // Triangle cell range
        int3 lo = vec3::min(vec3::min(cX, cY), cZ);
        int3 hi = vec3::max(vec3::max(cX, cY), cZ);

        // Clamp w/ lower padding to get all groups
        int lx = clamp(lo.x - 1, 0, GRID_DIMS.x - 1);
        int ly = clamp(lo.y - 1, 0, GRID_DIMS.y - 1);
        int lz = clamp(lo.z - 1, 0, GRID_DIMS.z - 1);

        // Clamp at grid max - incomplete boundary cells
        int hx = clamp(hi.x, 0, GRID_DIMS.x - 1);
        int hy = clamp(hi.y, 0, GRID_DIMS.y - 1);
        int hz = clamp(hi.z, 0, GRID_DIMS.z - 1);

        // Iterate over potential cells
        for (int x = lx; x <= hx; ++x)
        for (int y = ly; y <= hy; ++y)
        for (int z = lz; z <= hz; ++z)
        {
            cells[getFlatIdx(x, y, z)].push_back(i);
        }

        // Pre-compute triangle normals
        m.norms[i] = getTriNormal(v0, v1, v2);

        // Pre-compute triangle AABBs
        m.aabbMin[i] = vec3::min(vec3::min(v0, v1), v2);
        m.aabbMax[i] = vec3::max(vec3::max(v0, v1), v2);
    }

    // 1D grid storage via prefix sum
    std::vector<int> triPre(nCells + 1, 0);
    for (int i = 0; i < nCells; ++i)
    {
        triPre[i + 1] = triPre[i] + cells[i].size();
    }

    // Construct triangle indices
    std::vector<int> triIdx(triPre.back());
    for (int i = 0; i < nCells; ++i)
    {
        std::copy(cells[i].begin(), cells[i].end(), triIdx.begin() + triPre[i]);
    }

    return { triIdx, triPre };
}

ArenaMesh::ArenaMesh()
{
    // Construct broadphase grid
    Mesh m = loadMeshObj();
    Grid g = buildBroadphaseGrid(m);

    // Arena triangle mesh
    cudaMallocCpy(verts, m.verts.data(), m.verts.size());
    cudaMallocCpy(norms, m.norms.data(), m.norms.size());
    cudaMallocCpy(tris,  m.tris.data(), m.tris.size());

    // Triangle bounding boxes
    cudaMallocCpy(aabbMin, m.aabbMin.data(), m.aabbMin.size());
    cudaMallocCpy(aabbMax, m.aabbMax.data(), m.aabbMax.size());

    // Cell-triangle prefix sum for indexing
    cudaMallocCpy(triIdx, g.triIdx.data(), g.triIdx.size());
    cudaMallocCpy(triPre, g.triPre.data(), g.triPre.size());
}
