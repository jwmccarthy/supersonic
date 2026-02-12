#pragma once

struct Workspace
{
    // Car-arena collision
    int*  numTris;  // nCar
    int4* cellMin;  // nCar
    int4* cellMax;  // nCar
};