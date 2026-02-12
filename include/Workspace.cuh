#pragma once

struct Workspace
{
    // Car-arena collision
    int* cellIdx;  // nCar
    int* numTris;  // nCar
    int* triPrfx;  // nCar + 1
};