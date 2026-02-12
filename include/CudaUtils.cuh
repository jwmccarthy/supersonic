#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ int hash(int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    return (x >> 16) ^ x;
}

__device__ __forceinline__ int binarySearch(int* a, int n, int t)
{
    int lo = 0;
    int hi = n;

    while (lo < hi)
    {
        int mid = (lo + hi) / 2;

        if (a[mid + 1] <= t)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }

    return lo;
}