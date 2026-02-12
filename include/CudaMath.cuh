#pragma once

#include <cmath>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define rsqrtf(x) (1.0f / std::sqrt(x))
#endif

template <typename T>
__host__ __device__ __forceinline__ T min(T a, T b)
{ 
    return a < b ? a : b; 
}

template <typename T>
__host__ __device__ __forceinline__ T max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T, typename S>
__host__ __device__ __forceinline__ T clamp(T a, S l, S h)
{
    return a < l ? l : (a > h ? h : a);
}

namespace vec3
{

template <typename T>
__host__ __device__ __forceinline__ T min(T a, T b)
{
    return { ::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z) };
}

template <typename T>
__host__ __device__ __forceinline__ T max(T a, T b)
{
    return { ::max(a.x, b.x), ::max(a.y, b.y), ::max(a.z, b.z) };
}

template <typename T>
__host__ __device__ __forceinline__ T add(T a, T b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

template <typename T, typename S>
__host__ __device__ __forceinline__ T add(T a, S b)
{
    return { a.x + b, a.y + b, a.z + b };
}

template <typename T>
__host__ __device__ __forceinline__ T sub(T a, T b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

template <typename T, typename S>
__host__ __device__ __forceinline__ T sub(T a, S b)
{
    return { a.x - b, a.y - b, a.z - b };
}

template <typename T>
__host__ __device__ __forceinline__ T mul(T a, T b)
{
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

template <typename T, typename S>
__host__ __device__ __forceinline__ T mul(T a, S b)
{
    return { a.x * b, a.y * b, a.z * b };
}

template <typename T>
__host__ __device__ __forceinline__ T div(T a, T b)
{
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}

template <typename T, typename S>
__host__ __device__ __forceinline__ T div(T a, S b)
{
    return { a.x / b, a.y / b, a.z / b };
}

template <typename T>
__host__ __device__ __forceinline__ auto dot(T a, T b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
__host__ __device__ __forceinline__ auto prod(T a)
{
    return a.x * a.y * a.z;
}

template <typename T>
__host__ __device__ __forceinline__ T cross(T a, T b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

template <typename T>
__host__ __device__ __forceinline__ T norm(T a)
{
    auto lenSq = dot(a, a);
    if (lenSq <= 1e-6f) return {};
    return mul(a, rsqrtf(lenSq));
}

}

namespace vec4
{

template <typename T>
__host__ __device__ __forceinline__ T mul(T a, T b)
{
    return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

template <typename T, typename S>
__host__ __device__ __forceinline__ T mul(T a, S b)
{
    return { a.x * b, a.y * b, a.z * b, a.w * b };
}

template <typename T>
__host__ __device__ __forceinline__ auto dot(T a, T b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

}

namespace quat
{

__host__ __device__ __forceinline__ float4 norm(float4 q)
{
    float d = vec4::dot(q, q);
    if (d <= 0.0f) return { 0, 0, 0, 1 };
    return vec4::mul(q, 1.0f / sqrtf(d));
}

__host__ __device__ __forceinline__ float4 conj(float4 q)
{
    return { -q.x, -q.y, -q.z, q.w };
}

__host__ __device__ __forceinline__ float4 comp(float4 a, float4 b)
{
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
        a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    };
}

__host__ __device__ __forceinline__ float4 toWorld(float4 v, float4 q)
{
    float4 t = vec3::mul(vec3::cross(q, v), 2.0f);
    float4 u = vec3::cross(q, t);
    return {
        v.x + q.w * t.x + u.x,
        v.y + q.w * t.y + u.y,
        v.z + q.w * t.z + u.z,
        v.w
    };
}

__host__ __device__ __forceinline__ float4 toLocal(float4 v, float4 q)
{
    return toWorld(v, conj(q));
}

}