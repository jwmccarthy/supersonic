#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <boost/pfr.hpp>
#include <cuda_runtime.h>

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)

// General error checking for CUDA memory operations
inline void check(cudaError_t err, const char *const func,
                  const char *const file, const int line)
{
    if (err == cudaSuccess) return;
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
}

template <typename T>
inline void cudaMallocCpy(T*& dst, const T* src, size_t n = 1)
{
    CUDA_CHECK(cudaMalloc(&dst, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
inline void cudaMallocSOA(T& s, std::size_t n)
{
    boost::pfr::for_each_field(s, [n](auto& field)
    {
        using F = std::remove_reference_t<decltype(field)>;

        if constexpr (std::is_pointer_v<F>)
        {
            using P = std::remove_pointer_t<F>;

            if (field == nullptr)
            {
                CUDA_CHECK(cudaMalloc(&field, n * sizeof(P)));
            }
        }
    });
}

template <class T>
inline void cudaMallocSOA(T& s, std::initializer_list<int> n)
{
    auto it = n.begin();

    boost::pfr::for_each_field(s, [n, it](auto& field) mutable
    {
        using F = std::remove_reference_t<decltype(field)>;

        if constexpr (std::is_pointer_v<F>)
        {
            assert(it != n.end() && "No enough sizes provided");

            using P = std::remove_pointer_t<F>;

            if (field == nullptr)
            {
                CUDA_CHECK(cudaMalloc(&field, (*it++) * sizeof(P)));
            }
        }
    });
}

template <class T>
inline void cudaFreeSOA(T& s)
{
    boost::pfr::for_each_field(s, [](auto& field)
    {
        using F = std::remove_reference_t<decltype(field)>;

        if constexpr (std::is_pointer_v<F>)
        {
            using P = std::remove_pointer_t<F>;

            if (field != nullptr)
            {
                CUDA_CHECK(cudaFree(field));
            }
        }
    });
}
