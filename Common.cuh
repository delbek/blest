#pragma once

#include <iostream>
#include "omp.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <limits>
#include <fstream>
#include <type_traits>
#include <cuda/pipeline>
using namespace cooperative_groups;

#define ROAD_NETWORK_DEGREE 4
#define ROAD_NETWORK_FUSION_CONSTANT 1
#define SOCIAL_NETWORK_FUSION_CONSTANT 9
#define DIRECTION_SWITCHING_CONSTANT 1

#define M 8
#define K 128
#define MASK unsigned
constexpr unsigned MASK_BITS = sizeof(MASK) * 8;
constexpr unsigned UNSIGNED_BITS = sizeof(unsigned) * 8;
#define WARP_SIZE 32
#define UNSIGNED_MAX 4294967295

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ __forceinline__ void m8n8k128(unsigned* const __restrict__ fragC, const unsigned& fragA, const unsigned& fragB)
{
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"
                 " { %0, %1 }, "
                 " { %2 }, "
                 " { %3 }, "
                 " { %4, %5 };"
                 : "+r"(fragC[0]), "+r"(fragC[1])
                 : "r"(fragA), "r"(fragB), "r"(fragC[0]), "r"(fragC[1]));
}

template <typename T>
__device__ __forceinline__ void load_l1bypass(const T* const __restrict__ ptr, T& val)
{
    static_assert(sizeof(T) == 4);
    asm volatile("ld.global.cg.b32 %0, [%1];"
                 : "=r"(val)
                 : "l"(ptr)
                 : "memory");
}

__device__ __forceinline__ void loadMask_streaming(const MASK* const __restrict__ ptr, MASK& val)
{
    asm volatile("ld.global.cs.b32 %0, [%1];"
                 : "=r"(val)
                 : "l"(ptr)
                 : "memory");
}

__device__ __forceinline__ void loadRow4Ids_streaming(const uint4* const __restrict__ ptr, uint4& val)
{
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(ptr)
                 : "memory");
}

template <class T>
void fileFlush(std::ofstream& file, T el)
{
    file << "'" << el << '\t';
}

unsigned* chainPermutations(unsigned n, unsigned* perm1, unsigned* perm2)
{
    unsigned* chained = new unsigned[n];
    for (unsigned i = 0; i < n; ++i)
    {
        chained[i] = perm2[perm1[i]];
    }
    delete[] perm1;
    delete[] perm2;
    return chained;
}
