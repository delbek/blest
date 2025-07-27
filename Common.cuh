#ifndef COMMON_CUH
#define COMMON_CUH

#include <iostream>
#include "omp.h"

#define K 128
#define MASK unsigned
const unsigned MASK_BITS = sizeof(MASK) * 8;
#define UNSIGNED_MAX 4294967295U

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
    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %4, %5 };"
        : "+r"(fragC[0]), "+r"(fragC[1])
        : "r"(fragA), "r"(fragB), "r"(fragC[0]), "r"(fragC[1]));
}

#endif
