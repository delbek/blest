#ifndef COMMON_CUH
#define COMMON_CUH

#include <iostream>
#include "omp.h"

#define K 128
#define MASK unsigned
const unsigned MASK_BITS = sizeof(MASK) * 8;
#define UNSIGNED_MAX 4294967295U

__device__ __forceinline__ void mma_m8n8k128(unsigned* fragC, const unsigned& fragA, const unsigned& fragB)
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
