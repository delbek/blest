#ifndef HBRSBFSKernel_CUH
#define HBRSBFSKernel_CUH

#include "BFSKernel.cuh"
#include "HBRS.cuh"

namespace HBRSBFSKernels
{
    __global__ void HBRSBFS8Normal( const unsigned* const __restrict__ sliceStartPtr,
                                    const unsigned* const __restrict__ noSliceSetsPtr,
                                    const unsigned* const __restrict__ sliceSetPtrs,
                                    const unsigned* const __restrict__ rowIds,
                                    const MASK* const __restrict__ masks,
                                    const MASK* const __restrict__ frontier,
                                    MASK* const __restrict__ visited,
                                    unsigned* const __restrict__ frontierNextSizePtr,
                                    MASK* const __restrict__ frontierNext)
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned noThreads = gridDim.x * blockDim.x;
        unsigned noWarps = noThreads / WARP_SIZE;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned sliceStart = *sliceStartPtr;
        unsigned noSliceSets = *noSliceSetsPtr;

        for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
        {
            unsigned normalized = sliceStart + sliceSet;
            unsigned shift = (normalized % 4) * 8;
            MASK origFragB = ((frontier[normalized / 4] >> shift) & 0x000000FF);
            if (origFragB)
            {
                unsigned tileStart = sliceSetPtrs[sliceSet] / 4;
                unsigned tileEnd = sliceSetPtrs[sliceSet + 1] / 4;
                for (unsigned t = tileStart; t < tileEnd; t += WARP_SIZE)
                {
                    unsigned tile = t + laneID;
                    uint4 rows = {0, 0, 0, 0};
                    MASK mask = 0;
                    if (tile < tileEnd)
                    {
                        rows = reinterpret_cast<const uint4*>(rowIds)[tile];
                        mask = masks[tile];
                    }

                    MASK fragA = (mask & 0x0000FFFF);
                    MASK fragB = 0;
                    if (laneID % 9 == 4)
                    {
                        fragB = origFragB << 8;
                    }
                    else if (laneID % 9 == 0)
                    {
                        fragB = origFragB;
                    }
                    unsigned fragC[2];
                    fragC[0] = fragC[1] = 0;
                    m8n8k128(fragC, fragA, fragB);
                    
                    if (fragC[0])
                    {
                        unsigned word = rows.x / MASK_BITS;
                        unsigned bit = rows.x % MASK_BITS;
                        MASK temp = (static_cast<MASK>(1) << bit);
                        MASK old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            old = atomicOr(&frontierNext[word], temp);
                            if ((old & 0x000000FF) == 0)
                            {
                                atomicAdd(frontierNextSizePtr, 1);
                            }
                        }
                    }
                    if (fragC[1])
                    {
                        unsigned word = rows.y / MASK_BITS;
                        unsigned bit = rows.y % MASK_BITS;
                        MASK temp = (static_cast<MASK>(1) << bit);
                        MASK old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            old = atomicOr(&frontierNext[word], temp);
                            if ((old & 0x0000FF00) == 0)
                            {
                                atomicAdd(frontierNextSizePtr, 1);
                            }
                        }
                    }

                    fragA = (mask & 0xFFFF0000);
                    fragB = 0;
                    if (laneID % 9 == 4)
                    {
                        fragB = origFragB << 24;
                    }
                    else if (laneID % 9 == 0)
                    {
                        fragB = origFragB << 16;
                    }
                    fragC[0] = fragC[1] = 0;
                    m8n8k128(fragC, fragA, fragB);

                    if (fragC[0])
                    {
                        unsigned word = rows.z / MASK_BITS;
                        unsigned bit = rows.z % MASK_BITS;
                        MASK temp = (static_cast<MASK>(1) << bit);
                        MASK old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            old = atomicOr(&frontierNext[word], temp);
                            if ((old & 0x00FF0000) == 0)
                            {
                                atomicAdd(frontierNextSizePtr, 1);
                            }
                        }
                    }
                    if (fragC[1])
                    {
                        unsigned word = rows.w / MASK_BITS;
                        unsigned bit = rows.w % MASK_BITS;
                        MASK temp = (static_cast<MASK>(1) << bit);
                        MASK old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            old = atomicOr(&frontierNext[word], temp);
                            if ((old & 0xFF000000) == 0)
                            {
                                atomicAdd(frontierNextSizePtr, 1);
                            }
                        }
                    }
                }
            }
        }
    }

    __global__ void HBRSBFS8Hub(    const unsigned* const __restrict__ noSliceSetsPtr,
                                    const unsigned* const __restrict__ sliceSetPtrs,
                                    const unsigned* const __restrict__ encodingPtrs,
                                    const unsigned* const __restrict__ rowIds,
                                    const MASK* const __restrict__ masks,
                                    const MASK* const __restrict__ frontier,
                                    MASK* const __restrict__ visited,
                                    unsigned* const __restrict__ frontierNextSizePtr,
                                    MASK* const __restrict__ frontierNext)
    {
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned noThreads = gridDim.x * blockDim.x;
        unsigned noWarps = noThreads / WARP_SIZE;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noSliceSets = *noSliceSetsPtr;

        for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
        {
            unsigned shift = (sliceSet % 4) * 8;
            MASK origFragB = ((frontier[sliceSet / 4] >> shift) & 0x000000FF);
            if (origFragB)
            {
                unsigned tileStart = sliceSetPtrs[sliceSet] / 4;
                unsigned tileEnd = sliceSetPtrs[sliceSet + 1] / 4;
                for (unsigned tile = tileStart; tile < tileEnd; tile += WARP_SIZE)
                {
                    unsigned encodingTile = (tile + laneID);
                    MASK mask = 0;
                    if (encodingTile < tileEnd)
                    {
                        mask = masks[encodingTile];
                    }

                    MASK fragA = (mask & 0x0000FFFF);
                    MASK fragB = 0;
                    if (laneID % 9 == 4)
                    {
                        fragB = origFragB << 8;
                    }
                    else if (laneID % 9 == 0)
                    {
                        fragB = origFragB;
                    }
                    unsigned fragC[2];
                    fragC[0] = fragC[1] = 0;
                    m8n8k128(fragC, fragA, fragB);
                    
                    if (fragC[0])
                    {
                        unsigned encoding = encodingTile * 4;
                        for (unsigned slice = encodingPtrs[encoding]; slice < encodingPtrs[encoding + 1]; ++slice)
                        {
                            unsigned row = rowIds[slice];
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                if ((old & 0x000000FF) == 0)
                                {
                                    atomicAdd(frontierNextSizePtr, 1);
                                }
                            }
                        }
                    }
                    if (fragC[1])
                    {
                        unsigned encoding = encodingTile * 4 + 1;
                        for (unsigned slice = encodingPtrs[encoding]; slice < encodingPtrs[encoding + 1]; ++slice)
                        {
                            unsigned row = rowIds[slice];
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                if ((old & 0x0000FF00) == 0)
                                {
                                    atomicAdd(frontierNextSizePtr, 1);
                                }
                            }
                        }
                    }

                    fragA = (mask & 0xFFFF0000);
                    fragB = 0;
                    if (laneID % 9 == 4)
                    {
                        fragB = origFragB << 24;
                    }
                    else if (laneID % 9 == 0)
                    {
                        fragB = origFragB << 16;
                    }
                    fragC[0] = fragC[1] = 0;
                    m8n8k128(fragC, fragA, fragB);

                    if (fragC[0])
                    {
                        unsigned encoding = encodingTile * 4 + 2;
                        for (unsigned slice = encodingPtrs[encoding]; slice < encodingPtrs[encoding + 1]; ++slice)
                        {
                            unsigned row = rowIds[slice];
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                if ((old & 0x00FF0000) == 0)
                                {
                                    atomicAdd(frontierNextSizePtr, 1);
                                }
                            }
                        }
                    }
                    if (fragC[1])
                    {
                        unsigned encoding = encodingTile * 4 + 3;
                        for (unsigned slice = encodingPtrs[encoding]; slice < encodingPtrs[encoding + 1]; ++slice)
                        {
                            unsigned row = rowIds[slice];
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                if ((old & 0xFF000000) == 0)
                                {
                                    atomicAdd(frontierNextSizePtr, 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    __global__ void HBRSBFS16Hub(   const unsigned* const __restrict__ noSliceSetsPtr,
                                    const unsigned* const __restrict__ sliceSetPtrs,
                                    const unsigned* const __restrict__ encodingPtrs,
                                    const unsigned* const __restrict__ rowIds,
                                    const MASK* const __restrict__ masks,
                                    const MASK* const __restrict__ frontier,
                                    MASK* const __restrict__ visited,
                                    unsigned* const __restrict__ frontierNextSizePtr,
                                    MASK* const __restrict__ frontierNext)
    {
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned noThreads = gridDim.x * blockDim.x;
        unsigned noWarps = noThreads / WARP_SIZE;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noSliceSets = *noSliceSetsPtr;

        for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
        {
            unsigned shift = (sliceSet % 2) * 16;
            MASK origFragB = ((frontier[sliceSet / 2] >> shift) & 0x0000FFFF);
            if (origFragB)
            {
                unsigned tileStart = sliceSetPtrs[sliceSet] / 2;
                unsigned tileEnd = sliceSetPtrs[sliceSet + 1] / 2;
                for (unsigned tile = tileStart; tile < tileEnd; tile += WARP_SIZE)
                {
                    unsigned encodingTile = (tile + laneID);
                    MASK fragA = 0;
                    if (encodingTile < tileEnd)
                    {
                        fragA = masks[encodingTile];
                    }

                    MASK fragB = 0;
                    if (laneID % 9 == 0)
                    {
                        fragB = origFragB;
                    }
                    if (laneID % 9 == 4)
                    {
                        fragB = origFragB << 16;
                    }
                    unsigned fragC[2];
                    fragC[0] = 0; fragC[1] = 0;
                    m8n8k128(fragC, fragA, fragB);
                    
                    if (fragC[0])
                    {
                        unsigned encoding = encodingTile * 2;
                        for (unsigned slice = encodingPtrs[encoding]; slice < encodingPtrs[encoding + 1]; ++slice)
                        {
                            unsigned row = rowIds[slice];
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                if ((old & 0x0000FFFF) == 0)
                                {
                                    atomicAdd(frontierNextSizePtr, 1);
                                }
                            }
                        }
                    }
                    if (fragC[1])
                    {
                        unsigned encoding = encodingTile * 2 + 1;
                        for (unsigned slice = encodingPtrs[encoding]; slice < encodingPtrs[encoding + 1]; ++slice)
                        {
                            unsigned row = rowIds[slice];
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                if ((old & 0xFFFF0000) == 0)
                                {
                                    atomicAdd(frontierNextSizePtr, 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    __global__ void HBRSBFS8NormalLoadBalanced( const unsigned* const __restrict__ sliceStartPtr,
                                                const unsigned* const __restrict__ noWarpsPtr,
                                                const unsigned* const __restrict__ warpPtrs,
                                                const unsigned* const __restrict__ sliceSetIds,
                                                const unsigned* const __restrict__ rowIds,
                                                const MASK* const __restrict__ masks,
                                                const MASK* const __restrict__ frontier,
                                                MASK* const __restrict__ visited,
                                                unsigned* const __restrict__ frontierNextSizePtr,
                                                MASK* const __restrict__ frontierNext)
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWarps = *noWarpsPtr;
        if (warpID >= noWarps) return;

        unsigned sliceStart = *sliceStartPtr;

        unsigned sliceSet = sliceSetIds[warpID];
        unsigned normalized = sliceStart + sliceSet;
        unsigned shift = (normalized % 4) * 8;
        MASK origFragB = ((frontier[normalized / 4] >> shift) & 0x000000FF);
        if (origFragB)
        {
            unsigned tileStart = warpPtrs[warpID] / 4;
            unsigned tileEnd = warpPtrs[warpID + 1] / 4;
            for (unsigned t = tileStart; t < tileEnd; t += WARP_SIZE)
            {
                unsigned tile = t + laneID;
                uint4 rows = {0, 0, 0, 0};
                MASK mask = 0;
                if (tile < tileEnd)
                {
                    rows = reinterpret_cast<const uint4*>(rowIds)[tile];
                    mask = masks[tile];
                }

                MASK fragA = (mask & 0x0000FFFF);
                MASK fragB = 0;
                if (laneID % 9 == 4)
                {
                    fragB = origFragB << 8;
                }
                else if (laneID % 9 == 0)
                {
                    fragB = origFragB;
                }
                unsigned fragC[2];
                fragC[0] = fragC[1] = 0;
                m8n8k128(fragC, fragA, fragB);
                
                if (fragC[0])
                {
                    unsigned word = rows.x / MASK_BITS;
                    unsigned bit = rows.x % MASK_BITS;
                    MASK temp = (static_cast<MASK>(1) << bit);
                    MASK old = atomicOr(&visited[word], temp);
                    if ((old & temp) == 0)
                    {
                        old = atomicOr(&frontierNext[word], temp);
                        if ((old & 0x000000FF) == 0)
                        {
                            atomicAdd(frontierNextSizePtr, 1);
                        }
                    }
                }
                if (fragC[1])
                {
                    unsigned word = rows.y / MASK_BITS;
                    unsigned bit = rows.y % MASK_BITS;
                    MASK temp = (static_cast<MASK>(1) << bit);
                    MASK old = atomicOr(&visited[word], temp);
                    if ((old & temp) == 0)
                    {
                        old = atomicOr(&frontierNext[word], temp);
                        if ((old & 0x0000FF00) == 0)
                        {
                            atomicAdd(frontierNextSizePtr, 1);
                        }
                    }
                }

                fragA = (mask & 0xFFFF0000);
                fragB = 0;
                if (laneID % 9 == 4)
                {
                    fragB = origFragB << 24;
                }
                else if (laneID % 9 == 0)
                {
                    fragB = origFragB << 16;
                }
                fragC[0] = fragC[1] = 0;
                m8n8k128(fragC, fragA, fragB);

                if (fragC[0])
                {
                    unsigned word = rows.z / MASK_BITS;
                    unsigned bit = rows.z % MASK_BITS;
                    MASK temp = (static_cast<MASK>(1) << bit);
                    MASK old = atomicOr(&visited[word], temp);
                    if ((old & temp) == 0)
                    {
                        old = atomicOr(&frontierNext[word], temp);
                        if ((old & 0x00FF0000) == 0)
                        {
                            atomicAdd(frontierNextSizePtr, 1);
                        }
                    }
                }
                if (fragC[1])
                {
                    unsigned word = rows.w / MASK_BITS;
                    unsigned bit = rows.w % MASK_BITS;
                    MASK temp = (static_cast<MASK>(1) << bit);
                    MASK old = atomicOr(&visited[word], temp);
                    if ((old & temp) == 0)
                    {
                        old = atomicOr(&frontierNext[word], temp);
                        if ((old & 0xFF000000) == 0)
                        {
                            atomicAdd(frontierNextSizePtr, 1);
                        }
                    }
                }
            }
        }
    }

    __global__ void HBRSBFS8HubLoadBalanced(    const unsigned* const __restrict__ noWarpsPtr,
                                                const unsigned* const __restrict__ warpPtrs,
                                                const unsigned* const __restrict__ sliceSetIds,
                                                unsigned* const __restrict__ encodingPtrs,
                                                const unsigned* const __restrict__ rowIds,
                                                const MASK* const __restrict__ masks,
                                                const MASK* const __restrict__ frontier,
                                                MASK* const __restrict__ visited,
                                                unsigned* const __restrict__ frontierNextSizePtr,
                                                MASK* const __restrict__ frontierNext)
    {
        auto warp = coalesced_threads();
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWarps = *noWarpsPtr;
        if (warpID >= noWarps) return;

        unsigned sliceSet = sliceSetIds[warpID];
        unsigned shift = (sliceSet % 4) * 8;
        MASK origFragB = ((frontier[sliceSet / 4] >> shift) & 0x000000FF);
        if (origFragB)
        {
            unsigned tileStart = warpPtrs[warpID] / 4;
            unsigned tileEnd = warpPtrs[warpID + 1] / 4;
            for (unsigned tile = tileStart; tile < tileEnd; tile += WARP_SIZE)
            {
                unsigned encodingTile = (tile + laneID);
                MASK mask = 0;
                if (encodingTile < tileEnd)
                {
                    mask = masks[encodingTile];
                }

                MASK fragA = (mask & 0x0000FFFF);
                MASK fragB = 0;
                if (laneID % 9 == 4)
                {
                    fragB = origFragB << 8;
                }
                else if (laneID % 9 == 0)
                {
                    fragB = origFragB;
                }
                unsigned fragC[2];
                fragC[0] = fragC[1] = 0;
                m8n8k128(fragC, fragA, fragB);

                fragA = (mask & 0xFFFF0000);
                fragB = 0;
                if (laneID % 9 == 4)
                {
                    fragB = origFragB << 24;
                }
                else if (laneID % 9 == 0)
                {
                    fragB = origFragB << 16;
                }
                unsigned fragCPrime[2];
                fragCPrime[0] = fragCPrime[1] = 0;
                m8n8k128(fragCPrime, fragA, fragB);

                for (unsigned lane = 0; lane < WARP_SIZE; ++lane)
                {
                    unsigned current = tile + lane;
                    if (warp.shfl(fragC[0], lane))
                    {
                        unsigned encoding = current * 4;
                        unsigned start = encodingPtrs[encoding];
                        unsigned set = start >> 31;
                        if (!set)
                        {
                            unsigned end = encodingPtrs[encoding + 1] & 0x7FFFFFFF;
                            for (unsigned slice = start + laneID; slice < end; slice += WARP_SIZE)
                            {
                                unsigned row = rowIds[slice];
                                unsigned word = row / MASK_BITS;
                                unsigned bit = row % MASK_BITS;
                                MASK temp = (static_cast<MASK>(1) << bit);
                                MASK old = atomicOr(&visited[word], temp);
                                if ((old & temp) == 0)
                                {
                                    old = atomicOr(&frontierNext[word], temp);
                                    if ((old & 0x000000FF) == 0)
                                    {
                                        atomicAdd(frontierNextSizePtr, 1);
                                    }
                                }
                            }
                            encodingPtrs[encoding] |= (static_cast<unsigned>(1) << 31);
                        }
                    }

                    if (warp.shfl(fragC[1], lane))
                    {
                        unsigned encoding = current * 4 + 1;
                        unsigned start = encodingPtrs[encoding];
                        unsigned set = start >> 31;
                        if (!set)
                        {
                            unsigned end = encodingPtrs[encoding + 1] & 0x7FFFFFFF;
                            for (unsigned slice = start + laneID; slice < end; slice += WARP_SIZE)
                            {
                                unsigned row = rowIds[slice];
                                unsigned word = row / MASK_BITS;
                                unsigned bit = row % MASK_BITS;
                                MASK temp = (static_cast<MASK>(1) << bit);
                                MASK old = atomicOr(&visited[word], temp);
                                if ((old & temp) == 0)
                                {
                                    old = atomicOr(&frontierNext[word], temp);
                                    if ((old & 0x0000FF00) == 0)
                                    {
                                        atomicAdd(frontierNextSizePtr, 1);
                                    }
                                }
                            }
                            encodingPtrs[encoding] |= (static_cast<unsigned>(1) << 31);
                        }
                    }

                    if (warp.shfl(fragCPrime[0], lane))
                    {
                        unsigned encoding = current * 4 + 2;
                        unsigned start = encodingPtrs[encoding];
                        unsigned set = start >> 31;
                        if (!set)
                        {
                            unsigned end = encodingPtrs[encoding + 1] & 0x7FFFFFFF;
                            for (unsigned slice = start + laneID; slice < end; slice += WARP_SIZE)
                            {
                                unsigned row = rowIds[slice];
                                unsigned word = row / MASK_BITS;
                                unsigned bit = row % MASK_BITS;
                                MASK temp = (static_cast<MASK>(1) << bit);
                                MASK old = atomicOr(&visited[word], temp);
                                if ((old & temp) == 0)
                                {
                                    old = atomicOr(&frontierNext[word], temp);
                                    if ((old & 0x00FF0000) == 0)
                                    {
                                        atomicAdd(frontierNextSizePtr, 1);
                                    }
                                }
                            }
                            encodingPtrs[encoding] |= (static_cast<unsigned>(1) << 31);
                        }
                    }

                    if (warp.shfl(fragCPrime[1], lane))
                    {
                        unsigned encoding = current * 4 + 3;
                        unsigned start = encodingPtrs[encoding];
                        unsigned set = start >> 31;
                        if (!set)
                        {
                            unsigned end = encodingPtrs[encoding + 1] & 0x7FFFFFFF;
                            for (unsigned slice = start + laneID; slice < end; slice += WARP_SIZE)
                            {
                                unsigned row = rowIds[slice];
                                unsigned word = row / MASK_BITS;
                                unsigned bit = row % MASK_BITS;
                                MASK temp = (static_cast<MASK>(1) << bit);
                                MASK old = atomicOr(&visited[word], temp);
                                if ((old & temp) == 0)
                                {
                                    old = atomicOr(&frontierNext[word], temp);
                                    if ((old & 0xFF000000) == 0)
                                    {
                                        atomicAdd(frontierNextSizePtr, 1);
                                    }
                                }
                            }
                            encodingPtrs[encoding] |= (static_cast<unsigned>(1) << 31);
                        }
                    }
                }
            }
        }
    }
};

class HBRSBFSKernel: public BFSKernel
{
public:
    HBRSBFSKernel(BitMatrix* matrix);
    HBRSBFSKernel(const HBRSBFSKernel& other) = delete;
    HBRSBFSKernel(HBRSBFSKernel&& other) noexcept = delete;
    HBRSBFSKernel& operator=(const HBRSBFSKernel& other) = delete;
    HBRSBFSKernel& operator=(HBRSBFSKernel&& other) noexcept = delete;
    virtual ~HBRSBFSKernel() = default;

    virtual double hostCode(unsigned sourceVertex) final;
    double hostCodeLoadImbalanced(unsigned sourceVertex);
    double hostCodeLoadBalanced(unsigned sourceVertex);

private:
    void generateBalancingArrays(unsigned& warpNumberNormal, unsigned*& warpPtrsNormal, unsigned*& sliceSetIdsNormal,
                                 unsigned& warpNumberHub   , unsigned*& warpPtrsHub   , unsigned*& sliceSetIdsHub);
};

HBRSBFSKernel::HBRSBFSKernel(BitMatrix* matrix)
: BFSKernel(matrix)
{

}

double HBRSBFSKernel::hostCode(unsigned sourceVertex)
{
    return this->hostCodeLoadBalanced(sourceVertex);
}

double HBRSBFSKernel::hostCodeLoadImbalanced(unsigned sourceVertex)
{
    HBRS* hbrs = dynamic_cast<HBRS*>(matrix);

    // Normal
    unsigned n = hbrs->getN();
    unsigned sliceSize = hbrs->getSliceSize();
    unsigned noSliceSets = hbrs->getNoSliceSets();
    unsigned* sliceSetPtrs = hbrs->getSliceSetPtrs();
    unsigned* rowIds = hbrs->getRowIds();
    MASK* masks = hbrs->getMasks();

    unsigned noSlices = sliceSetPtrs[noSliceSets];
    unsigned noMasks = MASK_BITS / sliceSize;

    // Hub
    unsigned n_Hub = hbrs->getN_Hub();
    unsigned sliceSize_Hub = hbrs->getSliceSize_Hub();
    unsigned noSliceSets_Hub = hbrs->getNoSliceSets_Hub();
    unsigned* sliceSetPtrs_Hub = hbrs->getSliceSetPtrs_Hub();
    unsigned* encodingPtrs_Hub = hbrs->getEncodingPtrs_Hub();
    unsigned* rowIds_Hub = hbrs->getRowIds_Hub();
    MASK* masks_Hub = hbrs->getMasks_Hub();

    unsigned noEncodings_Hub = sliceSetPtrs_Hub[noSliceSets_Hub];
    unsigned noSlices_Hub = encodingPtrs_Hub[sliceSetPtrs_Hub[noSliceSets_Hub]];
    unsigned noMasks_Hub = MASK_BITS / sliceSize_Hub;

    void* hubKernelPtr;
    void* normalKernelPtr;
    if (sliceSize_Hub == 8)
    {
        hubKernelPtr = (void*)HBRSBFSKernels::HBRSBFS8Hub;
    }
    else if (sliceSize_Hub == 16)
    {
        hubKernelPtr = (void*)HBRSBFSKernels::HBRSBFS16Hub;
    }
    else
    {
        throw std::runtime_error("No appropriate hub kernel found matching your slice size.");
    }
    
    if (sliceSize == 8)
    {
        normalKernelPtr= (void*)HBRSBFSKernels::HBRSBFS8Normal;
    }
    else
    {
        throw std::runtime_error("No appropriate normal kernel found matching your slice size.");
    }

    int gridSizeNormal, blockSizeNormal;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(
                                                &gridSizeNormal, 
                                                &blockSizeNormal, 
                                                normalKernelPtr,
                                                0,
                                                0));


    int gridSizeHub, blockSizeHub;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(
                                                &gridSizeHub, 
                                                &blockSizeHub, 
                                                hubKernelPtr,
                                                0,
                                                0));

    unsigned total = n + n_Hub;
    unsigned factor = sliceSize_Hub / sliceSize;
    unsigned normalStart = noSliceSets_Hub * factor;

    // Normal
    unsigned* d_NormalStart;
    unsigned* d_NoSliceSets;
    unsigned* d_SliceSetPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    gpuErrchk(cudaMalloc(&d_NormalStart, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_NormalStart, &normalStart, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // Hub
    unsigned* d_NoSliceSets_Hub;
    unsigned* d_SliceSetPtrs_Hub;
    unsigned* d_EncodingPtrs_Hub;
    unsigned* d_RowIds_Hub;
    MASK* d_Masks_Hub;

    gpuErrchk(cudaMalloc(&d_NoSliceSets_Hub, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs_Hub, sizeof(unsigned) * (noSliceSets_Hub + 1)))
    gpuErrchk(cudaMalloc(&d_EncodingPtrs_Hub, sizeof(unsigned) * (noEncodings_Hub + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds_Hub, sizeof(unsigned) * noSlices_Hub))
    gpuErrchk(cudaMalloc(&d_Masks_Hub, sizeof(MASK) * (noEncodings_Hub / noMasks_Hub)))

    gpuErrchk(cudaMemcpy(d_NoSliceSets_Hub, &noSliceSets_Hub, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs_Hub, sliceSetPtrs_Hub, sizeof(unsigned) * (noSliceSets_Hub + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_EncodingPtrs_Hub, encodingPtrs_Hub, sizeof(unsigned) * (noEncodings_Hub + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds_Hub, rowIds_Hub, sizeof(unsigned) * noSlices_Hub, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks_Hub, masks_Hub, sizeof(MASK) * (noEncodings_Hub / noMasks_Hub), cudaMemcpyHostToDevice))

    // Algorithm
    MASK* d_Frontier;
    MASK* d_Visited;
    unsigned* d_FrontierNextSize;
    MASK* d_FrontierNext;

    unsigned noWords = (total + MASK_BITS - 1) / MASK_BITS;
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(MASK) * noWords))

    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(MASK) * noWords))

    unsigned word = sourceVertex / (MASK_BITS);
    unsigned bit = sourceVertex % (MASK_BITS);
    MASK temp = (static_cast<MASK>(1) << bit);
    gpuErrchk(cudaMemcpy(d_Frontier + word, &temp, sizeof(MASK), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Visited + word, &temp, sizeof(MASK), cudaMemcpyHostToDevice))

    cudaStream_t streamHub, streamNormal;
    cudaStreamCreate(&streamHub);
    cudaStreamCreate(&streamNormal);

    gridSizeHub = std::min(unsigned(gridSizeHub), noSliceSets_Hub);
    blockSizeHub = noSliceSets_Hub / gridSizeHub;
    for (; blockSizeHub % WARP_SIZE != 0; ++blockSizeHub);
    blockSizeHub = std::min(1024, blockSizeHub);
    
    unsigned frontierSize = 1;
    unsigned totalVisited = frontierSize;
    double start = omp_get_wtime();
    while (frontierSize != 0)
    {
        void *hubArgs[] = 
        {
            &d_NoSliceSets_Hub,
            &d_SliceSetPtrs_Hub,
            &d_EncodingPtrs_Hub,
            &d_RowIds_Hub,
            &d_Masks_Hub,
            &d_Frontier,
            &d_Visited,
            &d_FrontierNextSize,
            &d_FrontierNext
        };

        void *normalArgs[] = 
        {
            &d_NormalStart,
            &d_NoSliceSets,
            &d_SliceSetPtrs,
            &d_RowIds,
            &d_Masks,
            &d_Frontier,
            &d_Visited,
            &d_FrontierNextSize,
            &d_FrontierNext
        };
        
        cudaLaunchKernel(
            hubKernelPtr,
            gridSizeHub,
            blockSizeHub,
            hubArgs,
            0,
            streamHub
        );
        
        cudaLaunchKernel(
            normalKernelPtr,
            gridSizeNormal,
            blockSizeNormal,
            normalArgs,
            0,
            streamNormal
        );

        cudaStreamSynchronize(streamHub);
        cudaStreamSynchronize(streamNormal);

        gpuErrchk(cudaMemcpy(&frontierSize, d_FrontierNextSize, sizeof(unsigned), cudaMemcpyDeviceToHost))
        std::swap(d_Frontier, d_FrontierNext);
        gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
        gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(MASK) * noWords))
        totalVisited += frontierSize;
    }
    double end = omp_get_wtime();
    //std::cout << "Total traversed vertex count: " << totalVisited << std::endl;

    cudaStreamDestroy(streamHub);
    cudaStreamDestroy(streamNormal);

    // Normal
    gpuErrchk(cudaFree(d_NormalStart))
    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))

    // Hub
    gpuErrchk(cudaFree(d_NoSliceSets_Hub))
    gpuErrchk(cudaFree(d_SliceSetPtrs_Hub))
    gpuErrchk(cudaFree(d_EncodingPtrs_Hub))
    gpuErrchk(cudaFree(d_RowIds_Hub))
    gpuErrchk(cudaFree(d_Masks_Hub))

    // Algorithm
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_FrontierNext))

    return (end - start);
}

void HBRSBFSKernel::generateBalancingArrays(unsigned& warpNumberNormal, unsigned*& warpPtrsNormal, unsigned*& sliceSetIdsNormal,
                                            unsigned& warpNumberHub   , unsigned*& warpPtrsHub   , unsigned*& sliceSetIdsHub)
{
    HBRS* hbrs = dynamic_cast<HBRS*>(matrix);

    // Normal
    unsigned n = hbrs->getN();
    unsigned sliceSize = hbrs->getSliceSize();
    unsigned noSliceSets = hbrs->getNoSliceSets();
    unsigned* sliceSetPtrs = hbrs->getSliceSetPtrs();
    unsigned* rowIds = hbrs->getRowIds();
    MASK* masks = hbrs->getMasks();

    unsigned noSlices = sliceSetPtrs[noSliceSets];
    unsigned noMasks = MASK_BITS / sliceSize;

    // Hub
    unsigned n_Hub = hbrs->getN_Hub();
    unsigned sliceSize_Hub = hbrs->getSliceSize_Hub();
    unsigned noSliceSets_Hub = hbrs->getNoSliceSets_Hub();
    unsigned* sliceSetPtrs_Hub = hbrs->getSliceSetPtrs_Hub();
    unsigned* encodingPtrs_Hub = hbrs->getEncodingPtrs_Hub();
    unsigned* rowIds_Hub = hbrs->getRowIds_Hub();
    MASK* masks_Hub = hbrs->getMasks_Hub();

    unsigned noEncodings_Hub = sliceSetPtrs_Hub[noSliceSets_Hub];
    unsigned noSlices_Hub = encodingPtrs_Hub[sliceSetPtrs_Hub[noSliceSets_Hub]];
    unsigned noMasks_Hub = MASK_BITS / sliceSize_Hub;

    unsigned totalWork = noEncodings_Hub + noSlices;
    unsigned workPerWarp = WARP_SIZE * noMasks;
    assert(totalWork % workPerWarp == 0);

    // Normal
    std::vector<unsigned> ptrs;
    ptrs.emplace_back(0);
    std::vector<unsigned> ids;
    for (unsigned sliceSet = 0; sliceSet < noSliceSets; ++sliceSet)
    {
        unsigned sliceCountThisSet = sliceSetPtrs[sliceSet + 1] - sliceSetPtrs[sliceSet];
        if (sliceCountThisSet == 0)
        {
            continue;
        }
        for (unsigned offs = 0; offs < sliceCountThisSet; offs += workPerWarp)
        {
            unsigned assign = std::min(workPerWarp, sliceCountThisSet - offs);
            ptrs.emplace_back(ptrs.back() + assign);
            ids.emplace_back(sliceSet);
        }
    }

    warpNumberNormal = ids.size();
    warpPtrsNormal = new unsigned[warpNumberNormal + 1];
    std::copy(ptrs.begin(), ptrs.end(), warpPtrsNormal);
    sliceSetIdsNormal = new unsigned[warpNumberNormal];
    std::copy(ids.begin(), ids.end(), sliceSetIdsNormal);

    // Hub
    ptrs.clear();
    ptrs.emplace_back(0);
    ids.clear();
    for (unsigned sliceSet = 0; sliceSet < noSliceSets_Hub; ++sliceSet)
    {
        unsigned encodingCountThisSet = sliceSetPtrs_Hub[sliceSet + 1] - sliceSetPtrs_Hub[sliceSet];
        if (encodingCountThisSet == 0)
        {
            continue;
        }
        for (unsigned offs = 0; offs < encodingCountThisSet; offs += workPerWarp) 
        {
            unsigned assign = std::min(workPerWarp, encodingCountThisSet - offs);
            ptrs.emplace_back(ptrs.back() + assign);
            ids.emplace_back(sliceSet);
        }
    }

    warpNumberHub = ids.size();
    warpPtrsHub = new unsigned[warpNumberHub + 1];
    std::copy(ptrs.begin(), ptrs.end(), warpPtrsHub);
    sliceSetIdsHub = new unsigned[warpNumberHub];
    std::copy(ids.begin(), ids.end(), sliceSetIdsHub);
}

double HBRSBFSKernel::hostCodeLoadBalanced(unsigned sourceVertex)
{
    HBRS* hbrs = dynamic_cast<HBRS*>(matrix);

    // Normal
    unsigned n = hbrs->getN();
    unsigned sliceSize = hbrs->getSliceSize();
    unsigned noSliceSets = hbrs->getNoSliceSets();
    unsigned* sliceSetPtrs = hbrs->getSliceSetPtrs();
    unsigned warpNumber;
    unsigned* warpPtrs;
    unsigned* sliceSetIds;
    unsigned* rowIds = hbrs->getRowIds();
    MASK* masks = hbrs->getMasks();

    unsigned noSlices = sliceSetPtrs[noSliceSets];
    unsigned noMasks = MASK_BITS / sliceSize;

    // Hub
    unsigned n_Hub = hbrs->getN_Hub();
    unsigned sliceSize_Hub = hbrs->getSliceSize_Hub();
    unsigned noSliceSets_Hub = hbrs->getNoSliceSets_Hub();
    unsigned* sliceSetPtrs_Hub = hbrs->getSliceSetPtrs_Hub();
    unsigned warpNumber_Hub;
    unsigned* warpPtrs_Hub;
    unsigned* sliceSetIds_Hub;
    unsigned* encodingPtrs_Hub = hbrs->getEncodingPtrs_Hub();
    unsigned* rowIds_Hub = hbrs->getRowIds_Hub();
    MASK* masks_Hub = hbrs->getMasks_Hub();

    unsigned noEncodings_Hub = sliceSetPtrs_Hub[noSliceSets_Hub];
    unsigned noSlices_Hub = encodingPtrs_Hub[sliceSetPtrs_Hub[noSliceSets_Hub]];
    unsigned noMasks_Hub = MASK_BITS / sliceSize_Hub;

    this->generateBalancingArrays(warpNumber    , warpPtrs    , sliceSetIds, 
                                  warpNumber_Hub, warpPtrs_Hub, sliceSetIds_Hub);

    void* hubKernelPtr;
    void* normalKernelPtr;
    if (sliceSize_Hub == 8)
    {
        hubKernelPtr = (void*)HBRSBFSKernels::HBRSBFS8HubLoadBalanced;
    }
    else
    {
        throw std::runtime_error("No appropriate hub kernel found matching your slice size.");
    }
    
    if (sliceSize == 8)
    {
        normalKernelPtr= (void*)HBRSBFSKernels::HBRSBFS8NormalLoadBalanced;
    }
    else
    {
        throw std::runtime_error("No appropriate normal kernel found matching your slice size.");
    }

    unsigned total = n + n_Hub;
    unsigned factor = sliceSize_Hub / sliceSize;
    unsigned normalStart = noSliceSets_Hub * factor;

    // Normal
    unsigned* d_NormalStart;
    unsigned* d_NoWarps;
    unsigned* d_WarpPtrs;
    unsigned* d_SliceSetIds;
    unsigned* d_RowIds;
    MASK* d_Masks;

    gpuErrchk(cudaMalloc(&d_NormalStart, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_NoWarps, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_WarpPtrs, sizeof(unsigned) * (warpNumber + 1)))
    gpuErrchk(cudaMalloc(&d_SliceSetIds, sizeof(unsigned) * warpNumber))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_NormalStart, &normalStart, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoWarps, &warpNumber, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_WarpPtrs, warpPtrs, sizeof(unsigned) * (warpNumber + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetIds, sliceSetIds, sizeof(unsigned) * warpNumber, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // Hub
    unsigned* d_NoWarps_Hub;
    unsigned* d_WarpPtrs_Hub;
    unsigned* d_SliceSetIds_Hub;
    unsigned* d_EncodingPtrs_Hub;
    unsigned* d_RowIds_Hub;
    MASK* d_Masks_Hub;

    gpuErrchk(cudaMalloc(&d_NoWarps_Hub, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_WarpPtrs_Hub, sizeof(unsigned) * (warpNumber_Hub + 1)))
    gpuErrchk(cudaMalloc(&d_SliceSetIds_Hub, sizeof(unsigned) * warpNumber_Hub))
    gpuErrchk(cudaMalloc(&d_EncodingPtrs_Hub, sizeof(unsigned) * (noEncodings_Hub + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds_Hub, sizeof(unsigned) * noSlices_Hub))
    gpuErrchk(cudaMalloc(&d_Masks_Hub, sizeof(MASK) * (noEncodings_Hub / noMasks_Hub)))

    gpuErrchk(cudaMemcpy(d_NoWarps_Hub, &warpNumber_Hub, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_WarpPtrs_Hub, warpPtrs_Hub, sizeof(unsigned) * (warpNumber_Hub + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetIds_Hub, sliceSetIds_Hub, sizeof(unsigned) * warpNumber_Hub, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_EncodingPtrs_Hub, encodingPtrs_Hub, sizeof(unsigned) * (noEncodings_Hub + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds_Hub, rowIds_Hub, sizeof(unsigned) * noSlices_Hub, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks_Hub, masks_Hub, sizeof(MASK) * (noEncodings_Hub / noMasks_Hub), cudaMemcpyHostToDevice))

    // Algorithm
    MASK* d_Frontier;
    MASK* d_Visited;
    unsigned* d_FrontierNextSize;
    MASK* d_FrontierNext;

    unsigned noWords = (total + MASK_BITS - 1) / MASK_BITS;
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(MASK) * noWords))

    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(MASK) * noWords))

    unsigned word = sourceVertex / (MASK_BITS);
    unsigned bit = sourceVertex % (MASK_BITS);
    MASK temp = (static_cast<MASK>(1) << bit);
    gpuErrchk(cudaMemcpy(d_Frontier + word, &temp, sizeof(MASK), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Visited + word, &temp, sizeof(MASK), cudaMemcpyHostToDevice))

    cudaStream_t streamHub, streamNormal;
    cudaStreamCreate(&streamHub);
    cudaStreamCreate(&streamNormal);

    const unsigned totalThreadsHub = warpNumber_Hub * WARP_SIZE;
    const unsigned totalThreadsNormal = warpNumber * WARP_SIZE;

    const unsigned BLOCK_SIZE = 256;
    unsigned gridHub = (totalThreadsHub + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned gridNormal = (totalThreadsNormal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    unsigned frontierSize = 1;
    unsigned totalVisited = frontierSize;
    double start = omp_get_wtime();
    while (frontierSize != 0)
    {
        void *hubArgs[] = 
        {
            &d_NoWarps_Hub,
            &d_WarpPtrs_Hub,
            &d_SliceSetIds_Hub,
            &d_EncodingPtrs_Hub,
            &d_RowIds_Hub,
            &d_Masks_Hub,
            &d_Frontier,
            &d_Visited,
            &d_FrontierNextSize,
            &d_FrontierNext
        };

        void *normalArgs[] = 
        {
            &d_NormalStart,
            &d_NoWarps,
            &d_WarpPtrs,
            &d_SliceSetIds,
            &d_RowIds,
            &d_Masks,
            &d_Frontier,
            &d_Visited,
            &d_FrontierNextSize,
            &d_FrontierNext
        };
        
        cudaLaunchKernel(
            hubKernelPtr,
            gridHub,
            BLOCK_SIZE,
            hubArgs,
            0,
            streamHub
        );
        
        cudaLaunchKernel(
            normalKernelPtr,
            gridNormal,
            BLOCK_SIZE,
            normalArgs,
            0,
            streamNormal
        );

        cudaStreamSynchronize(streamHub);
        cudaStreamSynchronize(streamNormal);

        gpuErrchk(cudaMemcpy(&frontierSize, d_FrontierNextSize, sizeof(unsigned), cudaMemcpyDeviceToHost))
        std::swap(d_Frontier, d_FrontierNext);
        gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
        gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(MASK) * noWords))
        totalVisited += frontierSize;
    }
    double end = omp_get_wtime();
    //std::cout << "Total traversed vertex count: " << totalVisited << std::endl;

    cudaStreamDestroy(streamHub);
    cudaStreamDestroy(streamNormal);

    // Normal
    gpuErrchk(cudaFree(d_NormalStart))
    gpuErrchk(cudaFree(d_NoWarps))
    gpuErrchk(cudaFree(d_WarpPtrs))
    gpuErrchk(cudaFree(d_SliceSetIds))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))

    delete[] warpPtrs;
    delete[] sliceSetIds;

    // Hub
    gpuErrchk(cudaFree(d_NoWarps_Hub))
    gpuErrchk(cudaFree(d_WarpPtrs_Hub))
    gpuErrchk(cudaFree(d_SliceSetIds_Hub))
    gpuErrchk(cudaFree(d_EncodingPtrs_Hub))
    gpuErrchk(cudaFree(d_RowIds_Hub))
    gpuErrchk(cudaFree(d_Masks_Hub))

    delete[] warpPtrs_Hub;
    delete[] sliceSetIds_Hub;

    // Algorithm
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_FrontierNext))

    return (end - start);
}

#endif
