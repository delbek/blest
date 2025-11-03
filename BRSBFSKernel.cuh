#pragma once

#include "BFSKernel.cuh"
#include "BRS.cuh"

namespace BRSBFSKernels
{
    template<typename T>
    __device__ __forceinline__ void swap(T* __restrict__& ptr1, T* __restrict__& ptr2)
    {
        T* temp = ptr2;
        ptr2 = ptr1;
        ptr1 = temp;
    }

    __global__ void BRSBFS8EnhancedNoMasks2FullPad( const unsigned* const __restrict__ noSliceSetsPtr,
                                                    const unsigned* const __restrict__ sliceSetPtrs,
                                                    const unsigned* const __restrict__ virtualToReal,
                                                    const unsigned* const __restrict__ realPtrs,
                                                    const unsigned* const __restrict__ rowIds,
                                                    const MASK* const __restrict__ masks,
                                                    const unsigned* const __restrict__ noWordsPtr,
                                                    const unsigned* const __restrict__ directionThresholdPtr,
                                                    // current
                                                    unsigned* __restrict__ frontier,
                                                    unsigned* __restrict__ sparseFrontierIds,
                                                    unsigned* __restrict__ frontierCurrentSizePtr,
                                                    // next
                                                    unsigned* __restrict__ frontierNext,
                                                    unsigned* __restrict__ sparseFrontierNextIds,
                                                    unsigned* __restrict__ frontierNextSizePtr,
                                                    //
                                                    unsigned* const __restrict__ visited,
                                                    unsigned* const __restrict__ totalLevels,
                                                    unsigned* const __restrict__ levels
                                                    )
    {
        // MASK_BITS must be 16 BITS!

        auto warp = coalesced_threads();
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned noWords = *noWordsPtr;
        const unsigned DIRECTION_THRESHOLD = *directionThresholdPtr;
        const unsigned noSliceSets = *noSliceSetsPtr;
        unsigned levelCount = 0;

        const uint2* row2Ids = reinterpret_cast<const uint2*>(rowIds);

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            if (currentFrontierSize < DIRECTION_THRESHOLD) // spspmv
            {
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned vset = sparseFrontierIds[i];
                    unsigned rset = virtualToReal[vset];

                    unsigned shift = (rset % 4) << 3;
                    MASK origFragB = ((frontier[rset >> 2] >> shift) & 0x00FF);

                    unsigned tile = (vset << 5) + laneID; // slice: (vset * 64 + laneID * 2) - tile: slice / 2 = (vset * 32 + laneID)
                    uint2 rows = row2Ids[tile];
                    MASK fragA = masks[tile];

                    MASK fragB = 0;
                    if (laneID % 9 == 0 || laneID % 9 == 4)
                    {
                        fragB = (laneID % 9 == 0) ? (origFragB) : (origFragB << 8);
                    }
                    unsigned fragC[2];
                    fragC[0] = fragC[1] = 0;
                    m8n8k128(fragC, fragA, fragB);
                    
                    if (fragC[0])
                    {
                        unsigned word = rows.x / UNSIGNED_BITS;
                        unsigned bit = rows.x % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        unsigned old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            levels[rows.x] = levelCount;
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = (0xFF) << (sliceIdx << 3);
                            if ((old & sliceMask) == 0)
                            {
                                unsigned rss = rows.x >> 3;
                                unsigned start = realPtrs[rss];
                                unsigned end = realPtrs[rss + 1];
                                unsigned size = end - start;
                                unsigned loc = atomicAdd(frontierNextSizePtr, size);
                                for (unsigned vset = start; vset < end; ++vset)
                                {
                                    sparseFrontierNextIds[loc++] = vset;
                                }
                            }
                        }
                    }
                    if (fragC[1])
                    {
                        unsigned word = rows.y / UNSIGNED_BITS;
                        unsigned bit = rows.y % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        unsigned old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            levels[rows.y] = levelCount;
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = (0xFF) << (sliceIdx << 3);
                            if ((old & sliceMask) == 0)
                            {
                                unsigned rss = rows.y >> 3;
                                unsigned start = realPtrs[rss];
                                unsigned end = realPtrs[rss + 1];
                                unsigned size = end - start;
                                unsigned loc = atomicAdd(frontierNextSizePtr, size);
                                for (unsigned vset = start; vset < end; ++vset)
                                {
                                    sparseFrontierNextIds[loc++] = vset;
                                }
                            }
                        }
                    }
                }
            }
            else // spmv
            {
                for (unsigned vset = warpID; vset < noSliceSets; vset += noWarps)
                {
                    unsigned rset = virtualToReal[vset];
                    unsigned shift = (rset % 4) << 3;
                    MASK origFragB = ((frontier[rset >> 2] >> shift) & 0x00FF);
                    if (origFragB)
                    {
                        unsigned tile = (vset << 5) + laneID;
                        uint2 rows = row2Ids[tile];
                        MASK fragA = masks[tile];
                        
                        MASK fragB = 0;
                        if (laneID % 9 == 0 || laneID % 9 == 4)
                        {
                            fragB = (laneID % 9 == 0) ? (origFragB) : (origFragB << 8);
                        }
                        unsigned fragC[2];
                        fragC[0] = fragC[1] = 0;
                        m8n8k128(fragC, fragA, fragB);
                        
                        if (fragC[0])
                        {
                            unsigned word = rows.x / UNSIGNED_BITS;
                            unsigned bit = rows.x % UNSIGNED_BITS;
                            unsigned temp = (1 << bit);
                            unsigned old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                levels[rows.x] = levelCount;
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = (0xFF) << (sliceIdx << 3);
                                if ((old & sliceMask) == 0)
                                {
                                    unsigned rss = rows.x >> 3;
                                    unsigned start = realPtrs[rss];
                                    unsigned end = realPtrs[rss + 1];
                                    unsigned size = end - start;
                                    unsigned loc = atomicAdd(frontierNextSizePtr, size);
                                    for (unsigned vset = start; vset < end; ++vset)
                                    {
                                        sparseFrontierNextIds[loc++] = vset;
                                    }
                                }
                            }
                        }
                        if (fragC[1])
                        {
                            unsigned word = rows.y / UNSIGNED_BITS;
                            unsigned bit = rows.y % UNSIGNED_BITS;
                            unsigned temp = (1 << bit);
                            unsigned old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                levels[rows.y] = levelCount;
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = (0xFF) << (sliceIdx << 3);
                                if ((old & sliceMask) == 0)
                                {
                                    unsigned rss = rows.y >> 3;
                                    unsigned start = realPtrs[rss];
                                    unsigned end = realPtrs[rss + 1];
                                    unsigned size = end - start;
                                    unsigned loc = atomicAdd(frontierNextSizePtr, size);
                                    for (unsigned vset = start; vset < end; ++vset)
                                    {
                                        sparseFrontierNextIds[loc++] = vset;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(frontier, frontierNext);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            grid.sync();
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            if (cont)
            {
                for (unsigned i = threadID; i < noWords; i += noThreads)
                {
                    frontierNext[i] = 0;
                }
            }
            grid.sync();
        }
        if (threadID == 0)
        {
            *totalLevels = levelCount;
        }
    }

    __global__ void BRSBFS8EnhancedNoMasks2(    const unsigned* const __restrict__ noSliceSetsPtr,
                                                const unsigned* const __restrict__ sliceSetPtrs,
                                                const unsigned* const __restrict__ virtualToReal,
                                                const unsigned* const __restrict__ realPtrs,
                                                const unsigned* const __restrict__ rowIds,
                                                const MASK* const __restrict__ masks,
                                                const unsigned* const __restrict__ noWordsPtr,
                                                const unsigned* const __restrict__ directionThresholdPtr,
                                                // current
                                                unsigned* __restrict__ frontier,
                                                unsigned* __restrict__ sparseFrontierIds,
                                                unsigned* __restrict__ frontierCurrentSizePtr,
                                                // next
                                                unsigned* __restrict__ frontierNext,
                                                unsigned* __restrict__ sparseFrontierNextIds,
                                                unsigned* __restrict__ frontierNextSizePtr,
                                                //
                                                unsigned* const __restrict__ visited,
                                                unsigned* const __restrict__ totalLevels,
                                                unsigned* const __restrict__ levels
                                                )
    {
        // MASK_BITS must be 16 BITS!

        auto warp = coalesced_threads();
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned noWords = *noWordsPtr;
        const unsigned DIRECTION_THRESHOLD = *directionThresholdPtr;
        const unsigned noSliceSets = *noSliceSetsPtr;
        unsigned levelCount = 0;

        const uint2* row2Ids = reinterpret_cast<const uint2*>(rowIds);

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            if (currentFrontierSize < DIRECTION_THRESHOLD) // spspmv
            {
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned vset = sparseFrontierIds[i];
                    unsigned rset = virtualToReal[vset];

                    unsigned tileStart = sliceSetPtrs[vset] >> 1;
                    unsigned tileEnd = sliceSetPtrs[vset + 1] >> 1;

                    unsigned shift = (rset % 4) << 3;
                    MASK origFragB = ((frontier[rset >> 2] >> shift) & 0x00FF);

                    unsigned tile = tileStart + laneID;
                    uint2 rows = {0, 0};
                    MASK fragA = 0;
                    if (tile < tileEnd)
                    {
                        rows = row2Ids[tile];
                        fragA = masks[tile];
                    }

                    MASK fragB = 0;
                    if (laneID % 9 == 0 || laneID % 9 == 4)
                    {
                        fragB = (laneID % 9 == 0) ? (origFragB) : (origFragB << 8);
                    }
                    unsigned fragC[2];
                    fragC[0] = fragC[1] = 0;
                    m8n8k128(fragC, fragA, fragB);
                    
                    if (fragC[0])
                    {
                        unsigned word = rows.x / UNSIGNED_BITS;
                        unsigned bit = rows.x % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        unsigned old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            levels[rows.x] = levelCount;
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = (0xFF) << (sliceIdx << 3);
                            if ((old & sliceMask) == 0)
                            {
                                unsigned rss = rows.x >> 3;
                                unsigned start = realPtrs[rss];
                                unsigned end = realPtrs[rss + 1];
                                unsigned size = end - start;
                                unsigned loc = atomicAdd(frontierNextSizePtr, size);
                                for (unsigned vset = start; vset < end; ++vset)
                                {
                                    sparseFrontierNextIds[loc++] = vset;
                                }
                            }
                        }
                    }
                    if (fragC[1])
                    {
                        unsigned word = rows.y / UNSIGNED_BITS;
                        unsigned bit = rows.y % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        unsigned old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            levels[rows.y] = levelCount;
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = (0xFF) << (sliceIdx << 3);
                            if ((old & sliceMask) == 0)
                            {
                                unsigned rss = rows.y >> 3;
                                unsigned start = realPtrs[rss];
                                unsigned end = realPtrs[rss + 1];
                                unsigned size = end - start;
                                unsigned loc = atomicAdd(frontierNextSizePtr, size);
                                for (unsigned vset = start; vset < end; ++vset)
                                {
                                    sparseFrontierNextIds[loc++] = vset;
                                }
                            }
                        }
                    }
                }
            }
            else // spmv
            {
                for (unsigned vset = warpID; vset < noSliceSets; vset += noWarps)
                {
                    unsigned rset = virtualToReal[vset];
                    unsigned shift = (rset % 4) << 3;
                    MASK origFragB = ((frontier[rset >> 2] >> shift) & 0x00FF);
                    if (origFragB)
                    {
                        unsigned tileStart = sliceSetPtrs[vset] >> 1;
                        unsigned tileEnd = sliceSetPtrs[vset + 1] >> 1;

                        unsigned tile = tileStart + laneID;
                        uint2 rows = {0, 0};
                        MASK fragA = 0;
                        if (tile < tileEnd)
                        {
                            rows = row2Ids[tile];
                            fragA = masks[tile];
                        }
                        
                        MASK fragB = 0;
                        if (laneID % 9 == 0 || laneID % 9 == 4)
                        {
                            fragB = (laneID % 9 == 0) ? (origFragB) : (origFragB << 8);
                        }
                        unsigned fragC[2];
                        fragC[0] = fragC[1] = 0;
                        m8n8k128(fragC, fragA, fragB);
                        
                        if (fragC[0])
                        {
                            unsigned word = rows.x / UNSIGNED_BITS;
                            unsigned bit = rows.x % UNSIGNED_BITS;
                            unsigned temp = (1 << bit);
                            unsigned old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                levels[rows.x] = levelCount;
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = (0xFF) << (sliceIdx << 3);
                                if ((old & sliceMask) == 0)
                                {
                                    unsigned rss = rows.x >> 3;
                                    unsigned start = realPtrs[rss];
                                    unsigned end = realPtrs[rss + 1];
                                    unsigned size = end - start;
                                    unsigned loc = atomicAdd(frontierNextSizePtr, size);
                                    for (unsigned vset = start; vset < end; ++vset)
                                    {
                                        sparseFrontierNextIds[loc++] = vset;
                                    }
                                }
                            }
                        }
                        if (fragC[1])
                        {
                            unsigned word = rows.y / UNSIGNED_BITS;
                            unsigned bit = rows.y % UNSIGNED_BITS;
                            unsigned temp = (1 << bit);
                            unsigned old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                levels[rows.y] = levelCount;
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = (0xFF) << (sliceIdx << 3);
                                if ((old & sliceMask) == 0)
                                {
                                    unsigned rss = rows.y >> 3;
                                    unsigned start = realPtrs[rss];
                                    unsigned end = realPtrs[rss + 1];
                                    unsigned size = end - start;
                                    unsigned loc = atomicAdd(frontierNextSizePtr, size);
                                    for (unsigned vset = start; vset < end; ++vset)
                                    {
                                        sparseFrontierNextIds[loc++] = vset;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(frontier, frontierNext);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            grid.sync();
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            if (cont)
            {
                for (unsigned i = threadID; i < noWords; i += noThreads)
                {
                    frontierNext[i] = 0;
                }
            }
            grid.sync();
        }
        if (threadID == 0)
        {
            *totalLevels = levelCount;
        }
    }

    __global__ void BRSBFS8EnhancedNoMasks4FullPad( const unsigned* const __restrict__ noSliceSetsPtr,
                                                    const unsigned* const __restrict__ sliceSetPtrs,
                                                    const unsigned* const __restrict__ virtualToReal,
                                                    const unsigned* const __restrict__ realPtrs,
                                                    const unsigned* const __restrict__ rowIds,
                                                    const MASK*     const __restrict__ masks,
                                                    const unsigned* const __restrict__ noWordsPtr,
                                                    const unsigned* const __restrict__ directionThresholdPtr,
                                                    // current
                                                    unsigned*       const __restrict__ frontier,
                                                    unsigned*             __restrict__ sparseFrontierIds,
                                                    unsigned*             __restrict__ frontierCurrentSizePtr,
                                                    // next
                                                    unsigned*       const __restrict__ visitedNext,
                                                    unsigned*             __restrict__ sparseFrontierNextIds,
                                                    unsigned*             __restrict__ frontierNextSizePtr,
                                                    //
                                                    unsigned*       const __restrict__ visited,
                                                    unsigned*       const __restrict__ totalLevels,
                                                    unsigned*       const __restrict__ levels
                                                    )
    {
        // MASK_BITS must be 32 BITS!
        
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned noWords = *noWordsPtr;
        const unsigned DIRECTION_THRESHOLD = *directionThresholdPtr;
        const unsigned noSliceSets = *noSliceSetsPtr;
        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);
        unsigned char* frontierSlice = reinterpret_cast<unsigned char*>(frontier);

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            if (currentFrontierSize < DIRECTION_THRESHOLD) // spspmv
            {
                #pragma unroll 4
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned vset = sparseFrontierIds[i];
                    unsigned rset = virtualToReal[vset];
                    MASK origFragB = static_cast<MASK>(frontierSlice[rset]);

                    unsigned tile = (vset << 5) + laneID; // tile = vset * 128 + laneID * 4 - tile / 4 = vset * 32 + laneID
                    uint4 rows = row4Ids[tile];
                    MASK mask = masks[tile];

                    MASK fragB = 0;
                    {
                        unsigned res = laneID % 9;
                        if (res == 0)
                        {
                            fragB = origFragB;
                        }
                        else if (res == 4)
                        {
                            fragB = (origFragB << 8);
                        }
                    }
                    unsigned fragC[4];
                    fragC[0] = fragC[1] = 0;
                    MASK fragA = (mask & 0x0000FFFF);
                    m8n8k128(fragC, fragA, fragB);

                    fragC[2] = fragC[3] = 0;
                    fragA = ((mask & 0xFFFF0000) >> 16);
                    m8n8k128(&fragC[2], fragA, fragB);

                    unsigned word = rows.x >> 5;
                    unsigned bit = rows.x & 31;
                    unsigned temp = (1 << bit);
                    if (fragC[0])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.y >> 5;
                    bit = rows.y & 31;
                    temp = (1 << bit);
                    if (fragC[1])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.z >> 5;
                    bit = rows.z & 31;
                    temp = (1 << bit);
                    if (fragC[2])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.w >> 5;
                    bit = rows.w & 31;
                    temp = (1 << bit);
                    if (fragC[3])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }
                }
            }
            else // spmv
            {
                for (unsigned vset = warpID; vset < noSliceSets; vset += noWarps)
                {
                    unsigned rset = virtualToReal[vset];
                    MASK origFragB = static_cast<MASK>(frontierSlice[rset]);
                    if (origFragB)
                    {
                        unsigned tile = (vset << 5) + laneID; // tile = vset * 128 + laneID * 4 - tile / 4 = vset * 32 + laneID
                        uint4 rows = row4Ids[tile];
                        MASK mask = masks[tile];

                        MASK fragB = 0;
                        {
                            unsigned res = laneID % 9;
                            if (res == 0)
                            {
                                fragB = origFragB;
                            }
                            else if (res == 4)
                            {
                                fragB = (origFragB << 8);
                            }
                        }
                        unsigned fragC[4];
                        fragC[0] = fragC[1] = 0;
                        MASK fragA = (mask & 0x0000FFFF);
                        m8n8k128(fragC, fragA, fragB);

                        fragC[2] = fragC[3] = 0;
                        fragA = ((mask & 0xFFFF0000) >> 16);
                        m8n8k128(&fragC[2], fragA, fragB);

                        unsigned word = rows.x >> 5;
                        unsigned bit = rows.x & 31;
                        unsigned temp = (1 << bit);
                        if (fragC[0])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.y >> 5;
                        bit = rows.y & 31;
                        temp = (1 << bit);
                        if (fragC[1])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.z >> 5;
                        bit = rows.z & 31;
                        temp = (1 << bit);
                        if (fragC[2])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.w >> 5;
                        bit = rows.w & 31;
                        temp = (1 << bit);
                        if (fragC[3])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }
                    }
                }
            }
            grid.sync();
            for (unsigned i = threadID; i < noWords; i += noThreads)
            {
                unsigned next = visitedNext[i];
                unsigned diff = visited[i] ^ next;
                unsigned rssOffset = i << 2;
                if (diff != 0)
                {
                    visited[i] = next;
                    frontier[i] = diff;
                    #pragma unroll 4
                    for (unsigned set = 0; set < 4; ++set)
                    {
                        MASK sliceMask = ((diff >> (set << 3)) & 0x000000FF);
                        if (sliceMask != 0)
                        {
                            unsigned rss = rssOffset + set;
                            unsigned start = realPtrs[rss];
                            unsigned end = realPtrs[rss + 1];
                            unsigned scan = end - start;

                            auto coalesced = coalesced_threads();
                            unsigned lane = coalesced.thread_rank();

                            for (unsigned stride = 1; stride < coalesced.size(); stride <<= 1)
                            {
                                unsigned from = coalesced.shfl_up(scan, stride);
                                if (lane >= stride) scan += from;
                            }
                            
                            unsigned base;
                            if (lane == coalesced.size() - 1)
                            {
                                base = atomicAdd(frontierNextSizePtr, scan);
                            }
                            base = coalesced.shfl(base, coalesced.size() - 1);
                            for (unsigned vset = start; vset < end; ++vset)
                            {
                                sparseFrontierNextIds[base + --scan] = vset;
                            }
                        }
                    }
                }
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            grid.sync();
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            grid.sync();
        }
        if (threadID == 0)
        {
            *totalLevels = levelCount;
        }
    }

    __global__ void BRSBFS8EnhancedNoMasks4(const unsigned* const __restrict__ noSliceSetsPtr,
                                            const unsigned* const __restrict__ sliceSetPtrs,
                                            const unsigned* const __restrict__ virtualToReal,
                                            const unsigned* const __restrict__ realPtrs,
                                            const unsigned* const __restrict__ rowIds,
                                            const MASK*     const __restrict__ masks,
                                            const unsigned* const __restrict__ noWordsPtr,
                                            const unsigned* const __restrict__ directionThresholdPtr,
                                            // current
                                            unsigned*       const __restrict__ frontier,
                                            unsigned*             __restrict__ sparseFrontierIds,
                                            unsigned*             __restrict__ frontierCurrentSizePtr,
                                            // next
                                            unsigned*       const __restrict__ visitedNext,
                                            unsigned*             __restrict__ sparseFrontierNextIds,
                                            unsigned*             __restrict__ frontierNextSizePtr,
                                            //
                                            unsigned*       const __restrict__ visited,
                                            unsigned*       const __restrict__ totalLevels,
                                            unsigned*       const __restrict__ levels
                                            )
    {
        // MASK_BITS must be 32 BITS!
        
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned noWords = *noWordsPtr;
        const unsigned DIRECTION_THRESHOLD = *directionThresholdPtr;
        const unsigned noSliceSets = *noSliceSetsPtr;
        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);
        unsigned char* frontierSlice = reinterpret_cast<unsigned char*>(frontier);

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            if (currentFrontierSize < DIRECTION_THRESHOLD) // spspmv
            {
                #pragma unroll 4
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned vset = sparseFrontierIds[i];
                    unsigned rset = virtualToReal[vset];
                    MASK origFragB = static_cast<MASK>(frontierSlice[rset]);

                    unsigned tileStart = (sliceSetPtrs[vset] >> 2);
                    unsigned tileEnd = (sliceSetPtrs[vset + 1] >> 2);

                    unsigned tile = tileStart + laneID;
                    uint4 rows = {0, 0, 0, 0};
                    MASK mask = 0;
                    if (tile < tileEnd)
                    {
                        rows = row4Ids[tile];
                        mask = masks[tile];
                    }

                    MASK fragB = 0;
                    {
                        unsigned res = laneID % 9;
                        if (res == 0)
                        {
                            fragB = origFragB;
                        }
                        else if (res == 4)
                        {
                            fragB = (origFragB << 8);
                        }
                    }
                    unsigned fragC[4];
                    fragC[0] = fragC[1] = 0;
                    MASK fragA = (mask & 0x0000FFFF);
                    m8n8k128(fragC, fragA, fragB);

                    fragC[2] = fragC[3] = 0;
                    fragA = ((mask & 0xFFFF0000) >> 16);
                    m8n8k128(&fragC[2], fragA, fragB);

                    unsigned word = rows.x >> 5;
                    unsigned bit = rows.x & 31;
                    unsigned temp = (1 << bit);
                    if (fragC[0])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.y >> 5;
                    bit = rows.y & 31;
                    temp = (1 << bit);
                    if (fragC[1])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.z >> 5;
                    bit = rows.z & 31;
                    temp = (1 << bit);
                    if (fragC[2])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.w >> 5;
                    bit = rows.w & 31;
                    temp = (1 << bit);
                    if (fragC[3])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }
                }
            }
            else // spmv
            {
                for (unsigned vset = warpID; vset < noSliceSets; vset += noWarps)
                {
                    unsigned rset = virtualToReal[vset];
                    MASK origFragB = static_cast<MASK>(frontierSlice[rset]);
                    if (origFragB)
                    {
                        unsigned tileStart = (sliceSetPtrs[vset] >> 2);
                        unsigned tileEnd = (sliceSetPtrs[vset + 1] >> 2);

                        unsigned tile = tileStart + laneID;
                        uint4 rows = {0, 0, 0, 0};
                        MASK mask = 0;
                        if (tile < tileEnd)
                        {
                            rows = row4Ids[tile];
                            mask = masks[tile];
                        }

                        MASK fragB = 0;
                        {
                            unsigned res = laneID % 9;
                            if (res == 0)
                            {
                                fragB = origFragB;
                            }
                            else if (res == 4)
                            {
                                fragB = (origFragB << 8);
                            }
                        }
                        unsigned fragC[4];
                        fragC[0] = fragC[1] = 0;
                        MASK fragA = (mask & 0x0000FFFF);
                        m8n8k128(fragC, fragA, fragB);

                        fragC[2] = fragC[3] = 0;
                        fragA = ((mask & 0xFFFF0000) >> 16);
                        m8n8k128(&fragC[2], fragA, fragB);

                        unsigned word = rows.x >> 5;
                        unsigned bit = rows.x & 31;
                        unsigned temp = (1 << bit);
                        if (fragC[0])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.y >> 5;
                        bit = rows.y & 31;
                        temp = (1 << bit);
                        if (fragC[1])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.z >> 5;
                        bit = rows.z & 31;
                        temp = (1 << bit);
                        if (fragC[2])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.w >> 5;
                        bit = rows.w & 31;
                        temp = (1 << bit);
                        if (fragC[3])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }
                    }
                }
            }
            grid.sync();
            for (unsigned i = threadID; i < noWords; i += noThreads)
            {
                unsigned next = visitedNext[i];
                unsigned diff = visited[i] ^ next;
                unsigned rssOffset = i << 2;
                if (diff != 0)
                {
                    visited[i] = next;
                    frontier[i] = diff;
                    #pragma unroll 4
                    for (unsigned set = 0; set < 4; ++set)
                    {
                        MASK sliceMask = ((diff >> (set << 3)) & 0x000000FF);
                        if (sliceMask != 0)
                        {
                            unsigned rss = rssOffset + set;
                            unsigned start = realPtrs[rss];
                            unsigned end = realPtrs[rss + 1];
                            unsigned scan = end - start;

                            auto coalesced = coalesced_threads();
                            unsigned lane = coalesced.thread_rank();

                            for (unsigned stride = 1; stride < coalesced.size(); stride <<= 1)
                            {
                                unsigned from = coalesced.shfl_up(scan, stride);
                                if (lane >= stride) scan += from;
                            }
                            
                            unsigned base;
                            if (lane == coalesced.size() - 1)
                            {
                                base = atomicAdd(frontierNextSizePtr, scan);
                            }
                            base = coalesced.shfl(base, coalesced.size() - 1);
                            for (unsigned vset = start; vset < end; ++vset)
                            {
                                sparseFrontierNextIds[base + --scan] = vset;
                            }
                        }
                    }
                }
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            grid.sync();
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            grid.sync();
        }
        if (threadID == 0)
        {
            *totalLevels = levelCount;
        }
    }

    __global__ void BRSBFS8EnhancedNoMasks4Pipelined(   const unsigned* const __restrict__ noSliceSetsPtr,
                                                        const unsigned* const __restrict__ sliceSetPtrs,
                                                        const unsigned* const __restrict__ virtualToReal,
                                                        const unsigned* const __restrict__ realPtrs,
                                                        const unsigned* const __restrict__ rowIds,
                                                        const MASK*     const __restrict__ masks,
                                                        const unsigned* const __restrict__ noWordsPtr,
                                                        const unsigned* const __restrict__ directionThresholdPtr,
                                                        // current
                                                        unsigned*       const __restrict__ frontier,
                                                        unsigned*             __restrict__ sparseFrontierIds,
                                                        unsigned*             __restrict__ frontierCurrentSizePtr,
                                                        // next
                                                        unsigned*       const __restrict__ visitedNext,
                                                        unsigned*             __restrict__ sparseFrontierNextIds,
                                                        unsigned*             __restrict__ frontierNextSizePtr,
                                                        //
                                                        unsigned*       const __restrict__ visited,
                                                        unsigned*       const __restrict__ totalLevels,
                                                        unsigned*       const __restrict__ levels
                                                        )
    {
        // MASK_BITS must be 32 BITS!
        
        auto warp = coalesced_threads();
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned noWords = *noWordsPtr;
        const unsigned DIRECTION_THRESHOLD = *directionThresholdPtr;
        const unsigned noSliceSets = *noSliceSetsPtr;
        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);
        unsigned char* frontierSlice = reinterpret_cast<unsigned char*>(frontier);

        unsigned fragC[4];

        unsigned vset0;
        unsigned vset1;
        unsigned vset2;

        unsigned rset0;
        unsigned rset1;
        unsigned rset2;

        unsigned tileStart0;
        unsigned tileStart1;
        unsigned tileStart2;

        unsigned tileEnd0;
        unsigned tileEnd1;
        unsigned tileEnd2;

       auto vsetLoad = [&](unsigned& vset, const unsigned& iter, const unsigned& currentFrontierSize)
        {
            unsigned index = iter + noWarps * laneID;
            vset = (index < currentFrontierSize) ? sparseFrontierIds[index] : UNSIGNED_MAX;
        };

        auto registerLoad = [&](const unsigned& vset, unsigned& rset, unsigned& tileStart, unsigned& tileEnd)
        {
            if (vset != UNSIGNED_MAX)
            {
                rset = virtualToReal[vset];
                tileStart = (sliceSetPtrs[vset] >> 2);
                tileEnd = (sliceSetPtrs[vset + 1] >> 2);
            }
        };

        auto move = [&]()
        {
            vset0 = vset1;
            vset1 = vset2;

            rset0 = rset1;
            rset1 = rset2;

            tileStart0 = tileStart1;
            tileStart1 = tileStart2;

            tileEnd0 = tileEnd1;
            tileEnd1 = tileEnd2;
        };

        const unsigned B = noWarps * WARP_SIZE;
        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            if (currentFrontierSize < DIRECTION_THRESHOLD) // spspmv
            {
                vsetLoad(vset0, warpID, currentFrontierSize);
                vsetLoad(vset1, warpID + 1 * B, currentFrontierSize);
                vsetLoad(vset2, warpID + 2 * B, currentFrontierSize);

                registerLoad(vset0, rset0, tileStart0, tileEnd0);
                registerLoad(vset1, rset1, tileStart1, tileEnd1);
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps * WARP_SIZE)
                {
                    registerLoad(vset2, rset2, tileStart2, tileEnd2);
                    for (unsigned j = 0; j < WARP_SIZE; ++j)
                    {
                        if ((i + noWarps * j) >= currentFrontierSize) break;

                        unsigned tileStart = warp.shfl(tileStart0, j);
                        unsigned tileEnd = warp.shfl(tileEnd0, j);

                        unsigned rset = warp.shfl(rset0, j);

                        unsigned tile = tileStart + laneID;
                        uint4 rows = {0, 0, 0, 0};
                        MASK mask = 0;
                        if (tile < tileEnd)
                        {
                            rows = row4Ids[tile];
                            mask = masks[tile];
                        }

                        MASK origFragB = static_cast<MASK>(frontierSlice[rset]);
                        MASK fragB = 0;
                        {
                            unsigned res = laneID % 9;
                            if (res == 0)
                            {
                                fragB = origFragB;
                            }
                            else if (res == 4)
                            {
                                fragB = origFragB << 8;
                            }
                        }
                        fragC[0] = fragC[1] = 0;
                        MASK fragA = (mask & 0x0000FFFF);
                        m8n8k128(fragC, fragA, fragB);

                        fragC[2] = fragC[3] = 0;
                        fragA = ((mask & 0xFFFF0000) >> 16);
                        m8n8k128(&fragC[2], fragA, fragB);

                        unsigned word = rows.x / UNSIGNED_BITS;
                        unsigned bit = rows.x % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        if (fragC[0])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.y / UNSIGNED_BITS;
                        bit = rows.y % UNSIGNED_BITS;
                        temp = (1 << bit);
                        if (fragC[1])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.z / UNSIGNED_BITS;
                        bit = rows.z % UNSIGNED_BITS;
                        temp = (1 << bit);
                        if (fragC[2])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.w / UNSIGNED_BITS;
                        bit = rows.w % UNSIGNED_BITS;
                        temp = (1 << bit);
                        if (fragC[3])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }
                    }
                    move();
                    vsetLoad(vset2, i + 3 * B, currentFrontierSize);  
                }
            }
            else // spmv
            {
                for (unsigned vset = warpID; vset < noSliceSets; vset += noWarps)
                {
                    unsigned rset = virtualToReal[vset];
                    MASK origFragB = static_cast<MASK>(frontierSlice[rset]);
                    if (origFragB)
                    {
                        unsigned tileStart = (sliceSetPtrs[vset] >> 2);
                        unsigned tileEnd = (sliceSetPtrs[vset + 1] >> 2);

                        unsigned tile = tileStart + laneID;
                        uint4 rows = {0, 0, 0, 0};
                        MASK mask = 0;
                        if (tile < tileEnd)
                        {
                            rows = row4Ids[tile];
                            mask = masks[tile];
                        }

                        MASK fragB = 0;
                        {
                            unsigned res = laneID % 9;
                            if (res == 0)
                            {
                                fragB = origFragB;
                            }
                            else if (res == 4)
                            {
                                fragB = (origFragB << 8);
                            }
                        }
                        fragC[0] = fragC[1] = 0;
                        MASK fragA = (mask & 0x0000FFFF);
                        m8n8k128(fragC, fragA, fragB);

                        fragC[2] = fragC[3] = 0;
                        fragA = ((mask & 0xFFFF0000) >> 16);
                        m8n8k128(&fragC[2], fragA, fragB);

                        unsigned word = rows.x / UNSIGNED_BITS;
                        unsigned bit = rows.x % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        if (fragC[0])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.y / UNSIGNED_BITS;
                        bit = rows.y % UNSIGNED_BITS;
                        temp = (1 << bit);
                        if (fragC[1])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.z / UNSIGNED_BITS;
                        bit = rows.z % UNSIGNED_BITS;
                        temp = (1 << bit);
                        if (fragC[2])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }

                        word = rows.w / UNSIGNED_BITS;
                        bit = rows.w % UNSIGNED_BITS;
                        temp = (1 << bit);
                        if (fragC[3])
                        {
                            atomicOr(&visitedNext[word], temp);
                        }
                    }
                }
            }
            grid.sync();
            for (unsigned i = threadID; i < noWords; i += noThreads)
            {
                unsigned next = visitedNext[i];
                unsigned diff = visited[i] ^ next;
                unsigned rssOffset = i << 2;
                if (diff != 0)
                {
                    visited[i] = next;
                    frontier[i] = diff;
                    #pragma unroll 4
                    for (unsigned set = 0; set < 4; ++set)
                    {
                        MASK sliceMask = ((diff >> (set << 3)) & 0x000000FF);
                        if (sliceMask != 0)
                        {
                            unsigned rss = rssOffset + set;
                            unsigned start = realPtrs[rss];
                            unsigned end = realPtrs[rss + 1];
                            unsigned scan = end - start;

                            auto coalesced = coalesced_threads();
                            unsigned lane = coalesced.thread_rank();

                            for (unsigned stride = 1; stride < coalesced.size(); stride <<= 1)
                            {
                                unsigned from = coalesced.shfl_up(scan, stride);
                                if (lane >= stride) scan += from;
                            }
                            
                            unsigned base;
                            if (lane == coalesced.size() - 1)
                            {
                                base = atomicAdd(frontierNextSizePtr, scan);
                            }
                            base = coalesced.shfl(base, coalesced.size() - 1);
                            for (unsigned vset = start; vset < end; ++vset)
                            {
                                sparseFrontierNextIds[base + --scan] = vset;
                            }
                        }
                    }
                }
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            grid.sync();
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            grid.sync();
        }
        if (threadID == 0)
        {
            *totalLevels = levelCount;
        }
    }
};

class BRSBFSKernel: public BFSKernel
{
public:
    BRSBFSKernel(BitMatrix* matrix);
    BRSBFSKernel(const BRSBFSKernel& other) = delete;
    BRSBFSKernel(BRSBFSKernel&& other) noexcept = delete;
    BRSBFSKernel& operator=(const BRSBFSKernel& other) = delete;
    BRSBFSKernel& operator=(BRSBFSKernel&& other) noexcept = delete;
    virtual ~BRSBFSKernel() = default;

    virtual BFSResult hostCode(unsigned sourceVertex) final;
};

BRSBFSKernel::BRSBFSKernel(BitMatrix* matrix)
: BFSKernel(matrix)
{

}

BFSResult BRSBFSKernel::hostCode(unsigned sourceVertex)
{
    BRS* brs = dynamic_cast<BRS*>(matrix);
    unsigned n = brs->getN();
    unsigned sliceSize = brs->getSliceSize();
    unsigned noMasks = MASK_BITS / sliceSize;
    unsigned noRealSliceSets = brs->getNoRealSliceSets();
    unsigned noSlices = brs->getNoSlices();
    bool isFullPadding = brs->getIsFullPadding();
    unsigned noSliceSets = brs->getNoVirtualSliceSets();
    unsigned* sliceSetPtrs = brs->getSliceSetPtrs();
    unsigned* virtualToReal = brs->getVirtualToReal();
    unsigned* realPtrs = brs->getRealPtrs();
    unsigned* rowIds = brs->getRowIds();
    MASK* masks = brs->getMasks();
    const unsigned DIRECTION_THRESHOLD = noSliceSets * DIRECTION_SWITCHING_CONSTANT; // vset- or rset- based?

    BFSResult result;
    result.levels = new unsigned[n];
    std::fill(result.levels, result.levels + n, UNSIGNED_MAX);
    result.levels[sourceVertex] = 0;

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    void* kernelPtr;
    if (sliceSize == 8 && noMasks == 4)
    {
        if (isFullPadding)
        {
            kernelPtr = (void*)BRSBFSKernels::BRSBFS8EnhancedNoMasks4FullPad;
        }
        else
        {
            kernelPtr = (void*)BRSBFSKernels::BRSBFS8EnhancedNoMasks4;
        }
    }
    else
    {
        throw std::runtime_error("No appropriate kernel found meeting the selected slice size and noMasks.");
    }

    cudaFuncSetAttribute(
    kernelPtr,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    0
    );

    int gridSize, blockSize;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
                                                &gridSize, 
                                                &blockSize, 
                                                kernelPtr,
                                                allocateSharedMemory,
                                                0));

    unsigned* d_NoSliceSets;
    unsigned* d_SliceSetPtrs;
    unsigned* d_VirtualToReal;
    unsigned* d_RealPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    unsigned* d_NoWords;
    unsigned* d_DIRECTION_THRESHOLD;
    unsigned* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_FrontierCurrentSize;
    unsigned* d_FrontierNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_FrontierNextSize;
    unsigned* d_Visited;
    unsigned* d_TotalLevels;
    unsigned* d_Levels;

    // data structure
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_VirtualToReal, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_RealPtrs, sizeof(unsigned) * (noRealSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VirtualToReal, virtualToReal, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RealPtrs, realPtrs, sizeof(unsigned) * (noRealSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    unsigned noWords = (n + UNSIGNED_BITS - 1) / UNSIGNED_BITS;
    gpuErrchk(cudaMalloc(&d_NoWords, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_DIRECTION_THRESHOLD, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets)) // storing vset or rset?
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets)) // storing vset or rset?
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_TotalLevels, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Levels, sizeof(unsigned) * n))

    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_TotalLevels, 0, sizeof(unsigned)))

    gpuErrchk(cudaMemcpy(d_NoWords, &noWords, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_DIRECTION_THRESHOLD, &DIRECTION_THRESHOLD, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Levels, result.levels, sizeof(unsigned) * n, cudaMemcpyHostToDevice))

    std::vector<unsigned> initialVset;
    for (unsigned vset = realPtrs[sourceVertex / sliceSize]; vset < realPtrs[sourceVertex / sliceSize + 1]; ++vset)
    {
        initialVset.emplace_back(vset);
    }
    unsigned initialFrontierSize = initialVset.size();
    gpuErrchk(cudaMemcpy(d_SparseFrontierIds, initialVset.data(), sizeof(unsigned) * initialFrontierSize, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &initialFrontierSize, sizeof(unsigned), cudaMemcpyHostToDevice))

    unsigned word = sourceVertex / UNSIGNED_BITS;
    unsigned bit = sourceVertex % UNSIGNED_BITS;
    unsigned temp = (1 << bit);
    gpuErrchk(cudaMemcpy(d_Frontier + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Visited + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))

    // profiling
    /*
    BRS::SliceSetInformation* d_RSetInformation;
    BRS::SliceSetInformation* d_VSetInformation;
    BRS::SliceInformation* d_SliceInformation;
    gpuErrchk(cudaMalloc(&d_RSetInformation, sizeof(BRS::SliceSetInformation) * noRealSliceSets))
    gpuErrchk(cudaMalloc(&d_VSetInformation, sizeof(BRS::SliceSetInformation) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_SliceInformation, sizeof(BRS::SliceInformation) * noSlices))
    */

    double start = omp_get_wtime();
    void* kernelArgs[] = 
                        {
                            (void*)&d_NoSliceSets,
                            (void*)&d_SliceSetPtrs,
                            (void*)&d_VirtualToReal,
                            (void*)&d_RealPtrs,
                            (void*)&d_RowIds,
                            (void*)&d_Masks,
                            (void*)&d_NoWords,
                            (void*)&d_DIRECTION_THRESHOLD,
                            (void*)&d_Frontier,
                            (void*)&d_SparseFrontierIds,
                            (void*)&d_FrontierCurrentSize,
                            (void*)&d_FrontierNext,
                            (void*)&d_SparseFrontierNextIds,
                            (void*)&d_FrontierNextSize,
                            (void*)&d_Visited,
                            (void*)&d_TotalLevels,
                            (void*)&d_Levels
                        };
    gpuErrchk(cudaLaunchCooperativeKernel(
                                            kernelPtr,
                                            gridSize,
                                            blockSize,
                                            kernelArgs,
                                            allocateSharedMemory(blockSize),
                                            0))
    gpuErrchk(cudaDeviceSynchronize())
    double end = omp_get_wtime();

    unsigned* visited = new unsigned[noWords];
    gpuErrchk(cudaMemcpy(visited, d_Visited, sizeof(unsigned) * noWords, cudaMemcpyDeviceToHost))
    unsigned totalVisited = 0;
    for (unsigned i = 0; i < noWords; ++i)
    {
        totalVisited += __builtin_popcount(visited[i]);
    }
    delete[] visited;
    gpuErrchk(cudaMemcpy(&result.totalLevels, d_TotalLevels, sizeof(unsigned), cudaMemcpyDeviceToHost))
    gpuErrchk(cudaMemcpy(result.levels, d_Levels, sizeof(unsigned) * n, cudaMemcpyDeviceToHost))

    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_VirtualToReal))
    gpuErrchk(cudaFree(d_RealPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_NoWords))
    gpuErrchk(cudaFree(d_DIRECTION_THRESHOLD))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_SparseFrontierIds))
    gpuErrchk(cudaFree(d_FrontierCurrentSize))
    gpuErrchk(cudaFree(d_FrontierNext))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_TotalLevels))
    gpuErrchk(cudaFree(d_Levels))

    BRS::SliceSetInformation* h_RSetInformation = new BRS::SliceSetInformation[noRealSliceSets];
    BRS::SliceSetInformation* h_VSetInformation = new BRS::SliceSetInformation[noSliceSets];
    BRS::SliceInformation* h_SliceInformation = new BRS::SliceInformation[noSlices];

    /*
    gpuErrchk(cudaMemcpy(h_RSetInformation, d_RSetInformation, sizeof(BRS::SliceSetInformation) * noRealSliceSets, cudaMemcpyDeviceToHost))
    gpuErrchk(cudaMemcpy(h_VSetInformation, d_VSetInformation, sizeof(BRS::SliceSetInformation) * noSliceSets, cudaMemcpyDeviceToHost))
    gpuErrchk(cudaMemcpy(h_SliceInformation, d_SliceInformation, sizeof(BRS::SliceInformation) * noSlices, cudaMemcpyDeviceToHost))

    gpuErrchk(cudaFree(d_RSetInformation))
    gpuErrchk(cudaFree(d_VSetInformation))
    gpuErrchk(cudaFree(d_SliceInformation))
    */

    brs->kernelAnalysis(sourceVertex, result.totalLevels, totalVisited, (end - start), h_RSetInformation, h_VSetInformation, h_SliceInformation);

    result.time = (end - start);
    return result;
}
