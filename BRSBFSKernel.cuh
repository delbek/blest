#ifndef BRSBFSKernel_CUH
#define BRSBFSKernel_CUH

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
                                                    unsigned* const __restrict__ totalLevels
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
        unsigned levels = 0;

        const uint2* row2Ids = reinterpret_cast<const uint2*>(rowIds);

        bool cont = true;
        while (cont)
        {
            if (threadID == 0) ++levels;
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
            *totalLevels = levels;
        }
    }

    __global__ void BRSBFS8EnhancedNoMasks4FullPad( const unsigned* const __restrict__ noSliceSetsPtr,
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
                                                    unsigned* const __restrict__ totalLevels
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
        unsigned levels = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);

        bool cont = true;
        while (cont)
        {
            if (threadID == 0) ++levels;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            if (currentFrontierSize < DIRECTION_THRESHOLD) // spspmv
            {
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned vset = sparseFrontierIds[i];
                    unsigned rset = virtualToReal[vset];
                    
                    unsigned shift = (rset % 4) << 3;
                    MASK origFragB = ((frontier[rset >> 2] >> shift) & 0x000000FF);

                    unsigned tile = (vset << 5) + laneID; // slice: (vset * 128 + laneID * 4) - tile: slice / 4 = (vset * 32 + laneID)
                    uint4 rows = row4Ids[tile];
                    MASK mask = masks[tile];

                    MASK fragA = (mask & 0x0000FFFF);
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
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
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
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
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

                    fragA = (mask & 0xFFFF0000);
                    fragB = 0;
                    if (laneID % 9 == 0 || laneID % 9 == 4)
                    {
                        fragB = (laneID % 9 == 0) ? (origFragB << 16) : (origFragB << 24);
                    }
                    fragC[0] = fragC[1] = 0;
                    m8n8k128(fragC, fragA, fragB);

                    if (fragC[0])
                    {
                        unsigned word = rows.z / UNSIGNED_BITS;
                        unsigned bit = rows.z % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        unsigned old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
                            if ((old & sliceMask) == 0)
                            {
                                unsigned rss = rows.z >> 3;
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
                        unsigned word = rows.w / UNSIGNED_BITS;
                        unsigned bit = rows.w % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        unsigned old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
                            if ((old & sliceMask) == 0)
                            {
                                unsigned rss = rows.w >> 3;
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
                    MASK origFragB = ((frontier[rset >> 2] >> shift) & 0x000000FF);
                    if (origFragB)
                    {
                        unsigned tile = (vset << 5) + laneID;
                        uint4 rows = row4Ids[tile];
                        MASK mask = masks[tile];

                        MASK fragA = (mask & 0x0000FFFF);
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
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
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
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
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

                        fragA = (mask & 0xFFFF0000);
                        fragB = 0;
                        if (laneID % 9 == 0 || laneID % 9 == 4)
                        {
                            fragB = (laneID % 9 == 0) ? (origFragB << 16) : (origFragB << 24);
                        }
                        fragC[0] = fragC[1] = 0;
                        m8n8k128(fragC, fragA, fragB);

                        if (fragC[0])
                        {
                            unsigned word = rows.z / UNSIGNED_BITS;
                            unsigned bit = rows.z % UNSIGNED_BITS;
                            unsigned temp = (1 << bit);
                            unsigned old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
                                if ((old & sliceMask) == 0)
                                {
                                    unsigned rss = rows.z >> 3;
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
                            unsigned word = rows.w / UNSIGNED_BITS;
                            unsigned bit = rows.w % UNSIGNED_BITS;
                            unsigned temp = (1 << bit);
                            unsigned old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
                                if ((old & sliceMask) == 0)
                                {
                                    unsigned rss = rows.w >> 3;
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
            *totalLevels = levels;
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
                                                unsigned* const __restrict__ totalLevels
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
        unsigned levels = 0;

        const uint2* row2Ids = reinterpret_cast<const uint2*>(rowIds);

        bool cont = true;
        while (cont)
        {
            if (threadID == 0) ++levels;
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
            *totalLevels = levels;
        }
    }

    __global__ void BRSBFS8EnhancedNoMasks4(    const unsigned* const __restrict__ noSliceSetsPtr,
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
                                                unsigned* const __restrict__ totalLevels
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
        unsigned levels = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);

        bool cont = true;
        while (cont)
        {
            if (threadID == 0) ++levels;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            if (currentFrontierSize < DIRECTION_THRESHOLD) // spspmv
            {
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned vset = sparseFrontierIds[i];
                    unsigned rset = virtualToReal[vset];
                    
                    unsigned tileStart = sliceSetPtrs[vset] >> 2;
                    unsigned tileEnd = sliceSetPtrs[vset + 1] >> 2;

                    unsigned shift = (rset % 4) << 3;
                    MASK origFragB = ((frontier[rset >> 2] >> shift) & 0x000000FF);

                    unsigned tile = tileStart + laneID;
                    uint4 rows = {0, 0, 0, 0};
                    MASK mask = 0;
                    if (tile < tileEnd)
                    {
                        rows = row4Ids[tile];
                        mask = masks[tile];
                    }

                    MASK fragA = (mask & 0x0000FFFF);
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
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
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
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
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

                    fragA = (mask & 0xFFFF0000);
                    fragB = 0;
                    if (laneID % 9 == 0 || laneID % 9 == 4)
                    {
                        fragB = (laneID % 9 == 0) ? (origFragB << 16) : (origFragB << 24);
                    }
                    fragC[0] = fragC[1] = 0;
                    m8n8k128(fragC, fragA, fragB);

                    if (fragC[0])
                    {
                        unsigned word = rows.z / UNSIGNED_BITS;
                        unsigned bit = rows.z % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        unsigned old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
                            if ((old & sliceMask) == 0)
                            {
                                unsigned rss = rows.z >> 3;
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
                        unsigned word = rows.w / UNSIGNED_BITS;
                        unsigned bit = rows.w % UNSIGNED_BITS;
                        unsigned temp = (1 << bit);
                        unsigned old = atomicOr(&visited[word], temp);
                        if ((old & temp) == 0)
                        {
                            old = atomicOr(&frontierNext[word], temp);
                            unsigned sliceIdx = (bit >> 3);
                            unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
                            if ((old & sliceMask) == 0)
                            {
                                unsigned rss = rows.w >> 3;
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
                    MASK origFragB = ((frontier[rset >> 2] >> shift) & 0x000000FF);
                    if (origFragB)
                    {
                        unsigned tileStart = sliceSetPtrs[vset] >> 2;
                        unsigned tileEnd = sliceSetPtrs[vset + 1] >> 2;

                        unsigned tile = tileStart + laneID;
                        uint4 rows = {0, 0, 0, 0};
                        MASK mask = 0;
                        if (tile < tileEnd)
                        {
                            rows = row4Ids[tile];
                            mask = masks[tile];
                        }

                        MASK fragA = (mask & 0x0000FFFF);
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
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
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
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
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

                        fragA = (mask & 0xFFFF0000);
                        fragB = 0;
                        if (laneID % 9 == 0 || laneID % 9 == 4)
                        {
                            fragB = (laneID % 9 == 0) ? (origFragB << 16) : (origFragB << 24);
                        }
                        fragC[0] = fragC[1] = 0;
                        m8n8k128(fragC, fragA, fragB);
    
                        if (fragC[0])
                        {
                            unsigned word = rows.z / UNSIGNED_BITS;
                            unsigned bit = rows.z % UNSIGNED_BITS;
                            unsigned temp = (1 << bit);
                            unsigned old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
                                if ((old & sliceMask) == 0)
                                {
                                    unsigned rss = rows.z >> 3;
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
                            unsigned word = rows.w / UNSIGNED_BITS;
                            unsigned bit = rows.w % UNSIGNED_BITS;
                            unsigned temp = (1 << bit);
                            unsigned old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                unsigned sliceIdx = (bit >> 3);
                                unsigned sliceMask = ((0xFF) << (sliceIdx << 3));
                                if ((old & sliceMask) == 0)
                                {
                                    unsigned rss = rows.w >> 3;
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
            *totalLevels = levels;
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

    virtual double hostCode(unsigned sourceVertex) final;
};

BRSBFSKernel::BRSBFSKernel(BitMatrix* matrix)
: BFSKernel(matrix)
{

}

double BRSBFSKernel::hostCode(unsigned sourceVertex)
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
    const unsigned DIRECTION_THRESHOLD = noSliceSets / 2; // vset- or rset- based?

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    void* kernelPtr = nullptr;
    if (sliceSize == 8 && noMasks == 2)
    {
        if (isFullPadding)
        {
            kernelPtr = (void*)BRSBFSKernels::BRSBFS8EnhancedNoMasks2FullPad;
        }
        else
        {
            kernelPtr = (void*)BRSBFSKernels::BRSBFS8EnhancedNoMasks2;
        }
    }
    else if (sliceSize == 8 && noMasks == 4)
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

    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_TotalLevels, 0, sizeof(unsigned)))

    gpuErrchk(cudaMemcpy(d_NoWords, &noWords, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_DIRECTION_THRESHOLD, &DIRECTION_THRESHOLD, sizeof(unsigned), cudaMemcpyHostToDevice))
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
                            (void*)&d_TotalLevels
                            /*
                            (void*)&d_RSetInformation,
                            (void*)&d_VSetInformation,
                            (void*)&d_SliceInformation
                            */
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
    unsigned totalLevels;
    gpuErrchk(cudaMemcpy(&totalLevels, d_TotalLevels, sizeof(unsigned), cudaMemcpyDeviceToHost))

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

    brs->kernelAnalysis(sourceVertex, totalLevels, totalVisited, (end - start), h_RSetInformation, h_VSetInformation, h_SliceInformation);

    return (end - start);
}

#endif
