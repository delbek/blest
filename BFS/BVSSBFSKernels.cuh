/*
 * This file is part of the BLEST repository: https://github.com/delbek/blest
 * Author: Deniz Elbek
 *
 * Please see the paper:
 * 
 * @article{Elbek2025BLEST,
 *   title   = {BLEST: Blazingly Efficient BFS using Tensor Cores},
 *   author  = {Elbek, Deniz and Kaya, Kamer},
 *   journal = {arXiv preprint arXiv:2512.21967},
 *   year    = {2025},
 *   doi     = {10.48550/arXiv.2512.21967},
 *   url     = {https://www.arxiv.org/abs/2512.21967}
 * }
 */

#pragma once

#include "Common.cuh"

namespace BVSSBFSKernels
{
    template<typename T>
    __device__ __forceinline__ void swap(T* __restrict__& ptr1, T* __restrict__& ptr2)
    {
        T* temp = ptr2;
        ptr2 = ptr1;
        ptr1 = temp;
    }

    __global__ void BVSSBFS8EnhancedSliceSize8NoMasks4LazySwitching (
                                                                    const unsigned*   const __restrict__ rowPtrs,
                                                                    const unsigned*   const __restrict__ colIds,
                                                                    const unsigned*   const __restrict__ nPtr,
                                                                    const SLICE_TYPE* const __restrict__ sliceSetPtrs,
                                                                    const unsigned*   const __restrict__ virtualToReal,
                                                                    const unsigned*   const __restrict__ realPtrs,
                                                                    const unsigned*   const __restrict__ rowIds,
                                                                    const MASK*       const __restrict__ masks,
                                                                    const unsigned*   const __restrict__ noWordsPtr,
                                                                    // current
                                                                    unsigned*         const __restrict__ levels,
                                                                    unsigned*         const __restrict__ frontier,
                                                                    unsigned*         const __restrict__ visited,
                                                                    unsigned*               __restrict__ sparseFrontierIds,
                                                                    unsigned*               __restrict__ unvisitedCurrentSizePtr,
                                                                    unsigned*               __restrict__ frontierCurrentSizePtr,
                                                                    // next
                                                                    unsigned*         const __restrict__ visitedNext,
                                                                    unsigned*               __restrict__ sparseFrontierNextIds,
                                                                    unsigned*               __restrict__ unvisitedNextSizePtr,
                                                                    unsigned*               __restrict__ frontierNextSizePtr
                                                                    /*
                                                                    // profiling
                                                                    unsigned long long* levelTime
                                                                    //
                                                                    */
                                                                    )
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned n = *nPtr;
        const unsigned noWords = *noWordsPtr;
        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);
        unsigned char* frontierSlice = reinterpret_cast<unsigned char*>(frontier);

        /*
        if (threadID == 0)
        {
            levelTime[levelCount] = getTime();
        }
        grid.sync();
        */

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentUnvisitedSize = *unvisitedCurrentSizePtr;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            /*
            if (threadID == 0)
            {
                printf("Current unvisited: %u - Current frontier: %u - Ratio: %f\n", currentUnvisitedSize, currentFrontierSize, double(currentUnvisitedSize) / currentFrontierSize);
            }
            */
            if (currentUnvisitedSize < currentFrontierSize * SWITCHING_CONSTANT)
            {
                for (unsigned i = warpID; i < noWords; i += noWarps)
                {
                    unsigned unvisitedMask = ~visited[i];
                    bool isUnvisited = (unvisitedMask >> laneID) & 1;

                    if (isUnvisited)
                    {
                        unsigned u = i * WARP_SIZE + laneID;
                        if (u < n)
                        {
                            for (unsigned nnz = rowPtrs[u]; nnz < rowPtrs[u + 1]; ++nnz)
                            {
                                unsigned v = colIds[nnz];
                                if ((frontier[v >> 5] >> (v & 31)) & 1)
                                {
                                    atomicOr(&visitedNext[i], 1u << laneID);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned vset = sparseFrontierIds[i];
                    unsigned rset = virtualToReal[vset];
                    MASK origFragB = static_cast<MASK>(frontierSlice[rset]);

                    unsigned tileStart = static_cast<unsigned>(sliceSetPtrs[vset] >> 2);
                    unsigned tileEnd = static_cast<unsigned>(sliceSetPtrs[vset + 1] >> 2);

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
                    unsigned temp = (1u << bit);
                    if (fragC[0])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.y >> 5;
                    bit = rows.y & 31;
                    temp = (1u << bit);
                    if (fragC[1])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.z >> 5;
                    bit = rows.z & 31;
                    temp = (1u << bit);
                    if (fragC[2])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.w >> 5;
                    bit = rows.w & 31;
                    temp = (1u << bit);
                    if (fragC[3])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }
                }
            }
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
                *unvisitedNextSizePtr = 0;
            }
            grid.sync();
            unsigned totalUnvisited = 0;
            for (unsigned i = threadID; i < noWords; i += noThreads)
            {
                unsigned current = visited[i];
                unsigned next = visitedNext[i];
                totalUnvisited += __popc(~next);
                unsigned diff = current ^ next;
                frontier[i] = diff;
                unsigned rssOffset = i << 2;
                if (diff != 0)
                {
                    visited[i] = next;
                    #pragma unroll 4
                    for (unsigned set = 0; set < 4; ++set)
                    {
                        MASK sliceMask = ((diff >> (set << 3)) & 0x000000FF);
                        if (sliceMask != 0)
                        {
                            unsigned rss = rssOffset + set;
                            while (sliceMask)
                            {
                                unsigned vertex = (rss << 3) + (__ffs(sliceMask) - 1);
                                levels[vertex] = levelCount;
                                sliceMask &= sliceMask - 1;
                            }
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
                            
                            unsigned base = 0;
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
            #pragma unroll 5
            for (unsigned stride = WARP_SIZE / 2; stride > 0; stride >>= 1)
            {
                totalUnvisited += warp.shfl_down(totalUnvisited, stride);
            }
            if (warp.thread_rank() == 0)
            {
                atomicAdd(unvisitedNextSizePtr, totalUnvisited);
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            swap<unsigned>(unvisitedCurrentSizePtr, unvisitedNextSizePtr);
            grid.sync();
            /*
            if (threadID == 0)
            {
                levelTime[levelCount] = getTime();
            }
            grid.sync();
            */
        }
    }

    __global__ void BVSSBFS8EnhancedSliceSize8NoMasks4LazyFullPadSwitching  (
                                                                            const unsigned*   const __restrict__ rowPtrs,
                                                                            const unsigned*   const __restrict__ colIds,
                                                                            const unsigned*   const __restrict__ nPtr,
                                                                            const SLICE_TYPE* const __restrict__ sliceSetPtrs,
                                                                            const unsigned*   const __restrict__ virtualToReal,
                                                                            const unsigned*   const __restrict__ realPtrs,
                                                                            const unsigned*   const __restrict__ rowIds,
                                                                            const MASK*       const __restrict__ masks,
                                                                            const unsigned*   const __restrict__ noWordsPtr,
                                                                            // current
                                                                            unsigned*         const __restrict__ levels,
                                                                            unsigned*         const __restrict__ frontier,
                                                                            unsigned*         const __restrict__ visited,
                                                                            unsigned*               __restrict__ sparseFrontierIds,
                                                                            unsigned*               __restrict__ unvisitedCurrentSizePtr,
                                                                            unsigned*               __restrict__ frontierCurrentSizePtr,
                                                                            // next
                                                                            unsigned*         const __restrict__ visitedNext,
                                                                            unsigned*               __restrict__ sparseFrontierNextIds,
                                                                            unsigned*               __restrict__ unvisitedNextSizePtr,
                                                                            unsigned*               __restrict__ frontierNextSizePtr
                                                                            /*
                                                                            // profiling
                                                                            unsigned long long* levelTime
                                                                            //
                                                                            */
                                                                            )
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned n = *nPtr;
        const unsigned noWords = *noWordsPtr;
        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);
        unsigned char* frontierSlice = reinterpret_cast<unsigned char*>(frontier);

        /*
        if (threadID == 0)
        {
            levelTime[levelCount] = getTime();
        }
        grid.sync();
        */

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentUnvisitedSize = *unvisitedCurrentSizePtr;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            /*
            if (threadID == 0)
            {
                printf("Current unvisited: %u - Current frontier: %u - Ratio: %f\n", currentUnvisitedSize, currentFrontierSize, double(currentUnvisitedSize) / currentFrontierSize);
            }
            */
            if (currentUnvisitedSize < currentFrontierSize * SWITCHING_CONSTANT)
            {
                for (unsigned i = warpID; i < noWords; i += noWarps)
                {
                    unsigned unvisitedMask = ~visited[i];
                    bool isUnvisited = (unvisitedMask >> laneID) & 1;

                    if (isUnvisited)
                    {
                        unsigned u = i * WARP_SIZE + laneID;
                        if (u < n)
                        {
                            for (unsigned nnz = rowPtrs[u]; nnz < rowPtrs[u + 1]; ++nnz)
                            {
                                unsigned v = colIds[nnz];
                                if ((frontier[v >> 5] >> (v & 31)) & 1)
                                {
                                    atomicOr(&visitedNext[i], 1u << laneID);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned vset = sparseFrontierIds[i];
                    unsigned rset = virtualToReal[vset];
                    MASK origFragB = static_cast<MASK>(frontierSlice[rset]);

                    unsigned tile = (vset << 5) + laneID;
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
                    unsigned temp = (1u << bit);
                    if (fragC[0])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.y >> 5;
                    bit = rows.y & 31;
                    temp = (1u << bit);
                    if (fragC[1])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.z >> 5;
                    bit = rows.z & 31;
                    temp = (1u << bit);
                    if (fragC[2])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }

                    word = rows.w >> 5;
                    bit = rows.w & 31;
                    temp = (1u << bit);
                    if (fragC[3])
                    {
                        atomicOr(&visitedNext[word], temp);
                    }
                }
            }
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
                *unvisitedNextSizePtr = 0;
            }
            grid.sync();
            unsigned totalUnvisited = 0;
            for (unsigned i = threadID; i < noWords; i += noThreads)
            {
                unsigned current = visited[i];
                unsigned next = visitedNext[i];
                totalUnvisited += __popc(~next);
                unsigned diff = current ^ next;
                frontier[i] = diff;
                unsigned rssOffset = i << 2;
                if (diff != 0)
                {
                    visited[i] = next;
                    #pragma unroll 4
                    for (unsigned set = 0; set < 4; ++set)
                    {
                        MASK sliceMask = ((diff >> (set << 3)) & 0x000000FF);
                        if (sliceMask != 0)
                        {
                            unsigned rss = rssOffset + set;
                            while (sliceMask)
                            {
                                unsigned vertex = (rss << 3) + (__ffs(sliceMask) - 1);
                                levels[vertex] = levelCount;
                                sliceMask &= sliceMask - 1;
                            }
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
                            
                            unsigned base = 0;
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
            #pragma unroll 5
            for (unsigned stride = WARP_SIZE / 2; stride > 0; stride >>= 1)
            {
                totalUnvisited += warp.shfl_down(totalUnvisited, stride);
            }
            if (warp.thread_rank() == 0)
            {
                atomicAdd(unvisitedNextSizePtr, totalUnvisited);
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            swap<unsigned>(unvisitedCurrentSizePtr, unvisitedNextSizePtr);
            grid.sync();
            /*
            if (threadID == 0)
            {
                levelTime[levelCount] = getTime();
            }
            grid.sync();
            */
        }
    }

    __global__ void BVSSBFS8EnhancedSliceSize8NoMasks4LazyFullPad   (
                                                                    const SLICE_TYPE* const __restrict__ sliceSetPtrs,
                                                                    const unsigned*   const __restrict__ virtualToReal,
                                                                    const unsigned*   const __restrict__ realPtrs,
                                                                    const unsigned*   const __restrict__ rowIds,
                                                                    const MASK*       const __restrict__ masks,
                                                                    const unsigned*   const __restrict__ noWordsPtr,
                                                                    // current
                                                                    unsigned*         const __restrict__ levels,
                                                                    unsigned*         const __restrict__ frontier,
                                                                    unsigned*         const __restrict__ visited,
                                                                    unsigned*               __restrict__ sparseFrontierIds,
                                                                    unsigned*               __restrict__ frontierCurrentSizePtr,
                                                                    // next
                                                                    unsigned*         const __restrict__ visitedNext,
                                                                    unsigned*               __restrict__ sparseFrontierNextIds,
                                                                    unsigned*               __restrict__ frontierNextSizePtr
                                                                    )
    {
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned noWords = *noWordsPtr;
        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);
        unsigned char* frontierSlice = reinterpret_cast<unsigned char*>(frontier);

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
            {
                unsigned vset = sparseFrontierIds[i];
                unsigned rset = virtualToReal[vset];
                MASK origFragB = static_cast<MASK>(frontierSlice[rset]);

                unsigned tile = (vset << 5) + laneID;
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
                unsigned temp = (1u << bit);
                if (fragC[0])
                {
                    atomicOr(&visitedNext[word], temp);
                }

                word = rows.y >> 5;
                bit = rows.y & 31;
                temp = (1u << bit);
                if (fragC[1])
                {
                    atomicOr(&visitedNext[word], temp);
                }

                word = rows.z >> 5;
                bit = rows.z & 31;
                temp = (1u << bit);
                if (fragC[2])
                {
                    atomicOr(&visitedNext[word], temp);
                }

                word = rows.w >> 5;
                bit = rows.w & 31;
                temp = (1u << bit);
                if (fragC[3])
                {
                    atomicOr(&visitedNext[word], temp);
                }
            }
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
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
                            while (sliceMask)
                            {
                                unsigned vertex = (rss << 3) + (__ffs(sliceMask) - 1);
                                levels[vertex] = levelCount;
                                sliceMask &= sliceMask - 1;
                            }
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
                            
                            unsigned base = 0;
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
        }
    }

    __global__ void BVSSBFS8EnhancedSliceSize8NoMasks4Lazy  (
                                                            const SLICE_TYPE* const __restrict__ sliceSetPtrs,
                                                            const unsigned*   const __restrict__ virtualToReal,
                                                            const unsigned*   const __restrict__ realPtrs,
                                                            const unsigned*   const __restrict__ rowIds,
                                                            const MASK*       const __restrict__ masks,
                                                            const unsigned*   const __restrict__ noWordsPtr,
                                                            // current
                                                            unsigned*         const __restrict__ levels,
                                                            unsigned*         const __restrict__ frontier,
                                                            unsigned*         const __restrict__ visited,
                                                            unsigned*               __restrict__ sparseFrontierIds,
                                                            unsigned*               __restrict__ frontierCurrentSizePtr,
                                                            // next
                                                            unsigned*         const __restrict__ visitedNext,
                                                            unsigned*               __restrict__ sparseFrontierNextIds,
                                                            unsigned*               __restrict__ frontierNextSizePtr
                                                            )
    {
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned noWords = *noWordsPtr;
        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);
        unsigned char* frontierSlice = reinterpret_cast<unsigned char*>(frontier);

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
            {
                unsigned vset = sparseFrontierIds[i];
                unsigned rset = virtualToReal[vset];
                MASK origFragB = static_cast<MASK>(frontierSlice[rset]);

                unsigned tileStart = static_cast<unsigned>(sliceSetPtrs[vset] >> 2);
                unsigned tileEnd = static_cast<unsigned>(sliceSetPtrs[vset + 1] >> 2);

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
                unsigned temp = (1u << bit);
                if (fragC[0])
                {
                    atomicOr(&visitedNext[word], temp);
                }

                word = rows.y >> 5;
                bit = rows.y & 31;
                temp = (1u << bit);
                if (fragC[1])
                {
                    atomicOr(&visitedNext[word], temp);
                }

                word = rows.z >> 5;
                bit = rows.z & 31;
                temp = (1u << bit);
                if (fragC[2])
                {
                    atomicOr(&visitedNext[word], temp);
                }

                word = rows.w >> 5;
                bit = rows.w & 31;
                temp = (1u << bit);
                if (fragC[3])
                {
                    atomicOr(&visitedNext[word], temp);
                }
            }
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
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
                            while (sliceMask)
                            {
                                unsigned vertex = (rss << 3) + (__ffs(sliceMask) - 1);
                                levels[vertex] = levelCount;
                                sliceMask &= sliceMask - 1;
                            }
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
                            
                            unsigned base = 0;
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
        }
    }

    __global__ void BVSSBFS8EnhancedSliceSize8NoMasks4FullPad   (
                                                                const SLICE_TYPE* const __restrict__ sliceSetPtrs,
                                                                const unsigned*   const __restrict__ virtualToReal,
                                                                const unsigned*   const __restrict__ realPtrs,
                                                                const unsigned*   const __restrict__ rowIds,
                                                                const MASK*       const __restrict__ masks,
                                                                const unsigned*   const __restrict__ noWordsPtr,
                                                                // current
                                                                unsigned*         const __restrict__ levels,
                                                                unsigned*               __restrict__ frontier,
                                                                unsigned*               __restrict__ sparseFrontierIds,
                                                                unsigned*               __restrict__ frontierCurrentSizePtr,
                                                                // next
                                                                unsigned*               __restrict__ frontierNext,
                                                                unsigned*               __restrict__ sparseFrontierNextIds,
                                                                unsigned*               __restrict__ frontierNextSizePtr
                                                                )
    {
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned noWords = *noWordsPtr;
        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned char* frontierSlice = reinterpret_cast<unsigned char*>(frontier);
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
            {
                unsigned vset = sparseFrontierIds[i];
                unsigned rset = virtualToReal[vset];
                MASK origFragB = static_cast<MASK>(frontierSlice[rset]);

                unsigned tile = (vset << 5) + laneID;
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
                unsigned temp = (1u << bit);
                if (fragC[0])
                {
                    unsigned oldLevel = levels[rows.x];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.x] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
                        unsigned sliceIdx = (bit >> 3);
                        unsigned sliceMask = ((0x000000FF) << (sliceIdx << 3));
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

                word = rows.y >> 5;
                bit = rows.y & 31;
                temp = (1u << bit);
                if (fragC[1])
                {
                    unsigned oldLevel = levels[rows.y];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.y] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
                        unsigned sliceIdx = (bit >> 3);
                        unsigned sliceMask = ((0x000000FF) << (sliceIdx << 3));
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

                word = rows.z >> 5;
                bit = rows.z & 31;
                temp = (1u << bit);
                if (fragC[2])
                {
                    unsigned oldLevel = levels[rows.z];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.z] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
                        unsigned sliceIdx = (bit >> 3);
                        unsigned sliceMask = ((0x000000FF) << (sliceIdx << 3));
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

                word = rows.w >> 5;
                bit = rows.w & 31;
                temp = (1u << bit);
                if (fragC[3])
                {
                    unsigned oldLevel = levels[rows.w];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.w] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
                        unsigned sliceIdx = (bit >> 3);
                        unsigned sliceMask = ((0x000000FF) << (sliceIdx << 3));
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
    }

    __global__ void BVSSBFS8EnhancedSliceSize8NoMasks4  (
                                                        const SLICE_TYPE* const __restrict__ sliceSetPtrs,
                                                        const unsigned*   const __restrict__ virtualToReal,
                                                        const unsigned*   const __restrict__ realPtrs,
                                                        const unsigned*   const __restrict__ rowIds,
                                                        const MASK*       const __restrict__ masks,
                                                        const unsigned*   const __restrict__ noWordsPtr,
                                                        // current
                                                        unsigned*         const __restrict__ levels,
                                                        unsigned*               __restrict__ frontier,
                                                        unsigned*               __restrict__ sparseFrontierIds,
                                                        unsigned*               __restrict__ frontierCurrentSizePtr,
                                                        // next
                                                        unsigned*               __restrict__ frontierNext,
                                                        unsigned*               __restrict__ sparseFrontierNextIds,
                                                        unsigned*               __restrict__ frontierNextSizePtr
                                                        )
    {
        auto grid = this_grid();
        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned noWords = *noWordsPtr;
        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned char* frontierSlice = reinterpret_cast<unsigned char*>(frontier);
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
            {
                unsigned vset = sparseFrontierIds[i];
                unsigned rset = virtualToReal[vset];
                MASK origFragB = static_cast<MASK>(frontierSlice[rset]);

                unsigned tileStart = static_cast<unsigned>(sliceSetPtrs[vset] >> 2);
                unsigned tileEnd = static_cast<unsigned>(sliceSetPtrs[vset + 1] >> 2);

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
                unsigned temp = (1u << bit);
                if (fragC[0])
                {
                    unsigned oldLevel = levels[rows.x];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.x] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
                        unsigned sliceIdx = (bit >> 3);
                        unsigned sliceMask = ((0x000000FF) << (sliceIdx << 3));
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

                word = rows.y >> 5;
                bit = rows.y & 31;
                temp = (1u << bit);
                if (fragC[1])
                {
                    unsigned oldLevel = levels[rows.y];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.y] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
                        unsigned sliceIdx = (bit >> 3);
                        unsigned sliceMask = ((0x000000FF) << (sliceIdx << 3));
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

                word = rows.z >> 5;
                bit = rows.z & 31;
                temp = (1u << bit);
                if (fragC[2])
                {
                    unsigned oldLevel = levels[rows.z];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.z] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
                        unsigned sliceIdx = (bit >> 3);
                        unsigned sliceMask = ((0x000000FF) << (sliceIdx << 3));
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

                word = rows.w >> 5;
                bit = rows.w & 31;
                temp = (1u << bit);
                if (fragC[3])
                {
                    unsigned oldLevel = levels[rows.w];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.w] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
                        unsigned sliceIdx = (bit >> 3);
                        unsigned sliceMask = ((0x000000FF) << (sliceIdx << 3));
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
    }
};
