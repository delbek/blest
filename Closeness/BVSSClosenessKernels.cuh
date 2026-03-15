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

namespace BVSSClosenessKernels
{
    template<typename T>
    __device__ __forceinline__ void swap(T* __restrict__& ptr1, T* __restrict__& ptr2)
    {
        T* temp = ptr2;
        ptr2 = ptr1;
        ptr1 = temp;
    }

    __device__ __forceinline__ unsigned getVertexIndex(const unsigned& vertex, const unsigned& partitionSize)
    {
        return ((vertex & 7u) * partitionSize) + (vertex >> 3);
    }

    __device__ __forceinline__ ulonglong4_32a xor256(const ulonglong4_32a& u1, const ulonglong4_32a& u2)
    {
        ulonglong4_32a ret;
        ret.x = (u1.x ^ u2.x);
        ret.y = (u1.y ^ u2.y);
        ret.z = (u1.z ^ u2.z);
        ret.w = (u1.w ^ u2.w);
        return ret;
    }

    __global__ void BVSSCloseness8EnhancedSliceSize8NoMasks4LazyChunkFusion     (
                                                                                const unsigned*             const   __restrict__ nPtr,
                                                                                const unsigned*             const   __restrict__ paddedNPtr,
                                                                                const unsigned*             const   __restrict__ noRealSliceSetsPtr,
                                                                                const unsigned*             const   __restrict__ rowPtrs,
                                                                                const unsigned*             const   __restrict__ colIds,
                                                                                const SLICE_TYPE*           const   __restrict__ sliceSetPtrs,
                                                                                const unsigned*             const   __restrict__ virtualToReal,
                                                                                const unsigned*             const   __restrict__ realPtrs,
                                                                                const unsigned*             const   __restrict__ rowIds,
                                                                                const MASK*                 const   __restrict__ masks,
                                                                                // current
                                                                                unsigned long long*         const   __restrict__ far,
                                                                                ulonglong4_32a*             const   __restrict__ activeRSets,
                                                                                bool*                       const   __restrict__ dirtyRSets,
                                                                                ulonglong4_32a*             const   __restrict__ frontier,
                                                                                ulonglong4_32a*             const   __restrict__ visited,
                                                                                unsigned*                           __restrict__ sparseFrontierIds,
                                                                                unsigned*                           __restrict__ frontierCurrentSizePtr,
                                                                                // next
                                                                                ulonglong4_32a*             const   __restrict__ visitedNext,
                                                                                unsigned*                           __restrict__ sparseFrontierNextIds,
                                                                                unsigned*                           __restrict__ frontierNextSizePtr
                                                                                )
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        auto block = this_thread_block();

        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned n = *nPtr;
        const unsigned paddedN = *paddedNPtr;
        const unsigned noRealSliceSets = *noRealSliceSetsPtr;
        const unsigned partitionSize = paddedN >> 3;

        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);

        bool cont = true;
        while (cont)
        {
            ++levelCount;
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
            {
                unsigned vset = sparseFrontierIds[i];
                unsigned rset = virtualToReal[vset];

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

                bool dirtyBits[4] = {false, false, false, false};
                for (unsigned chunk = 0; chunk < TASK_SIZE; ++chunk)
                {
                    unsigned iter = chunk * paddedN;

                    ulonglong4_32a frontierVertex;
                    if (laneID < 8)
                    {
                        frontierVertex = frontier[iter + laneID * partitionSize + rset]; // very bad access
                    }
                    else
                    {
                        frontierVertex = {0, 0, 0, 0};
                    }

                    ulonglong4_32a activeSet = activeRSets[chunk * noRealSliceSets + rset];
                    ulonglong4_32a visitedMarks[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
                    // visitedMarks[x][y]: x pulling vertex local ID - y BFS id (bitwise)

                    // first 64 BFS
                    while (activeSet.x)
                    {
                        unsigned f = (__ffsll(activeSet.x) - 1);
                        activeSet.x &= activeSet.x - 1;

                        unsigned included = ((frontierVertex.x >> f) & 1);
                        MASK origFragB = static_cast<MASK>(warp.ballot(included));
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

                        if (fragC[0])
                        {
                            visitedMarks[0].x |= (1ull << f);
                        }

                        if (fragC[1])
                        {
                            visitedMarks[1].x |= (1ull << f);
                        }

                        if (fragC[2])
                        {
                            visitedMarks[2].x |= (1ull << f);
                        }

                        if (fragC[3])
                        {
                            visitedMarks[3].x |= (1ull << f);
                        }
                    }

                    // second 64 BFS
                    while (activeSet.y)
                    {
                        unsigned f = (__ffsll(activeSet.y) - 1);
                        activeSet.y &= activeSet.y - 1;

                        unsigned included = ((frontierVertex.y >> f) & 1);
                        MASK origFragB = static_cast<MASK>(warp.ballot(included));
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

                        if (fragC[0])
                        {
                            visitedMarks[0].y |= (1ull << f);
                        }

                        if (fragC[1])
                        {
                            visitedMarks[1].y |= (1ull << f);
                        }

                        if (fragC[2])
                        {
                            visitedMarks[2].y |= (1ull << f);
                        }

                        if (fragC[3])
                        {
                            visitedMarks[3].y |= (1ull << f);
                        }
                    }

                    // third 64 BFS
                    while (activeSet.z)
                    {
                        unsigned f = (__ffsll(activeSet.z) - 1);
                        activeSet.z &= activeSet.z - 1;

                        unsigned included = ((frontierVertex.z >> f) & 1);
                        MASK origFragB = static_cast<MASK>(warp.ballot(included));
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

                        if (fragC[0])
                        {
                            visitedMarks[0].z |= (1ull << f);
                        }

                        if (fragC[1])
                        {
                            visitedMarks[1].z |= (1ull << f);
                        }

                        if (fragC[2])
                        {
                            visitedMarks[2].z |= (1ull << f);
                        }

                        if (fragC[3])
                        {
                            visitedMarks[3].z |= (1ull << f);
                        }
                    }

                    // fourth 64 BFS
                    while (activeSet.w)
                    {
                        unsigned f = (__ffsll(activeSet.w) - 1);
                        activeSet.w &= activeSet.w - 1;

                        unsigned included = ((frontierVertex.w >> f) & 1);
                        MASK origFragB = static_cast<MASK>(warp.ballot(included));
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

                        if (fragC[0])
                        {
                            visitedMarks[0].w |= (1ull << f);
                        }

                        if (fragC[1])
                        {
                            visitedMarks[1].w |= (1ull << f);
                        }

                        if (fragC[2])
                        {
                            visitedMarks[2].w |= (1ull << f);
                        }

                        if (fragC[3])
                        {
                            visitedMarks[3].w |= (1ull << f);
                        }
                    }

                    unsigned long long* firstAddress = reinterpret_cast<unsigned long long*>(&visitedNext[iter + getVertexIndex(rows.x, partitionSize)]);
                    unsigned long long* secondAddress = reinterpret_cast<unsigned long long*>(&visitedNext[iter + getVertexIndex(rows.y, partitionSize)]);
                    unsigned long long* thirdAddress = reinterpret_cast<unsigned long long*>(&visitedNext[iter + getVertexIndex(rows.z, partitionSize)]);
                    unsigned long long* fourthAddress = reinterpret_cast<unsigned long long*>(&visitedNext[iter + getVertexIndex(rows.w, partitionSize)]);

                    // first vertex
                    if (visitedMarks[0].x)
                    {
                        dirtyBits[0] = true;
                        atomicOr(firstAddress, visitedMarks[0].x);
                    }
                    if (visitedMarks[0].y)
                    {
                        dirtyBits[0] = true;
                        atomicOr(firstAddress + 1, visitedMarks[0].y);
                    }
                    if (visitedMarks[0].z)
                    {
                        dirtyBits[0] = true;
                        atomicOr(firstAddress + 2, visitedMarks[0].z);
                    }
                    if (visitedMarks[0].w)
                    {
                        dirtyBits[0] = true;
                        atomicOr(firstAddress + 3, visitedMarks[0].w);
                    }

                    // second vertex
                    if (visitedMarks[1].x)
                    {
                        dirtyBits[1] = true;
                        atomicOr(secondAddress, visitedMarks[1].x);
                    }
                    if (visitedMarks[1].y)
                    {
                        dirtyBits[1] = true;
                        atomicOr(secondAddress + 1, visitedMarks[1].y);
                    }
                    if (visitedMarks[1].z)
                    {
                        dirtyBits[1] = true;
                        atomicOr(secondAddress + 2, visitedMarks[1].z);
                    }
                    if (visitedMarks[1].w)
                    {
                        dirtyBits[1] = true;
                        atomicOr(secondAddress + 3, visitedMarks[1].w);
                    }

                    // third vertex
                    if (visitedMarks[2].x)
                    {
                        dirtyBits[2] = true;
                        atomicOr(thirdAddress, visitedMarks[2].x);
                    }
                    if (visitedMarks[2].y)
                    {
                        dirtyBits[2] = true;
                        atomicOr(thirdAddress + 1, visitedMarks[2].y);
                    }
                    if (visitedMarks[2].z)
                    {
                        dirtyBits[2] = true;
                        atomicOr(thirdAddress + 2, visitedMarks[2].z);
                    }
                    if (visitedMarks[2].w)
                    {
                        dirtyBits[2] = true;
                        atomicOr(thirdAddress + 3, visitedMarks[2].w);
                    }

                    // fourth vertex
                    if (visitedMarks[3].x)
                    {
                        dirtyBits[3] = true;
                        atomicOr(fourthAddress, visitedMarks[3].x);
                    }
                    if (visitedMarks[3].y)
                    {
                        dirtyBits[3] = true;
                        atomicOr(fourthAddress + 1, visitedMarks[3].y);
                    }
                    if (visitedMarks[3].z)
                    {
                        dirtyBits[3] = true;
                        atomicOr(fourthAddress + 2, visitedMarks[3].z);
                    }
                    if (visitedMarks[3].w)
                    {
                        dirtyBits[3] = true;
                        atomicOr(fourthAddress + 3, visitedMarks[3].w);
                    }
                }

                if (dirtyBits[0])
                {
                    dirtyRSets[rows.x >> 3] = true;
                }
                if (dirtyBits[1])
                {
                    dirtyRSets[rows.y >> 3] = true;
                }
                if (dirtyBits[2])
                {
                    dirtyRSets[rows.z >> 3] = true;
                }
                if (dirtyBits[3])
                {
                    dirtyRSets[rows.w >> 3] = true;
                }
            }
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            grid.sync();
            for (unsigned vBase = threadID << 3; vBase < paddedN; vBase += noThreads << 3)
            {
                unsigned rss = vBase >> 3;
                if (dirtyRSets[rss] == false)
                {
                    continue;
                }
                dirtyRSets[rss] = false;

                bool activeRset = false;
                for (unsigned chunk = 0; chunk < TASK_SIZE; ++chunk)
                {
                    ulonglong4_32a bfsSliceSetMask = {0, 0, 0, 0};

                    for (unsigned vertex = vBase; vertex < vBase + 8; ++vertex)
                    {
                        unsigned rawIndex = getVertexIndex(vertex, partitionSize);
                        unsigned vertexIndex = chunk * paddedN + rawIndex;
                        ulonglong4_32a next = visitedNext[vertexIndex];
                        ulonglong4_32a current = visited[vertexIndex];
                        ulonglong4_32a diff = xor256(current, next);
                        frontier[vertexIndex] = diff;
                        unsigned totalSet = __popcll(diff.x) + __popcll(diff.y) + __popcll(diff.z) + __popcll(diff.w);
                        if (totalSet != 0)
                        {
                            visited[vertexIndex] = next;
                            far[rawIndex] += levelCount * totalSet;
                            bfsSliceSetMask.x |= diff.x;
                            bfsSliceSetMask.y |= diff.y;
                            bfsSliceSetMask.z |= diff.z;
                            bfsSliceSetMask.w |= diff.w;
                        }
                    }
                    activeRSets[chunk * noRealSliceSets + rss] = bfsSliceSetMask;
                    if (bfsSliceSetMask.x || bfsSliceSetMask.y || bfsSliceSetMask.z || bfsSliceSetMask.w)
                    {
                        activeRset = true;
                    }
                }
                if (activeRset)
                {
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
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            grid.sync();
        }
    }

    __global__ void BVSSCloseness8EnhancedSliceSize8NoMasks4LazyChunkFusionSwitching    (
                                                                                        const unsigned*             const   __restrict__ nPtr,
                                                                                        const unsigned*             const   __restrict__ paddedNPtr,
                                                                                        const unsigned*             const   __restrict__ noRealSliceSetsPtr,
                                                                                        const unsigned*             const   __restrict__ rowPtrs,
                                                                                        const unsigned*             const   __restrict__ colIds,
                                                                                        const SLICE_TYPE*           const   __restrict__ sliceSetPtrs,
                                                                                        const unsigned*             const   __restrict__ virtualToReal,
                                                                                        const unsigned*             const   __restrict__ realPtrs,
                                                                                        const unsigned*             const   __restrict__ rowIds,
                                                                                        const MASK*                 const   __restrict__ masks,
                                                                                        // current
                                                                                        unsigned long long*         const   __restrict__ far,
                                                                                        ulonglong4_32a*             const   __restrict__ activeRSets,
                                                                                        bool*                       const   __restrict__ dirtyRSets,
                                                                                        ulonglong4_32a*             const   __restrict__ frontier,
                                                                                        ulonglong4_32a*             const   __restrict__ visited,
                                                                                        unsigned*                           __restrict__ sparseFrontierIds,
                                                                                        unsigned*                           __restrict__ unvisitedCurrentSizePtr,
                                                                                        unsigned*                           __restrict__ frontierCurrentSizePtr,
                                                                                        // next
                                                                                        ulonglong4_32a*             const   __restrict__ visitedNext,
                                                                                        unsigned*                           __restrict__ sparseFrontierNextIds,
                                                                                        unsigned*                           __restrict__ unvisitedNextSizePtr,
                                                                                        unsigned*                           __restrict__ frontierNextSizePtr
                                                                                        /*
                                                                                        // profiling
                                                                                        unsigned long long*         const   __restrict__ levelTime
                                                                                        //
                                                                                        */
                                                                                        )
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        auto block = this_thread_block();

        const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned noThreads = gridDim.x * blockDim.x;
        const unsigned noWarps = noThreads / WARP_SIZE;
        const unsigned warpID = threadID / WARP_SIZE;
        const unsigned laneID = threadID % WARP_SIZE;

        const unsigned n = *nPtr;
        const unsigned paddedN = *paddedNPtr;
        const unsigned noRealSliceSets = *noRealSliceSetsPtr;
        const unsigned partitionSize = paddedN >> 3;

        unsigned levelCount = 0;

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);

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
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            unsigned currentUnvisitedSize = *unvisitedCurrentSizePtr;
            /*
            if (threadID == 0)
            {
                printf("Level: %u - Current unvisited: %u - Current frontier: %u - Pull: %u\n", levelCount, currentUnvisitedSize, currentFrontierSize, currentUnvisitedSize < currentFrontierSize * TASK_SIZE * 2);
            }
            */
            if (currentUnvisitedSize < currentFrontierSize * TASK_SIZE * 2)
            {
                for (unsigned u = threadID; u < n; u += noThreads)
                {
                    for (unsigned chunk = 0; chunk < TASK_SIZE; ++chunk)
                    {
                        unsigned iter = chunk * paddedN;

                        ulonglong4_32a visitedMask = visited[iter + getVertexIndex(u, partitionSize)];
                        ulonglong4_32a visitedNextMask = visitedMask;

                        unsigned long long unvisitedMaskx = ~visitedMask.x;
                        unsigned long long unvisitedMasky = ~visitedMask.y;
                        unsigned long long unvisitedMaskz = ~visitedMask.z;
                        unsigned long long unvisitedMaskw = ~visitedMask.w;

                        if (unvisitedMaskx || unvisitedMasky || unvisitedMaskz || unvisitedMaskw)
                        {
                            for (unsigned nnz = rowPtrs[u]; nnz < rowPtrs[u + 1]; ++nnz)
                            {
                                unsigned v = colIds[nnz];
                                ulonglong4_32a frontierMask = frontier[iter + getVertexIndex(v, partitionSize)];

                                visitedNextMask.x |= (frontierMask.x & unvisitedMaskx);
                                visitedNextMask.y |= (frontierMask.y & unvisitedMasky);
                                visitedNextMask.z |= (frontierMask.z & unvisitedMaskz);
                                visitedNextMask.w |= (frontierMask.w & unvisitedMaskw);
                                if (visitedNextMask.x == UNSIGNED_LONG_MAX && visitedNextMask.y == UNSIGNED_LONG_MAX && visitedNextMask.z == UNSIGNED_LONG_MAX && visitedNextMask.w == UNSIGNED_LONG_MAX)
                                {
                                    break;
                                }
                            }
                            visitedNext[iter + getVertexIndex(u, partitionSize)] = visitedNextMask;
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
                    
                    for (unsigned chunk = 0; chunk < TASK_SIZE; ++chunk)
                    {
                        unsigned iter = chunk * paddedN;

                        ulonglong4_32a frontierVertex;
                        if (laneID < 8)
                        {
                            frontierVertex = frontier[iter + laneID * partitionSize + rset]; // very bad access
                        }
                        else
                        {
                            frontierVertex = {0, 0, 0, 0};
                        }

                        ulonglong4_32a activeSet = activeRSets[chunk * noRealSliceSets + rset];
                        ulonglong4_32a visitedMarks[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
                        // visitedMarks[x][y]: x pulling vertex local ID - y BFS id (bitwise)

                        // first 64 BFS
                        while (activeSet.x)
                        {
                            unsigned f = (__ffsll(activeSet.x) - 1);
                            activeSet.x &= activeSet.x - 1;

                            unsigned included = ((frontierVertex.x >> f) & 1);
                            MASK origFragB = static_cast<MASK>(warp.ballot(included));
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

                            if (fragC[0])
                            {
                                visitedMarks[0].x |= (1ull << f);
                            }

                            if (fragC[1])
                            {
                                visitedMarks[1].x |= (1ull << f);
                            }

                            if (fragC[2])
                            {
                                visitedMarks[2].x |= (1ull << f);
                            }

                            if (fragC[3])
                            {
                                visitedMarks[3].x |= (1ull << f);
                            }
                        }

                        // second 64 BFS
                        while (activeSet.y)
                        {
                            unsigned f = (__ffsll(activeSet.y) - 1);
                            activeSet.y &= activeSet.y - 1;

                            unsigned included = ((frontierVertex.y >> f) & 1);
                            MASK origFragB = static_cast<MASK>(warp.ballot(included));
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

                            if (fragC[0])
                            {
                                visitedMarks[0].y |= (1ull << f);
                            }

                            if (fragC[1])
                            {
                                visitedMarks[1].y |= (1ull << f);
                            }

                            if (fragC[2])
                            {
                                visitedMarks[2].y |= (1ull << f);
                            }

                            if (fragC[3])
                            {
                                visitedMarks[3].y |= (1ull << f);
                            }
                        }

                        // third 64 BFS
                        while (activeSet.z)
                        {
                            unsigned f = (__ffsll(activeSet.z) - 1);
                            activeSet.z &= activeSet.z - 1;

                            unsigned included = ((frontierVertex.z >> f) & 1);
                            MASK origFragB = static_cast<MASK>(warp.ballot(included));
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

                            if (fragC[0])
                            {
                                visitedMarks[0].z |= (1ull << f);
                            }

                            if (fragC[1])
                            {
                                visitedMarks[1].z |= (1ull << f);
                            }

                            if (fragC[2])
                            {
                                visitedMarks[2].z |= (1ull << f);
                            }

                            if (fragC[3])
                            {
                                visitedMarks[3].z |= (1ull << f);
                            }
                        }

                        // fourth 64 BFS
                        while (activeSet.w)
                        {
                            unsigned f = (__ffsll(activeSet.w) - 1);
                            activeSet.w &= activeSet.w - 1;

                            unsigned included = ((frontierVertex.w >> f) & 1);
                            MASK origFragB = static_cast<MASK>(warp.ballot(included));
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

                            if (fragC[0])
                            {
                                visitedMarks[0].w |= (1ull << f);
                            }

                            if (fragC[1])
                            {
                                visitedMarks[1].w |= (1ull << f);
                            }

                            if (fragC[2])
                            {
                                visitedMarks[2].w |= (1ull << f);
                            }

                            if (fragC[3])
                            {
                                visitedMarks[3].w |= (1ull << f);
                            }
                        }

                        unsigned long long* firstAddress = reinterpret_cast<unsigned long long*>(&visitedNext[iter + getVertexIndex(rows.x, partitionSize)]);
                        unsigned long long* secondAddress = reinterpret_cast<unsigned long long*>(&visitedNext[iter + getVertexIndex(rows.y, partitionSize)]);
                        unsigned long long* thirdAddress = reinterpret_cast<unsigned long long*>(&visitedNext[iter + getVertexIndex(rows.z, partitionSize)]);
                        unsigned long long* fourthAddress = reinterpret_cast<unsigned long long*>(&visitedNext[iter + getVertexIndex(rows.w, partitionSize)]);

                        // first vertex
                        if (visitedMarks[0].x)
                        {
                            atomicOr(firstAddress, visitedMarks[0].x);
                        }
                        if (visitedMarks[0].y)
                        {
                            atomicOr(firstAddress + 1, visitedMarks[0].y);
                        }
                        if (visitedMarks[0].z)
                        {
                            atomicOr(firstAddress + 2, visitedMarks[0].z);
                        }
                        if (visitedMarks[0].w)
                        {
                            atomicOr(firstAddress + 3, visitedMarks[0].w);
                        }

                        // second vertex
                        if (visitedMarks[1].x)
                        {
                            atomicOr(secondAddress, visitedMarks[1].x);
                        }
                        if (visitedMarks[1].y)
                        {
                            atomicOr(secondAddress + 1, visitedMarks[1].y);
                        }
                        if (visitedMarks[1].z)
                        {
                            atomicOr(secondAddress + 2, visitedMarks[1].z);
                        }
                        if (visitedMarks[1].w)
                        {
                            atomicOr(secondAddress + 3, visitedMarks[1].w);
                        }

                        // third vertex
                        if (visitedMarks[2].x)
                        {
                            atomicOr(thirdAddress, visitedMarks[2].x);
                        }
                        if (visitedMarks[2].y)
                        {
                            atomicOr(thirdAddress + 1, visitedMarks[2].y);
                        }
                        if (visitedMarks[2].z)
                        {
                            atomicOr(thirdAddress + 2, visitedMarks[2].z);
                        }
                        if (visitedMarks[2].w)
                        {
                            atomicOr(thirdAddress + 3, visitedMarks[2].w);
                        }

                        // fourth vertex
                        if (visitedMarks[3].x)
                        {
                            atomicOr(fourthAddress, visitedMarks[3].x);
                        }
                        if (visitedMarks[3].y)
                        {                            
                            atomicOr(fourthAddress + 1, visitedMarks[3].y);
                        }
                        if (visitedMarks[3].z)
                        {
                            atomicOr(fourthAddress + 2, visitedMarks[3].z);
                        }
                        if (visitedMarks[3].w)
                        {
                            atomicOr(fourthAddress + 3, visitedMarks[3].w);
                        }
                    }
                }
            }
            if (threadID == 0)
            {
                *unvisitedNextSizePtr = 0;
                *frontierNextSizePtr = 0;
            }
            grid.sync();
            unsigned totalUnvisited = 0;
            for (unsigned vBase = threadID << 3; vBase < paddedN; vBase += noThreads << 3)
            {
                unsigned rss = vBase >> 3;

                bool activeRset = false;
                for (unsigned chunk = 0; chunk < TASK_SIZE; ++chunk)
                {
                    ulonglong4_32a bfsSliceSetMask = {0, 0, 0, 0};

                    for (unsigned vertex = vBase; vertex < vBase + 8; ++vertex)
                    {
                        unsigned rawIndex = getVertexIndex(vertex, partitionSize);
                        unsigned vertexIndex = chunk * paddedN + rawIndex;
                        ulonglong4_32a next = visitedNext[vertexIndex];
                        ulonglong4_32a current = visited[vertexIndex];
                        ulonglong4_32a diff = xor256(current, next);
                        frontier[vertexIndex] = diff;
                        unsigned totalSet = __popcll(diff.x) + __popcll(diff.y) + __popcll(diff.z) + __popcll(diff.w);
                        if (totalSet != 0) // vertex is in frontier in at least one BFS
                        {
                            visited[vertexIndex] = next;
                            far[rawIndex] += levelCount * totalSet;
                            bfsSliceSetMask.x |= diff.x;
                            bfsSliceSetMask.y |= diff.y;
                            bfsSliceSetMask.z |= diff.z;
                            bfsSliceSetMask.w |= diff.w;
                        }
                        else
                        {
                            ++totalUnvisited;
                        }
                    }
                    activeRSets[chunk * noRealSliceSets + rss] = bfsSliceSetMask;
                    if (bfsSliceSetMask.x || bfsSliceSetMask.y || bfsSliceSetMask.z || bfsSliceSetMask.w)
                    {
                        activeRset = true;
                    }
                }
                if (activeRset)
                {
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
            swap<unsigned>(unvisitedCurrentSizePtr, unvisitedNextSizePtr);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
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
};
