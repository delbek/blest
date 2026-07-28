/*
 * This file is part of the BLEST repository: https://github.com/delbek/blest
 * Author: Deniz Elbek
 *
 * Please see the papers:
 * 
 * @inproceedings{elbek2026blest,
 *   author    = {Elbek, Deniz and Kaya, Kamer},
 *   title     = {BLEST: Blazingly Efficient BFS using Tensor Cores},
 *   booktitle = {Proceedings of the 40th ACM International Conference on Supercomputing},
 *   series    = {ICS '26},
 *   year      = {2026},
 *   pages     = {714--726},
 *   doi       = {10.1145/3797905.3800531},
 *   url       = {https://dl.acm.org/doi/10.1145/3797905.3800531},
 *   publisher = {ACM},
 *   address   = {New York, NY, USA}
 * }

 * @article{elbek2026bfs,
 *   author  = {Elbek, Deniz and Kaya, Kamer},
 *   title   = {Graph Traversal on Tensor Cores: A BFS Framework for Modern GPUs},
 *   journal = {arXiv preprint arXiv:2606.05081},
 *   year    = {2026},
 *   doi     = {10.48550/arXiv.2606.05081},
 *   url     = {https://arxiv.org/abs/2606.05081}
 * }
 */

#pragma once

#include "Common.cuh"

namespace BVSSCCKernels
{
    template<typename T>
    __device__ __forceinline__ void swap(T* __restrict__& ptr1, T* __restrict__& ptr2)
    {
        T* temp = ptr2;
        ptr2 = ptr1;
        ptr1 = temp;
    }

    __global__ void BVSSCC8EnhancedSliceSize8NoMasks4Lazy   (
                                                            const SLICE_TYPE* const __restrict__ sliceSetPtrs,
                                                            const unsigned*   const __restrict__ virtualToReal,
                                                            const unsigned*   const __restrict__ realPtrs,
                                                            const unsigned*   const __restrict__ rowIds,
                                                            const MASK*       const __restrict__ masks,
                                                            const unsigned*   const __restrict__ noWordsPtr,
                                                            // current
                                                            unsigned*         const __restrict__ components,
                                                            unsigned*         const __restrict__ frontier,
                                                            unsigned*         const __restrict__ marker,
                                                            unsigned*               __restrict__ sparseFrontierIds,
                                                            unsigned*               __restrict__ frontierCurrentSizePtr,
                                                            // next
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

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);

        bool cont = true;
        while (cont)
        {
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
                    unsigned char validVertices = (mask & 0xFF) & origFragB;
                    unsigned minComponentID = components[rows.x];
                    bool updated = false;
                    while (validVertices)
                    {
                        unsigned vertex = (rset << 3) + (__ffs(validVertices) - 1);
                        unsigned nbComponentID = components[vertex];
                        if (nbComponentID < minComponentID)
                        {
                            updated = true;
                            minComponentID = nbComponentID;
                        }
                        validVertices &= validVertices - 1;
                    }
                    if (updated)
                    {
                        unsigned oldComponentID = atomicMin(components + rows.x, minComponentID);
                        if (minComponentID < oldComponentID)
                        {
                            atomicOr(&marker[word], temp);
                        }
                    }
                }

                word = rows.y >> 5;
                bit = rows.y & 31;
                temp = (1u << bit);
                if (fragC[1])
                {
                    unsigned char validVertices = ((mask >> 8) & 0xFF) & origFragB;
                    unsigned minComponentID = components[rows.y];
                    bool updated = false;
                    while (validVertices)
                    {
                        unsigned vertex = (rset << 3) + (__ffs(validVertices) - 1);
                        unsigned nbComponentID = components[vertex];
                        if (nbComponentID < minComponentID)
                        {
                            updated = true;
                            minComponentID = nbComponentID;
                        }
                        validVertices &= validVertices - 1;
                    }
                    if (updated)
                    {
                        unsigned oldComponentID = atomicMin(components + rows.y, minComponentID);
                        if (minComponentID < oldComponentID)
                        {
                            atomicOr(&marker[word], temp);
                        }
                    }
                }

                word = rows.z >> 5;
                bit = rows.z & 31;
                temp = (1u << bit);
                if (fragC[2])
                {
                    unsigned char validVertices = ((mask >> 16) & 0xFF) & origFragB;
                    unsigned minComponentID = components[rows.z];
                    bool updated = false;
                    while (validVertices)
                    {
                        unsigned vertex = (rset << 3) + (__ffs(validVertices) - 1);
                        unsigned nbComponentID = components[vertex];
                        if (nbComponentID < minComponentID)
                        {
                            updated = true;
                            minComponentID = nbComponentID;
                        }
                        validVertices &= validVertices - 1;
                    }
                    if (updated)
                    {
                        unsigned oldComponentID = atomicMin(components + rows.z, minComponentID);
                        if (minComponentID < oldComponentID)
                        {
                            atomicOr(&marker[word], temp);
                        }
                    }
                }

                word = rows.w >> 5;
                bit = rows.w & 31;
                temp = (1u << bit);
                if (fragC[3])
                {
                    unsigned char validVertices = ((mask >> 24) & 0xFF) & origFragB;
                    unsigned minComponentID = components[rows.w];
                    bool updated = false;
                    while (validVertices)
                    {
                        unsigned vertex = (rset << 3) + (__ffs(validVertices) - 1);
                        unsigned nbComponentID = components[vertex];
                        if (nbComponentID < minComponentID)
                        {
                            updated = true;
                            minComponentID = nbComponentID;
                        }
                        validVertices &= validVertices - 1;
                    }
                    if (updated)
                    {
                        unsigned oldComponentID = atomicMin(components + rows.w, minComponentID);
                        if (minComponentID < oldComponentID)
                        {
                            atomicOr(&marker[word], temp);
                        }
                    }
                }
            }

            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            grid.sync();
            for (unsigned i = threadID; i < noWords; i += noThreads)
            {
                unsigned mark = marker[i];
                frontier[i] = mark;
                unsigned rssOffset = i << 2;
                if (mark != 0)
                {
                    #pragma unroll 4
                    for (unsigned set = 0; set < 4; ++set)
                    {
                        MASK sliceMask = ((mark >> (set << 3)) & 0xFF);
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
                marker[i] = 0;
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            grid.sync();
        }
    }

    __global__ void BVSSCC8EnhancedSliceSize8NoMasks4LazyFullPad    (
                                                                    const SLICE_TYPE* const __restrict__ sliceSetPtrs,
                                                                    const unsigned*   const __restrict__ virtualToReal,
                                                                    const unsigned*   const __restrict__ realPtrs,
                                                                    const unsigned*   const __restrict__ rowIds,
                                                                    const MASK*       const __restrict__ masks,
                                                                    const unsigned*   const __restrict__ noWordsPtr,
                                                                    // current
                                                                    unsigned*         const __restrict__ components,
                                                                    unsigned*         const __restrict__ frontier,
                                                                    unsigned*         const __restrict__ marker,
                                                                    unsigned*               __restrict__ sparseFrontierIds,
                                                                    unsigned*               __restrict__ frontierCurrentSizePtr,
                                                                    // next
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

        const uint4* row4Ids = reinterpret_cast<const uint4*>(rowIds);

        bool cont = true;
        while (cont)
        {
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
                    unsigned char validVertices = (mask & 0xFF) & origFragB;
                    unsigned minComponentID = components[rows.x];
                    bool updated = false;
                    while (validVertices)
                    {
                        unsigned vertex = (rset << 3) + (__ffs(validVertices) - 1);
                        unsigned nbComponentID = components[vertex];
                        if (nbComponentID < minComponentID)
                        {
                            updated = true;
                            minComponentID = nbComponentID;
                        }
                        validVertices &= validVertices - 1;
                    }
                    if (updated)
                    {
                        unsigned oldComponentID = atomicMin(components + rows.x, minComponentID);
                        if (minComponentID < oldComponentID)
                        {
                            atomicOr(&marker[word], temp);
                        }
                    }
                }

                word = rows.y >> 5;
                bit = rows.y & 31;
                temp = (1u << bit);
                if (fragC[1])
                {
                    unsigned char validVertices = ((mask >> 8) & 0xFF) & origFragB;
                    unsigned minComponentID = components[rows.y];
                    bool updated = false;
                    while (validVertices)
                    {
                        unsigned vertex = (rset << 3) + (__ffs(validVertices) - 1);
                        unsigned nbComponentID = components[vertex];
                        if (nbComponentID < minComponentID)
                        {
                            updated = true;
                            minComponentID = nbComponentID;
                        }
                        validVertices &= validVertices - 1;
                    }
                    if (updated)
                    {
                        unsigned oldComponentID = atomicMin(components + rows.y, minComponentID);
                        if (minComponentID < oldComponentID)
                        {
                            atomicOr(&marker[word], temp);
                        }
                    }
                }

                word = rows.z >> 5;
                bit = rows.z & 31;
                temp = (1u << bit);
                if (fragC[2])
                {
                    unsigned char validVertices = ((mask >> 16) & 0xFF) & origFragB;
                    unsigned minComponentID = components[rows.z];
                    bool updated = false;
                    while (validVertices)
                    {
                        unsigned vertex = (rset << 3) + (__ffs(validVertices) - 1);
                        unsigned nbComponentID = components[vertex];
                        if (nbComponentID < minComponentID)
                        {
                            updated = true;
                            minComponentID = nbComponentID;
                        }
                        validVertices &= validVertices - 1;
                    }
                    if (updated)
                    {
                        unsigned oldComponentID = atomicMin(components + rows.z, minComponentID);
                        if (minComponentID < oldComponentID)
                        {
                            atomicOr(&marker[word], temp);
                        }
                    }
                }

                word = rows.w >> 5;
                bit = rows.w & 31;
                temp = (1u << bit);
                if (fragC[3])
                {
                    unsigned char validVertices = ((mask >> 24) & 0xFF) & origFragB;
                    unsigned minComponentID = components[rows.w];
                    bool updated = false;
                    while (validVertices)
                    {
                        unsigned vertex = (rset << 3) + (__ffs(validVertices) - 1);
                        unsigned nbComponentID = components[vertex];
                        if (nbComponentID < minComponentID)
                        {
                            updated = true;
                            minComponentID = nbComponentID;
                        }
                        validVertices &= validVertices - 1;
                    }
                    if (updated)
                    {
                        unsigned oldComponentID = atomicMin(components + rows.w, minComponentID);
                        if (minComponentID < oldComponentID)
                        {
                            atomicOr(&marker[word], temp);
                        }
                    }
                }
            }

            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            grid.sync();
            for (unsigned i = threadID; i < noWords; i += noThreads)
            {
                unsigned mark = marker[i];
                frontier[i] = mark;
                unsigned rssOffset = i << 2;
                if (mark != 0)
                {
                    #pragma unroll 4
                    for (unsigned set = 0; set < 4; ++set)
                    {
                        MASK sliceMask = ((mark >> (set << 3)) & 0xFF);
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
                marker[i] = 0;
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<unsigned>(sparseFrontierIds, sparseFrontierNextIds);
            swap<unsigned>(frontierCurrentSizePtr, frontierNextSizePtr);
            grid.sync();
        }
    }
};
