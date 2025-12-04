#pragma once

#include "BFSKernel.cuh"
#include "BRS.cuh"
#include <array>

namespace BRSBFSKernels
{
    template<typename T>
    __device__ __forceinline__ void swap(T* __restrict__& ptr1, T* __restrict__& ptr2)
    {
        T* temp = ptr2;
        ptr2 = ptr1;
        ptr1 = temp;
    }

    __global__ void BRSBFS8EnhancedSliceSize8NoMasks4Social(
                                                            const unsigned* const __restrict__ sliceSetPtrs,
                                                            const unsigned* const __restrict__ virtualToReal,
                                                            const unsigned* const __restrict__ realPtrs,
                                                            const unsigned* const __restrict__ rowIds,
                                                            const MASK*     const __restrict__ masks,
                                                            const unsigned* const __restrict__ noWordsPtr,
                                                            // current
                                                            unsigned*       const __restrict__ levels,
                                                            unsigned*       const __restrict__ frontier,
                                                            unsigned*       const __restrict__ visited,
                                                            unsigned*             __restrict__ sparseFrontierIds,
                                                            unsigned*             __restrict__ frontierCurrentSizePtr,
                                                            // next
                                                            unsigned*       const __restrict__ visitedNext,
                                                            unsigned*             __restrict__ sparseFrontierNextIds,
                                                            unsigned*             __restrict__ frontierNextSizePtr
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

                #ifdef FULL_PADDING
                unsigned tile = (vset << 5) + laneID;
                uint4 rows = row4Ids[tile];
                MASK mask = masks[tile];
                #else
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
                #endif

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
                                unsigned vertex = (rss << 3) + ((__ffs(sliceMask) - 1) & 7);
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
        }
    }

    __global__ void BRSBFS8EnhancedSliceSize8NoMasks4Road(
                                                            const unsigned* const __restrict__ sliceSetPtrs,
                                                            const unsigned* const __restrict__ virtualToReal,
                                                            const unsigned* const __restrict__ realPtrs,
                                                            const unsigned* const __restrict__ rowIds,
                                                            const MASK*     const __restrict__ masks,
                                                            const unsigned* const __restrict__ noWordsPtr,
                                                            // current
                                                            unsigned*       const __restrict__ levels,
                                                            unsigned*             __restrict__ frontier,
                                                            unsigned*             __restrict__ sparseFrontierIds,
                                                            unsigned*             __restrict__ frontierCurrentSizePtr,
                                                            // next
                                                            unsigned*             __restrict__ frontierNext,
                                                            unsigned*             __restrict__ sparseFrontierNextIds,
                                                            unsigned*             __restrict__ frontierNextSizePtr
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

                #ifdef FULL_PADDING
                unsigned tile = (vset << 5) + laneID;
                uint4 rows = row4Ids[tile];
                MASK mask = masks[tile];
                #else
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
                #endif

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
                    unsigned oldLevel = levels[rows.x];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.x] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
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

                word = rows.y >> 5;
                bit = rows.y & 31;
                temp = (1 << bit);
                if (fragC[1])
                {
                    unsigned oldLevel = levels[rows.y];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.y] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
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

                word = rows.z >> 5;
                bit = rows.z & 31;
                temp = (1 << bit);
                if (fragC[2])
                {
                    unsigned oldLevel = levels[rows.z];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.z] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
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

                word = rows.w >> 5;
                bit = rows.w & 31;
                temp = (1 << bit);
                if (fragC[3])
                {
                    unsigned oldLevel = levels[rows.w];
                    if (levelCount < oldLevel)
                    {
                        levels[rows.w] = levelCount;
                        unsigned old = atomicOr(&frontierNext[word], temp);
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
    unsigned noMasks = brs->getNoMasks();
    unsigned noRealSliceSets = brs->getNoRealSliceSets();
    unsigned noSlices = brs->getNoSlices();
    bool isSocialNetwork = brs->IsSocialNetwork();
    unsigned noSliceSets = brs->getNoVirtualSliceSets();
    unsigned* sliceSetPtrs = brs->getSliceSetPtrs();
    unsigned* virtualToReal = brs->getVirtualToReal();
    unsigned* realPtrs = brs->getRealPtrs();
    unsigned* rowIds = brs->getRowIds();
    MASK* masks = brs->getMasks();
    const unsigned DIRECTION_THRESHOLD = noSliceSets * DIRECTION_SWITCHING_CONSTANT; // vset- or rset- based?

    BFSResult result;
    result.sourceVertex = sourceVertex;
    result.levels = new unsigned[n];
    std::fill(result.levels, result.levels + n, UNSIGNED_MAX);
    result.levels[sourceVertex] = 0;

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    void* kernelPtr;
    if (sliceSize == 8)
    {
        if (isSocialNetwork)
        {
            kernelPtr = (void*)BRSBFSKernels::BRSBFS8EnhancedSliceSize8NoMasks4Social;
        }
        else
        {
            kernelPtr = (void*)BRSBFSKernels::BRSBFS8EnhancedSliceSize8NoMasks4Road;
        }
    }
    else
    {
        throw std::runtime_error("No appropriate kernel found meeting the selected slice size and noMasks.");
    }

    gpuErrchk(cudaFuncSetAttribute(
        kernelPtr,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        0))

    int gridSize, blockSize;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
                                                &gridSize, 
                                                &blockSize, 
                                                kernelPtr,
                                                allocateSharedMemory,
                                                0))

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
    unsigned* d_VisitedNext;
    unsigned* d_FrontierNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_FrontierNextSize;
    unsigned* d_Visited;
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
    unsigned noWords = (n + 31) / 32;
    gpuErrchk(cudaMalloc(&d_NoWords, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_DIRECTION_THRESHOLD, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets)) // storing vset or rset?
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_VisitedNext, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets)) // storing vset or rset?
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_Levels, sizeof(unsigned) * n))

    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_VisitedNext, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(unsigned) * noWords))

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

    unsigned word = sourceVertex >> 5;
    unsigned bit = sourceVertex & 31;
    unsigned temp = (1 << bit);
    gpuErrchk(cudaMemcpy(d_Frontier + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Visited + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VisitedNext + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))

    double start;
    if (!isSocialNetwork)
    {
        std::array<void*, 13> argsA =
        {
            (void*)&d_SliceSetPtrs,
            (void*)&d_VirtualToReal,
            (void*)&d_RealPtrs,
            (void*)&d_RowIds,
            (void*)&d_Masks,
            (void*)&d_NoWords,
            (void*)&d_Levels,
            (void*)&d_Frontier,
            (void*)&d_SparseFrontierIds,
            (void*)&d_FrontierCurrentSize,
            (void*)&d_FrontierNext,
            (void*)&d_SparseFrontierNextIds,
            (void*)&d_FrontierNextSize
        };

        start = omp_get_wtime();
        gpuErrchk(cudaLaunchCooperativeKernel(
            kernelPtr,
            gridSize,
            blockSize,
            argsA.data(),
            allocateSharedMemory(blockSize),
            0))
    }
    else
    {
        std::array<void*, 14> argsB =
        {
            (void*)&d_SliceSetPtrs,
            (void*)&d_VirtualToReal,
            (void*)&d_RealPtrs,
            (void*)&d_RowIds,
            (void*)&d_Masks,
            (void*)&d_NoWords,
            (void*)&d_Levels,
            (void*)&d_Frontier,
            (void*)&d_Visited,
            (void*)&d_SparseFrontierIds,
            (void*)&d_FrontierCurrentSize,
            (void*)&d_VisitedNext,
            (void*)&d_SparseFrontierNextIds,
            (void*)&d_FrontierNextSize
        };

        start = omp_get_wtime();
        gpuErrchk(cudaLaunchCooperativeKernel(
            kernelPtr,
            gridSize,
            blockSize,
            argsB.data(),
            allocateSharedMemory(blockSize),
            0))
    }
    gpuErrchk(cudaPeekAtLastError())
    gpuErrchk(cudaDeviceSynchronize())
    double end = omp_get_wtime();

    result.time = (end - start);

    gpuErrchk(cudaMemcpy(result.levels, d_Levels, sizeof(unsigned) * n, cudaMemcpyDeviceToHost))
    result.totalLevels = 0;
    result.noVisited = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        if (result.levels[i] != UNSIGNED_MAX)
        {
            result.totalLevels = std::max(result.totalLevels, result.levels[i]);
            ++result.noVisited;
        }
    }

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
    gpuErrchk(cudaFree(d_VisitedNext))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_Levels))
    gpuErrchk(cudaFree(d_FrontierNext))

    return result;
}
