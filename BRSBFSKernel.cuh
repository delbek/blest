#ifndef BRSBFSKernel_CUH
#define BRSBFSKernel_CUH

#include "BFSKernel.cuh"
#include "BRS.cuh"

namespace BRSBFSKernels
{
    __global__ void BRSBFS8Enhanced(    const unsigned* const __restrict__ noSliceSetsPtr,
                                        const unsigned* const __restrict__ sliceSetPtrs,
                                        const unsigned* const __restrict__ rowIds,
                                        const MASK* const __restrict__ masks,
                                        const unsigned* const __restrict__ noWordsPtr,
                                        MASK* __restrict__ frontier,
                                        MASK* const __restrict__ visited,
                                        unsigned* const __restrict__ frontierNextSizePtr,
                                        MASK* __restrict__ frontierNext)
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned noThreads = gridDim.x * blockDim.x;
        unsigned noWarps = noThreads / WARP_SIZE;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWords = *noWordsPtr;
        unsigned noSliceSets = *noSliceSetsPtr;

        bool cont = true;
        while (cont)
        {
            for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
            {
                unsigned shift = (sliceSet % 4) * 8;
                MASK origFragB = ((frontier[sliceSet / 4] >> shift) & 0x000000FF);
                if (origFragB)
                {
                    unsigned tileStart = sliceSetPtrs[sliceSet] / 4;
                    unsigned tileEnd = sliceSetPtrs[sliceSet + 1] / 4;
                    for (unsigned tilePtr = tileStart; tilePtr < tileEnd; tilePtr += WARP_SIZE)
                    {
                        unsigned tile = tilePtr + laneID;
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

            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            grid.sync();
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }

            if (cont)
            {
                // frontier swap
                MASK* temp = frontier;
                frontier = frontierNext;
                frontierNext = temp;
                for (unsigned i = threadID; i < noWords; i += noThreads)
                {
                    frontierNext[i] = 0;
                }
                //
            }
            grid.sync();
        }
    }

    __global__ void BRSBFS32Enhanced(   const unsigned* const __restrict__ noSliceSetsPtr,
                                        const unsigned* const __restrict__ sliceSetPtrs,
                                        const unsigned* const __restrict__ rowIds,
                                        const MASK* const __restrict__ masks,
                                        const unsigned* const __restrict__ noWordsPtr,
                                        MASK* __restrict__ frontier,
                                        MASK* const __restrict__ visited,
                                        unsigned* const __restrict__ frontierNextSizePtr,
                                        MASK* __restrict__ frontierNext)
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned noThreads = gridDim.x * blockDim.x;
        unsigned noWarps = noThreads / WARP_SIZE;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWords = *noWordsPtr;
        unsigned noSliceSets = *noSliceSetsPtr;

        bool cont = true;
        while (cont)
        {
            for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
            {
                MASK fragB = frontier[sliceSet];
                if (fragB)
                {
                    unsigned start = sliceSetPtrs[sliceSet];
                    unsigned end = sliceSetPtrs[sliceSet + 1];
                    for (unsigned slicePtr = start; slicePtr < end; slicePtr += WARP_SIZE)
                    {
                        unsigned slice = slicePtr + laneID;
                        unsigned row = 0;
                        MASK fragA = 0;
                        if (slice < end)
                        {
                            row = rowIds[slice];
                            fragA = masks[slice];
                        }
                        
                        if (laneID % 9 != 0)
                        {
                            fragB = 0;
                        }

                        unsigned fragC[2];
                        fragC[0] = 0;

                        m8n8k128(fragC, fragA, fragB);
                        if (fragC[0])
                        {
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                old = atomicOr(&frontierNext[word], temp);
                                if (old == 0)
                                {
                                    atomicAdd(frontierNextSizePtr, 1);
                                }
                            }
                        }
                    }
                }
            }

            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            grid.sync();
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }

            if (cont)
            {
                // frontier swap
                MASK* temp = frontier;
                frontier = frontierNext;
                frontierNext = temp;
                for (unsigned i = threadID; i < noWords; i += noThreads)
                {
                    frontierNext[i] = 0;
                }
                //
            }
            grid.sync();
        }
    }

    __global__ void BRSBFS8(    const unsigned* const __restrict__ noSliceSetsPtr,
                                const unsigned* const __restrict__ sliceSetPtrs,
                                const unsigned* const __restrict__ rowIds,
                                const MASK* const __restrict__ masks,
                                const unsigned* const __restrict__ noWordsPtr,
                                MASK* __restrict__ frontier,
                                MASK* const __restrict__ visited,
                                unsigned* const __restrict__ frontierNextSizePtr,
                                MASK* __restrict__ frontierNext)
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned noThreads = gridDim.x * blockDim.x;
        unsigned noWarps = noThreads / WARP_SIZE;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWords = *noWordsPtr;
        unsigned noSliceSets = *noSliceSetsPtr;

        unsigned threadIDInGroup = laneID % 4;

        bool cont = true;
        while (cont)
        {
            for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
            {
                unsigned shift = (sliceSet % 4) * 8;
                MASK fragB = ((frontier[sliceSet / 4] >> shift) & 0x000000FF);
                if (fragB)
                {
                    fragB *= 0x01010101;
                    unsigned tileStart = sliceSetPtrs[sliceSet] / 4;
                    unsigned tileEnd = sliceSetPtrs[sliceSet + 1] / 4;
                    for (unsigned tilePtr = tileStart; tilePtr < tileEnd; tilePtr += WARP_SIZE)
                    {
                        unsigned tile = tilePtr + laneID;
                        uint4 rows = {0, 0, 0, 0};
                        MASK mask = 0;
                        if (tile < tileEnd)
                        {
                            rows = reinterpret_cast<const uint4*>(rowIds)[tile];
                            mask = masks[tile];
                        }
                        unsigned fragC[2];
                        for (unsigned round = 0; round < 4; ++round)
                        {
                            fragC[0] = 0;
                            MASK fragA = threadIDInGroup == round ? (mask & 0x000000FF) : 0;
                            m8n8k128(fragC, fragA, fragB);
                            if (threadIDInGroup == round && fragC[0])
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
                        }
                        for (unsigned round = 0; round < 4; ++round)
                        {
                            fragC[0] = 0;
                            MASK fragA = threadIDInGroup == round ? (mask & 0x0000FF00) : 0;
                            m8n8k128(fragC, fragA, fragB);
                            if (threadIDInGroup == round && fragC[0])
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
                        }
                        for (unsigned round = 0; round < 4; ++round)
                        {
                            fragC[0] = 0;
                            MASK fragA = threadIDInGroup == round ? (mask & 0x00FF0000) : 0;
                            m8n8k128(fragC, fragA, fragB);
                            if (threadIDInGroup == round && fragC[0])
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
                        }
                        for (unsigned round = 0; round < 4; ++round)
                        {
                            fragC[0] = 0;
                            MASK fragA = threadIDInGroup == round ? (mask & 0xFF000000) : 0;
                            m8n8k128(fragC, fragA, fragB);
                            if (threadIDInGroup == round && fragC[0])
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

            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            grid.sync();
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }

            if (cont)
            {
                // frontier swap
                MASK* temp = frontier;
                frontier = frontierNext;
                frontierNext = temp;
                for (unsigned i = threadID; i < noWords; i += noThreads)
                {
                    frontierNext[i] = 0;
                }
                //
            }
            grid.sync();
        }
    }

    __global__ void BRSBFS32(   const unsigned* const __restrict__ noSliceSetsPtr,
                                const unsigned* const __restrict__ sliceSetPtrs,
                                const unsigned* const __restrict__ rowIds,
                                const MASK* const __restrict__ masks,
                                const unsigned* const __restrict__ noWordsPtr,
                                MASK* __restrict__ frontier,
                                MASK* const __restrict__ visited,
                                unsigned* const __restrict__ frontierNextSizePtr,
                                MASK* __restrict__ frontierNext)
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned noThreads = gridDim.x * blockDim.x;
        unsigned noWarps = noThreads / WARP_SIZE;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWords = *noWordsPtr;
        unsigned noSliceSets = *noSliceSetsPtr;

        unsigned threadIDInGroup = laneID % 4;

        bool cont = true;
        while (cont)
        {
            for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
            {
                MASK fragB = frontier[sliceSet]; // we do not use the leftover 124 bytes that we fetched from L2, perhaps not big of a deal but try to optimize it later
                if (fragB)
                {
                    unsigned start = sliceSetPtrs[sliceSet];
                    unsigned end = sliceSetPtrs[sliceSet + 1];
                    for (unsigned slicePtr = start; slicePtr < end; slicePtr += WARP_SIZE)
                    {
                        unsigned slice = slicePtr + laneID;
                        unsigned row = 0;
                        MASK mask = 0;
                        if (slice < end)
                        {
                            row = rowIds[slice];
                            mask = masks[slice];
                        }
                        unsigned fragC[2];
                        for (unsigned round = 0; round < 4; ++round)
                        {
                            fragC[0] = 0;
                            MASK fragA = threadIDInGroup == round ? mask : 0;
                            m8n8k128(fragC, fragA, fragB);
                            if (threadIDInGroup == round && fragC[0])
                            {
                                unsigned word = row / MASK_BITS;
                                unsigned bit = row % MASK_BITS;
                                MASK temp = (static_cast<MASK>(1) << bit);
                                MASK old = atomicOr(&visited[word], temp); // this visited[word] and the below frontierNext[word] access is quite problematic in that rows each thread is assigned to in the warp can differ dramatically. Smth ordering may fix.
                                if ((old & temp) == 0)
                                {
                                    old = atomicOr(&frontierNext[word], temp);
                                    if (old == 0) // be careful that when slice size changes this code must be altered as well
                                    {
                                        atomicAdd(frontierNextSizePtr, 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            grid.sync();
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }

            if (cont)
            {
                // frontier swap
                MASK* temp = frontier;
                frontier = frontierNext;
                frontierNext = temp;
                for (unsigned i = threadID; i < noWords; i += noThreads)
                {
                    frontierNext[i] = 0;
                }
                //
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
    unsigned noSliceSets = brs->getNoSliceSets();
    unsigned* sliceSetPtrs = brs->getSliceSetPtrs();
    unsigned* rowIds = brs->getRowIds();
    MASK* masks = brs->getMasks();

    void* kernelPtr = nullptr;
    if (sliceSize == 8)
    {
        kernelPtr = (void*)BRSBFSKernels::BRSBFS8Enhanced;
    }
    else if (sliceSize == 32)
    {
        kernelPtr = (void*)BRSBFSKernels::BRSBFS32Enhanced;
    }
    else
    {
        throw std::runtime_error("No appropriate kernel found meeting the selected slice size.");
    }

    int gridSize, blockSize;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(
                                                &gridSize, 
                                                &blockSize, 
                                                kernelPtr,
                                                0,
                                                0));

    unsigned* d_NoSliceSets;
    unsigned* d_SliceSetPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    unsigned* d_NoWords;
    MASK* d_Frontier;
    MASK* d_Visited;
    unsigned* d_FrontierNextSize;
    MASK* d_FrontierNext;

    unsigned noMasks = MASK_BITS / sliceSize;
    // data structure
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * sliceSetPtrs[noSliceSets]))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (sliceSetPtrs[noSliceSets] / noMasks)))

    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * sliceSetPtrs[noSliceSets], cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (sliceSetPtrs[noSliceSets] / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    unsigned noWords = (n + MASK_BITS - 1) / MASK_BITS;
    gpuErrchk(cudaMalloc(&d_NoWords, sizeof(unsigned)))
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
    gpuErrchk(cudaMemcpy(d_NoWords, &noWords, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Frontier + word, &temp, sizeof(MASK), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Visited + word, &temp, sizeof(MASK), cudaMemcpyHostToDevice))

    double start = omp_get_wtime();
    void* kernelArgs[] = {
                            (void*)&d_NoSliceSets,
                            (void*)&d_SliceSetPtrs,
                            (void*)&d_RowIds,
                            (void*)&d_Masks,
                            (void*)&d_NoWords,
                            (void*)&d_Frontier,
                            (void*)&d_Visited,
                            (void*)&d_FrontierNextSize,
                            (void*)&d_FrontierNext};
    gpuErrchk(cudaLaunchCooperativeKernel(
                                            kernelPtr,
                                            gridSize,
                                            blockSize,
                                            kernelArgs,
                                            0,
                                            0))
    gpuErrchk(cudaDeviceSynchronize())
    double end = omp_get_wtime();

    unsigned* visited = new unsigned[noWords];
    gpuErrchk(cudaMemcpy(visited, d_Visited, sizeof(MASK) * noWords, cudaMemcpyDeviceToHost))
    unsigned totalVisited = 0;
    for (unsigned i = 0; i < noWords; ++i)
    {
        totalVisited += __builtin_popcount(visited[i]);
    }
    delete[] visited;
    std::cout << "Total traversed number of nodes: " << totalVisited << std::endl;

    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_NoWords))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_FrontierNext))

    return (end - start);
}

#endif
