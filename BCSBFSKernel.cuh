#ifndef BCSBFSKernel_CUH
#define BCSBFSKernel_CUH

#include "BFSKernel.cuh"
#include "BCS.cuh"

namespace BCSBFSKernels
{
    __global__ void BCSBFS8(    const unsigned* const __restrict__ noSliceSetsPtr,
                                const unsigned* const __restrict__ sliceSetPtrs,
                                const unsigned* const __restrict__ colIds,
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
                unsigned tileStart = sliceSetPtrs[sliceSet] / 4;
                unsigned tileEnd = sliceSetPtrs[sliceSet + 1] / 4;
                for (unsigned tilePtr = tileStart; tilePtr < tileEnd; tilePtr += WARP_SIZE)
                {
                    unsigned idx = tilePtr + laneID;
                    uint4 cols = {0, 0, 0, 0};
                    MASK mask = 0;
                    if (idx < tileEnd)
                    {
                        cols = reinterpret_cast<const uint4*>(colIds)[idx];
                        mask = masks[idx];
                    }
                    unsigned fragC[2];
                    fragC[0] = 0; fragC[1] = 0;
                    MASK fragA = mask;
                    MASK fragB =    ((frontier[(cols.x / MASK_BITS)] >> (cols.x % MASK_BITS)) & 0x000000FF)         |
                                    (((frontier[(cols.y / MASK_BITS)] >> (cols.y % MASK_BITS)) & 0x000000FF) << 8)  |
                                    (((frontier[(cols.z / MASK_BITS)] >> (cols.z % MASK_BITS)) & 0x000000FF) << 16) |
                                    (((frontier[(cols.w / MASK_BITS)] >> (cols.w % MASK_BITS)) & 0x000000FF) << 24) ;
                    m8n8k128(fragC, fragA, fragB);
                    int diag = -1;
                    switch (laneID)
                    {
                        case 0: diag = 0; break;
                        case 4: diag = 1; break;
                        case 9: diag = 2; break;
                        case 13: diag = 3; break;
                        case 18: diag = 4; break;
                        case 22: diag = 5; break;
                        case 27: diag = 6; break;
                        case 31: diag = 7; break;
                        default: break;
                    }
                    if (diag >= 0)
                    {
                        unsigned regIndex = (diag % 2 == 0) ? 0 : 1;
                        unsigned regVal = fragC[regIndex];
                        if (regVal)
                        {
                            unsigned row = sliceSet * M + (laneID / 4);
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                atomicAdd(frontierNextSizePtr, 1);
                                atomicOr(&frontierNext[word], temp);
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

    __global__ void BCSBFS32(    const unsigned* const __restrict__ noSliceSetsPtr,
                            const unsigned* const __restrict__ sliceSetPtrs,
                            const unsigned* const __restrict__ colIds,
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
                unsigned start = sliceSetPtrs[sliceSet];
                unsigned end = sliceSetPtrs[sliceSet + 1];
                for (unsigned slicePtr = start; slicePtr < end; slicePtr += WARP_SIZE)
                {
                    unsigned idx = slicePtr + laneID;
                    unsigned col = 0;
                    MASK mask = 0;
                    if (idx < end)
                    {
                        col = colIds[idx];
                        mask = masks[idx];
                    }
                    unsigned fragC[2];
                    fragC[0] = 0; fragC[1] = 0;
                    MASK fragA = mask;
                    MASK fragB = frontier[col / 32];
                    m8n8k128(fragC, fragA, fragB);
                    int diag = -1;
                    switch (laneID)
                    {
                        case 0: diag = 0; break;
                        case 4: diag = 1; break;
                        case 9: diag = 2; break;
                        case 13: diag = 3; break;
                        case 18: diag = 4; break;
                        case 22: diag = 5; break;
                        case 27: diag = 6; break;
                        case 31: diag = 7; break;
                        default: break;
                    }
                    if (diag >= 0)
                    {
                        unsigned regIndex = (diag % 2 == 0) ? 0 : 1;
                        unsigned regVal = fragC[regIndex];
                        if (regVal)
                        {
                            unsigned row = sliceSet * M + (laneID / 4);
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp);
                            if ((old & temp) == 0)
                            {
                                atomicAdd(frontierNextSizePtr, 1);
                                atomicOr(&frontierNext[word], temp);
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

class BCSBFSKernel: public BFSKernel
{
public:
    BCSBFSKernel(BitMatrix* matrix);
    BCSBFSKernel(const BCSBFSKernel& other) = delete;
    BCSBFSKernel(BCSBFSKernel&& other) noexcept = delete;
    BCSBFSKernel& operator=(const BCSBFSKernel& other) = delete;
    BCSBFSKernel& operator=(BCSBFSKernel&& other) noexcept = delete;
    virtual ~BCSBFSKernel() = default;

    virtual double hostCode(unsigned sourceVertex) final;
};

BCSBFSKernel::BCSBFSKernel(BitMatrix* matrix)
: BFSKernel(matrix)
{

}

double BCSBFSKernel::hostCode(unsigned sourceVertex)
{
    BCS* bcs = dynamic_cast<BCS*>(matrix);
    unsigned n = bcs->getN();
    unsigned sliceSize = bcs->getSliceSize();
    unsigned noSliceSets = bcs->getNoSliceSets();
    unsigned* sliceSetPtrs = bcs->getSliceSetPtrs();
    unsigned* colIds = bcs->getColIds();
    MASK* masks = bcs->getMasks();

    void* kernelPtr = nullptr;
    if (sliceSize == 8)
    {
        kernelPtr = (void*)BCSBFSKernels::BCSBFS8;
    }
    else if (sliceSize == 32)
    {
        kernelPtr = (void*)BCSBFSKernels::BCSBFS32;
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
    unsigned* d_ColIds;
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
    gpuErrchk(cudaMalloc(&d_ColIds, sizeof(unsigned) * sliceSetPtrs[noSliceSets]))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (sliceSetPtrs[noSliceSets] / noMasks)))

    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_ColIds, colIds, sizeof(unsigned) * sliceSetPtrs[noSliceSets], cudaMemcpyHostToDevice))
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
                            (void*)&d_ColIds,
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
    gpuErrchk(cudaFree(d_ColIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_NoWords))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_FrontierNext))

    return (end - start);
}

#endif
