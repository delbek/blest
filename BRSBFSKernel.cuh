#ifndef BRSBFSKernel_CUH
#define BRSBFSKernel_CUH

#include "BFSKernel.cuh"
#include "BRS.cuh"

namespace BRSBFSKernels
{
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
        unsigned noWarps = (gridDim.x * blockDim.x) / WARP_SIZE;
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWords = *noWordsPtr;
        unsigned noSliceSets = *noSliceSetsPtr;

        unsigned threadIDInGroup = laneID % 4;

        bool cont = true;
        while (cont)
        {
            unsigned myFrontierNextSize = 0;
            for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
            {
                MASK fragB = frontier[sliceSet * 4 + threadIDInGroup]; // we do not use the leftover 112 bytes that we fetched from L2, perhaps not big of a deal but try to optimize it later
                bool any = (fragB != 0);
                bool cont = warp.any(any);
                if (cont)
                {
                    unsigned start = sliceSetPtrs[sliceSet];
                    unsigned end = sliceSetPtrs[sliceSet + 1];
                    for (unsigned slicePtr = start; slicePtr < end; slicePtr += WARP_SIZE)
                    {
                        unsigned idx = slicePtr + laneID;
                        unsigned row = 0;
                        MASK mask = 0;
                        if (idx < end)
                        {
                            row = rowIds[idx];
                            mask = masks[idx];
                        }
                        unsigned fragC[2];
                        for (unsigned round = 0; round < 4; ++round)
                        {
                            any = (threadIDInGroup == round && fragB != 0) ? true : false;
                            cont = warp.any(any);
                            if (cont)
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
                                        ++myFrontierNextSize;
                                        atomicOr(&frontierNext[word], temp);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (unsigned offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1)
            {
                myFrontierNextSize += warp.shfl_down(myFrontierNextSize, offset);
            }
            if (laneID == 0)
            {
                atomicAdd(frontierNextSizePtr, myFrontierNextSize);
            }
            grid.sync();
            if (laneID == 0)
            {
                myFrontierNextSize = *frontierNextSizePtr;
            }
            grid.sync();
            myFrontierNextSize = warp.shfl(myFrontierNextSize, 0);
            cont = (myFrontierNextSize != 0);
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            MASK* temp = frontier;
            frontier = frontierNext;
            frontierNext = temp;
            if (threadID < noWords)
            {
                frontierNext[threadID] = 0;
            }
            grid.sync();
        }
    }

    __global__ void BRSBFS128(  const unsigned* const __restrict__ noSliceSetsPtr,
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
        unsigned noWarps = (gridDim.x * blockDim.x) / WARP_SIZE;
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWords = *noWordsPtr;
        unsigned noSliceSets = *noSliceSetsPtr;

        unsigned threadIDInGroup = laneID % 4;
        unsigned groupID = laneID / 4;
        
        bool cont = true;
        while (cont)
        {
            unsigned myFrontierNextSize = 0;
            for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
            {
                MASK fragB = frontier[sliceSet * 4 + threadIDInGroup]; // we do not use the leftover 112 bytes that we fetched from L2, perhaps not big of a deal but try to optimize it later
                bool any = (fragB != 0);
                bool cont = warp.any(any);
                if (cont)
                {
                    unsigned start = sliceSetPtrs[sliceSet];
                    unsigned end = sliceSetPtrs[sliceSet + 1];
                    for (unsigned slicePtr = start; slicePtr < end; slicePtr += 8)
                    {
                        unsigned sliceIdx = slicePtr + groupID;
                        unsigned row = (sliceIdx < end) ? rowIds[sliceIdx] : 0;
                        unsigned maskIdx = sliceIdx * 4 + threadIDInGroup;
                        MASK fragA = (maskIdx < (end * 4)) ? masks[maskIdx] : 0;
                        unsigned fragC[2];
                        fragC[0] = 0;
                        m8n8k128(fragC, fragA, fragB);
                        for (unsigned offset = 2; offset > 0; offset >>= 1)
                        {
                            fragC[0] |= warp.shfl_down(fragC[0], offset);
                        }
                        if (threadIDInGroup == 0 && fragC[0])
                        {
                            unsigned word = row / MASK_BITS;
                            unsigned bit = row % MASK_BITS;
                            MASK temp = (static_cast<MASK>(1) << bit);
                            MASK old = atomicOr(&visited[word], temp); // this visited[word] and the below frontierNext[word] access is quite problematic in that rows each thread is assigned to in the warp can differ dramatically. Smth ordering may fix.
                            if ((old & temp) == 0)
                            {
                                ++myFrontierNextSize;
                                atomicOr(&frontierNext[word], temp);
                            }
                        }
                    }
                }
            }
            for (unsigned offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1)
            {
                myFrontierNextSize += warp.shfl_down(myFrontierNextSize, offset);
            }
            if (laneID == 0)
            {
                atomicAdd(frontierNextSizePtr, myFrontierNextSize);
            }
            grid.sync();
            if (laneID == 0)
            {
                myFrontierNextSize = *frontierNextSizePtr;
            }
            grid.sync();
            myFrontierNextSize = warp.shfl(myFrontierNextSize, 0);
            cont = (myFrontierNextSize != 0);
            if (threadID == 0)
            {
                *frontierNextSizePtr = 0;
            }
            MASK* temp = frontier;
            frontier = frontierNext;
            frontierNext = temp;
            if (threadID < noWords)
            {
                frontierNext[threadID] = 0;
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
    if (sliceSize == 32)
    {
        kernelPtr = (void*)BRSBFSKernels::BRSBFS32;
    }
    if (sliceSize == 128)
    {
        kernelPtr = (void*)BRSBFSKernels::BRSBFS128;
    }

    int gridSize, blockSize;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(
                                                &gridSize, 
                                                &blockSize, 
                                                kernelPtr,
                                                0,
                                                0));
    std::cout << "Grid Size: " << gridSize << std::endl;
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Total number of threads: " << gridSize * blockSize << std::endl;

    unsigned* d_NoSliceSets;
    unsigned* d_SliceSetPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    unsigned* d_NoWords;
    MASK* d_Frontier;
    MASK* d_Visited;
    unsigned* d_FrontierNextSize;
    MASK* d_FrontierNext;

    unsigned noMasks = sliceSize / MASK_BITS;
    // data structure
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * sliceSetPtrs[noSliceSets]))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * sliceSetPtrs[noSliceSets] * noMasks))

    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * sliceSetPtrs[noSliceSets], cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * sliceSetPtrs[noSliceSets] * noMasks, cudaMemcpyHostToDevice))

    // algorithm
    unsigned noWords = (n + MASK_BITS - 1) / MASK_BITS;
    gpuErrchk(cudaMalloc(&d_NoWords, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(MASK) * noWords))

    unsigned word = sourceVertex / (MASK_BITS);
    unsigned bit = sourceVertex % (MASK_BITS);
    MASK temp = (static_cast<MASK>(1) << bit);
    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(MASK) * noWords))

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
    gpuErrchk(cudaMemcpy(visited, d_Visited, sizeof(unsigned) * noWords, cudaMemcpyDeviceToHost))
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
