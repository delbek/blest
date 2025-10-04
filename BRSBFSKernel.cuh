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

    __global__ void BRSBFS8Enhanced(    const unsigned* const __restrict__ noSliceSetsPtr,
                                        const unsigned* const __restrict__ sliceSetPtrs,
                                        const unsigned* const __restrict__ sliceSetIds,
                                        const unsigned* const __restrict__ sliceSetOffsets,
                                        const unsigned* const __restrict__ rowIds,
                                        const MASK* const __restrict__ masks,
                                        const unsigned* const __restrict__ noWordsPtr,
                                        const unsigned* const __restrict__ directionThresholdPtr,
                                        // current
                                        MASK* __restrict__ frontier,
                                        unsigned* __restrict__ sparseFrontierIds,
                                        unsigned* __restrict__ frontierCurrentSizePtr,
                                        // next
                                        MASK* __restrict__ frontierNext,
                                        unsigned* __restrict__ sparseFrontierNextIds,
                                        unsigned* __restrict__ frontierNextSizePtr,
                                        //
                                        MASK* const __restrict__ visited)
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned noThreads = gridDim.x * blockDim.x;
        unsigned noWarps = noThreads / WARP_SIZE;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWords = *noWordsPtr;
        unsigned DIRECTION_THRESHOLD = *directionThresholdPtr;
        unsigned noSliceSets = *noSliceSetsPtr;

        bool cont = true;
        while (cont)
        {
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            if (currentFrontierSize < DIRECTION_THRESHOLD)
            {
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned realSliceSetId = sparseFrontierIds[i];
                    unsigned shift = (realSliceSetId % 4) << 3;
                    MASK origFragB = ((frontier[realSliceSetId >> 2] >> shift) & 0x000000FF);
                    for (unsigned sliceSet = sliceSetOffsets[realSliceSetId]; sliceSet < sliceSetOffsets[realSliceSetId + 1]; ++sliceSet)
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
                            if (laneID % 9 == 0 || laneID % 9 == 4)
                            {
                                fragB = (laneID % 9 == 0) ? (origFragB) : (origFragB << 8);
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
                                    unsigned sliceIdx = (bit >> 3);
                                    MASK sliceMask = (static_cast<MASK>(0xFF) << (sliceIdx << 3));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.x >> 3;
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
                                    unsigned sliceIdx = (bit >> 3);
                                    MASK sliceMask = (static_cast<MASK>(0xFF) << (sliceIdx << 3));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.y >> 3;
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
                                unsigned word = rows.z / MASK_BITS;
                                unsigned bit = rows.z % MASK_BITS;
                                MASK temp = (static_cast<MASK>(1) << bit);
                                MASK old = atomicOr(&visited[word], temp);
                                if ((old & temp) == 0)
                                {
                                    old = atomicOr(&frontierNext[word], temp);
                                    unsigned sliceIdx = (bit >> 3);
                                    MASK sliceMask = (static_cast<MASK>(0xFF) << (sliceIdx << 3));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.z >> 3;
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
                                    unsigned sliceIdx = (bit >> 3);
                                    MASK sliceMask = (static_cast<MASK>(0xFF) << (sliceIdx << 3));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.w >> 3;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
                {
                    unsigned realSliceSetId = sliceSetIds[sliceSet];
                    unsigned shift = (realSliceSetId % 4) << 3;
                    MASK origFragB = ((frontier[realSliceSetId >> 2] >> shift) & 0x000000FF);
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
                            if (laneID % 9 == 0 || laneID % 9 == 4)
                            {
                                fragB = (laneID % 9 == 0) ? (origFragB) : (origFragB << 8);
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
                                    unsigned sliceIdx = (bit >> 3);
                                    MASK sliceMask = (static_cast<MASK>(0xFF) << (sliceIdx << 3));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.x >> 3;
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
                                    unsigned sliceIdx = (bit >> 3);
                                    MASK sliceMask = (static_cast<MASK>(0xFF) << (sliceIdx << 3));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.y >> 3;
                                    }
                                }
                            }
    
                            fragA = (mask & 0xFFFF0000);
                            fragB = 0;
                            if (laneID % 9 == 0 || laneID % 9 == 4)
                            {
                                fragB = (laneID % 9 == 0) ? (origFragB) : (origFragB << 8);
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
                                    unsigned sliceIdx = (bit >> 3);
                                    MASK sliceMask = (static_cast<MASK>(0xFF) << (sliceIdx << 3));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.z >> 3;
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
                                    unsigned sliceIdx = (bit >> 3);
                                    MASK sliceMask = (static_cast<MASK>(0xFF) << (sliceIdx << 3));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.w >> 3;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<MASK>(frontier, frontierNext);
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

    __global__ void BRSBFS16Enhanced(   const unsigned* const __restrict__ noSliceSetsPtr,
                                        const unsigned* const __restrict__ sliceSetPtrs,
                                        const unsigned* const __restrict__ sliceSetIds,
                                        const unsigned* const __restrict__ sliceSetOffsets,
                                        const unsigned* const __restrict__ rowIds,
                                        const MASK* const __restrict__ masks,
                                        const unsigned* const __restrict__ noWordsPtr,
                                        const unsigned* const __restrict__ directionThresholdPtr,
                                        // current
                                        MASK* __restrict__ frontier,
                                        unsigned* __restrict__ sparseFrontierIds,
                                        unsigned* __restrict__ frontierCurrentSizePtr,
                                        // next
                                        MASK* __restrict__ frontierNext,
                                        unsigned* __restrict__ sparseFrontierNextIds,
                                        unsigned* __restrict__ frontierNextSizePtr,
                                        //
                                        MASK* const __restrict__ visited)
    {
        auto warp = coalesced_threads();
        auto grid = this_grid();
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned noThreads = gridDim.x * blockDim.x;
        unsigned noWarps = noThreads / WARP_SIZE;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;

        unsigned noWords = *noWordsPtr;
        unsigned DIRECTION_THRESHOLD = *directionThresholdPtr;
        unsigned noSliceSets = *noSliceSetsPtr;

        bool cont = true;
        while (cont)
        {
            unsigned currentFrontierSize = *frontierCurrentSizePtr;
            if (currentFrontierSize < DIRECTION_THRESHOLD)
            {
                for (unsigned i = warpID; i < currentFrontierSize; i += noWarps)
                {
                    unsigned realSliceSetId = sparseFrontierIds[i];
                    unsigned shift = (realSliceSetId % 2) << 4;
                    MASK origFragB = ((frontier[realSliceSetId >> 1] >> shift) & 0x0000FFFF);
                    for (unsigned sliceSet = sliceSetOffsets[realSliceSetId]; sliceSet < sliceSetOffsets[realSliceSetId + 1]; ++sliceSet)
                    {
                        unsigned tileStart = sliceSetPtrs[sliceSet] / 2;
                        unsigned tileEnd = sliceSetPtrs[sliceSet + 1] / 2;
                        for (unsigned t = tileStart; t < tileEnd; t += WARP_SIZE)
                        {
                            unsigned tile = t + laneID;
                            uint2 rows = {0, 0};
                            MASK fragA = 0;
                            if (tile < tileEnd)
                            {
                                rows = reinterpret_cast<const uint2*>(rowIds)[tile];
                                fragA = masks[tile];
                            }

                            MASK fragB = 0;
                            if (laneID % 9 == 0 || laneID % 9 == 4)
                            {
                                fragB = (laneID % 9 == 0) ? (origFragB) : (origFragB << 16);
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
                                    unsigned sliceIdx = (bit >> 4);
                                    MASK sliceMask = (static_cast<MASK>(0xFFFF) << (sliceIdx << 4));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.x >> 4;
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
                                    unsigned sliceIdx = (bit >> 4);
                                    MASK sliceMask = (static_cast<MASK>(0xFFFF) << (sliceIdx << 4));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.y >> 4;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
                {
                    unsigned realSliceSetId = sliceSetIds[sliceSet];
                    unsigned shift = (realSliceSetId % 2) << 4;
                    MASK origFragB = ((frontier[realSliceSetId >> 1] >> shift) & 0x0000FFFF);
                    if (origFragB)
                    {
                        unsigned tileStart = sliceSetPtrs[sliceSet] / 2;
                        unsigned tileEnd = sliceSetPtrs[sliceSet + 1] / 2;
                        for (unsigned t = tileStart; t < tileEnd; t += WARP_SIZE)
                        {
                            unsigned tile = t + laneID;
                            uint2 rows = {0, 0};
                            MASK fragA = 0;
                            if (tile < tileEnd)
                            {
                                rows = reinterpret_cast<const uint2*>(rowIds)[tile];
                                fragA = masks[tile];
                            }
    
                            MASK fragB = 0;
                            if (laneID % 9 == 0 || laneID % 9 == 4)
                            {
                                fragB = (laneID % 9 == 0) ? (origFragB) : (origFragB << 16);
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
                                    unsigned sliceIdx = (bit >> 4);
                                    MASK sliceMask = (static_cast<MASK>(0xFFFF) << (sliceIdx << 4));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.x >> 4;
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
                                    unsigned sliceIdx = (bit >> 4);
                                    MASK sliceMask = (static_cast<MASK>(0xFFFF) << (sliceIdx << 4));
                                    if ((old & sliceMask) == 0)
                                    {
                                        unsigned loc = atomicAdd(frontierNextSizePtr, 1);
                                        sparseFrontierNextIds[loc] = rows.y >> 4;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            grid.sync();
            cont = (*frontierNextSizePtr != 0);
            swap<MASK>(frontier, frontierNext);
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
    unsigned noRealSliceSets = brs->getNoRealSliceSets();
    unsigned noSliceSets = brs->getNoVirtualSliceSets();
    unsigned* sliceSetPtrs = brs->getSliceSetPtrs();
    unsigned* sliceSetIds = brs->getSliceSetIds();
    unsigned* sliceSetOffsets = brs->getSliceSetOffsets();
    unsigned* rowIds = brs->getRowIds();
    MASK* masks = brs->getMasks();
    const unsigned DIRECTION_THRESHOLD = noSliceSets / 4;  

    void* kernelPtr = nullptr;
    if (sliceSize == 8)
    {
        kernelPtr = (void*)BRSBFSKernels::BRSBFS8Enhanced;
    }
    else if (sliceSize == 16)
    {
        kernelPtr = (void*)BRSBFSKernels::BRSBFS16Enhanced;
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
    unsigned* d_SliceSetIds;
    unsigned* d_SliceSetOffsets;
    unsigned* d_RowIds;
    MASK* d_Masks;

    unsigned* d_NoWords;
    unsigned* d_DIRECTION_THRESHOLD;
    MASK* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_FrontierCurrentSize;
    MASK* d_FrontierNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_FrontierNextSize;
    MASK* d_Visited;

    unsigned noMasks = MASK_BITS / sliceSize;
    // data structure
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_SliceSetIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_SliceSetOffsets, sizeof(unsigned) * noRealSliceSets))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * sliceSetPtrs[noSliceSets]))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (sliceSetPtrs[noSliceSets] / noMasks)))

    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetIds, sliceSetIds, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetOffsets, sliceSetOffsets, sizeof(unsigned) * noRealSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * sliceSetPtrs[noSliceSets], cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (sliceSetPtrs[noSliceSets] / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    unsigned noWords = (n + MASK_BITS - 1) / MASK_BITS;
    gpuErrchk(cudaMalloc(&d_NoWords, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_DIRECTION_THRESHOLD, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(MASK) * noWords))

    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(MASK) * noWords))

    gpuErrchk(cudaMemcpy(d_NoWords, &noWords, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_DIRECTION_THRESHOLD, &DIRECTION_THRESHOLD, sizeof(unsigned), cudaMemcpyHostToDevice))
    unsigned initialSliceSet = sourceVertex / sliceSize;
    unsigned frontierCurrentSize = 1;
    gpuErrchk(cudaMemcpy(d_SparseFrontierIds, &initialSliceSet, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &frontierCurrentSize, sizeof(unsigned), cudaMemcpyHostToDevice))
    unsigned word = sourceVertex / (MASK_BITS);
    unsigned bit = sourceVertex % (MASK_BITS);
    MASK temp = (static_cast<MASK>(1) << bit);
    gpuErrchk(cudaMemcpy(d_Frontier + word, &temp, sizeof(MASK), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Visited + word, &temp, sizeof(MASK), cudaMemcpyHostToDevice))

    double start = omp_get_wtime();
    void* kernelArgs[] = {
                            (void*)&d_NoSliceSets,
                            (void*)&d_SliceSetPtrs,
                            (void*)&d_SliceSetIds,
                            (void*)&d_SliceSetOffsets,
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
                            (void*)&d_Visited};
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
    //std::cout << "Total traversed number of nodes: " << totalVisited << std::endl;

    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_SliceSetIds))
    gpuErrchk(cudaFree(d_SliceSetOffsets))
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

    return (end - start);
}

#endif
