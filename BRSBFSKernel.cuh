#ifndef BRSBFSKernel_CUH
#define BRSBFSKernel_CUH

#include "BFSKernel.cuh"
#include "BRS.cuh"

namespace BRSBFSKernels
{
    __global__ void BRSBFS( const unsigned* const __restrict__ sliceSizePtr,
                            const unsigned* const __restrict__ noSliceSetsPtr,
                            const unsigned* const __restrict__ sliceSetPtrs,
                            const unsigned* const __restrict__ rowIds,
                            const MASK* const __restrict__ masks,
                            const MASK* const __restrict__ frontier,
                            MASK* const __restrict__ visited,
                            unsigned* const __restrict__ frontierNextSizePtr,
                            MASK* const __restrict__ frontierNext)
    {
        unsigned sliceSize = *sliceSizePtr;
        unsigned noSliceSets = *noSliceSetsPtr;
        unsigned noWarps = (gridDim.x * blockDim.x) / WARP_SIZE;
        unsigned noSlices = K / sliceSize;
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned warpID = threadID / WARP_SIZE;
        unsigned laneID = threadID % WARP_SIZE;
        unsigned groupID = laneID / noSlices;
        unsigned threadIDInGroup = laneID % noSlices;
        auto warp = coalesced_threads();
        
        for (unsigned sliceSet = warpID; sliceSet < noSliceSets; sliceSet += noWarps)
        {
            MASK fragB = frontier[sliceSet * noSlices + threadIDInGroup]; // we do not use the leftover 112 bytes that we fetched from L2, perhaps not big of a deal but try to optimize it later
            bool any = (fragB != 0);
            bool cont = warp.any(any);
            if (cont)
            {
                unsigned start = sliceSetPtrs[sliceSet];
                unsigned end = sliceSetPtrs[sliceSet + 1];
                for (unsigned rowPtr = start; rowPtr < end; rowPtr += M)
                {
                    unsigned row = (rowPtr + groupID < end) ? rowIds[rowPtr * noSlices + laneID] : 0;
                    MASK mask = (rowPtr + groupID < end) ? masks[rowPtr * noSlices + laneID] : 0;
                    unsigned fragC[2];
                    for (unsigned round = 0; round < noSlices; ++round)
                    {
                        any = (threadIDInGroup == round && fragB != 0) ? true : false;
                        cont = warp.any(any);
                        if (cont)
                        {
                            fragC[0] = 0;
                            MASK fragA = threadIDInGroup == round ? mask : 0;
                            m8n8k128(fragC, fragA, fragB);
                            if (fragC[0] && threadIDInGroup == round)
                            {
                                unsigned word = row / MASK_BITS;
                                unsigned bit = row % MASK_BITS;
                                MASK temp = (static_cast<MASK>(1) << bit);
                                MASK old = atomicOr(&visited[word], temp); // this visited[word] and the below frontierNext[word] access is quite problematic in that rows each thread is assigned to in the warp can differ dramatically. Smth ordering may fix.
                                if ((old & temp) == 0)
                                {
                                    atomicAdd(frontierNextSizePtr, 1);
                                    atomicOr(&frontierNext[word], temp);
                                }
                            }
                        }
                    }
                }
            }
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

    virtual double hostCode(const std::string& sourceVertexFilename) final;

private:
    void initializeSourceVertices(const std::string& sourceVertexFilename, MASK* sourceVertices);
};

BRSBFSKernel::BRSBFSKernel(BitMatrix* matrix)
: BFSKernel(matrix)
{

}

double BRSBFSKernel::hostCode(const std::string& sourceVertexFilename)
{
    BRS* brs = dynamic_cast<BRS*>(matrix);
    unsigned n = brs->getN();
    unsigned sliceSize = brs->getSliceSize();
    unsigned noSlices = K / sliceSize;
    unsigned noSliceSets = brs->getNoSliceSets();
    unsigned* sliceSetPtrs = brs->getSliceSetPtrs();
    unsigned* rowIds = brs->getRowIds();
    MASK* masks = brs->getMasks();
    unsigned totalSlots = sliceSetPtrs[noSliceSets] * noSlices;

    int gridSize, blockSize;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(
                                                &gridSize, 
                                                &blockSize, 
                                                BRSBFSKernels::BRSBFS,
                                                0,
                                                0));

    unsigned* d_SliceSize;
    unsigned* d_NoSliceSets;
    unsigned* d_SliceSetPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    MASK* d_Frontier;
    MASK* d_Visited;
    unsigned* d_FrontierNextSize;
    MASK* d_FrontierNext;

    // data structure
    gpuErrchk(cudaMalloc(&d_SliceSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * totalSlots))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * totalSlots))

    gpuErrchk(cudaMemcpy(d_SliceSize, &sliceSize, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(unsigned) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * totalSlots, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * totalSlots, cudaMemcpyHostToDevice))

    // algorithm
    unsigned noWords = (n + MASK_BITS - 1) / MASK_BITS;
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(MASK) * noWords))

    gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(MASK) * noWords))

    MASK* sourceVertices = new MASK[noWords];
    std::fill(sourceVertices, sourceVertices + noWords, 0);
    this->initializeSourceVertices(sourceVertexFilename, sourceVertices);
    gpuErrchk(cudaMemcpy(d_Frontier, sourceVertices, sizeof(MASK) * noWords, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Visited, sourceVertices, sizeof(MASK) * noWords, cudaMemcpyHostToDevice))
    delete[] sourceVertices;

    unsigned frontierSize = 1;
    unsigned totalVisited = frontierSize;
    double start = omp_get_wtime();
    while (frontierSize != 0)
    {
        BRSBFSKernels::BRSBFS<<<gridSize, blockSize>>>( d_SliceSize,
                                                        d_NoSliceSets,
                                                        d_SliceSetPtrs,
                                                        d_RowIds,
                                                        d_Masks,
                                                        d_Frontier,
                                                        d_Visited,
                                                        d_FrontierNextSize,
                                                        d_FrontierNext);
        gpuErrchk(cudaDeviceSynchronize())

        gpuErrchk(cudaMemcpy(&frontierSize, d_FrontierNextSize, sizeof(unsigned), cudaMemcpyDeviceToHost))
        std::swap(d_Frontier, d_FrontierNext);
        gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
        gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(MASK) * noWords))
        totalVisited += frontierSize;
    }
    double end = omp_get_wtime();
    std::cout << "Total Frontier: " << totalVisited << std::endl;

    gpuErrchk(cudaFree(d_SliceSize))
    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_FrontierNext))

    return (end - start);
}

void BRSBFSKernel::initializeSourceVertices(const std::string& sourceVertexFilename, MASK* sourceVertices)
{
    std::ifstream file(sourceVertexFilename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file from which to read source vertices.");
    }

    unsigned vertex;
    while (file >> vertex)
    {
        unsigned word = vertex / MASK_BITS;
        unsigned bit = vertex % MASK_BITS;
        MASK temp = (static_cast<MASK>(1) << bit);
        sourceVertices[word] |= temp;
    }
}

#endif
