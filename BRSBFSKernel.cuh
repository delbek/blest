#ifndef BRSBFSKernel_CUH
#define BRSBFSKernel_CUH

#include "BFSKernel.cuh"
#include "BRS.cuh"
#include "cuda_runtime.h"

namespace BRSBFSKernels
{
    __global__ void BRSBFS( const unsigned* const __restrict__ sliceSize,
                            const unsigned* const __restrict__ noSliceSets,
                            const unsigned* const __restrict__ sliceSetPtrs,
                            const unsigned* const __restrict__ rowIds,
                            const MASK* const __restrict__ masks,
                            const MASK* const __restrict__ frontier,
                            MASK* const __restrict__ visited,
                            unsigned* const __restrict__ frontierNextSize,
                            MASK* const __restrict__ frontierNext)
    {

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

    virtual double hostCode(unsigned sourceVertex) override;
};

BRSBFSKernel::BRSBFSKernel(BitMatrix* matrix)
: BFSKernel(matrix)
{

}

double BRSBFSKernel::hostCode(unsigned sourceVertex)
{
    BRS* brs = dynamic_cast<BRS*>(matrix);
    unsigned N = brs->getN();
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
    unsigned noWords = std::ceil(double(N) / MASK_BITS);
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(MASK) * noWords))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(MASK) * noWords))

    unsigned wordIdx = sourceVertex / (MASK_BITS);
    unsigned bitIdx = sourceVertex % (MASK_BITS);
    MASK temp = 0;
    temp |= (static_cast<MASK>(1) << bitIdx);
    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(MASK) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNextSize, 0, sizeof(unsigned)))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(MASK) * noWords))

    gpuErrchk(cudaMemcpy(d_Frontier + wordIdx, &temp, sizeof(MASK), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Visited + wordIdx, &temp, sizeof(MASK), cudaMemcpyHostToDevice))

    unsigned frontierSize = 1;
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
    }
    double end = omp_get_wtime();

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

#endif
