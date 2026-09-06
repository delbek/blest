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

#include "BVSS.cuh"
#include "BVSSCCKernels.cuh"
#include <array>

struct CCResult
{
    double time;
    unsigned* components;
};

class CCKernel
{
public:
    CCKernel(BVSS* matrix);
    CCKernel(const CCKernel& other) = delete;
    CCKernel(CCKernel&& other) noexcept = delete;
    CCKernel& operator=(const CCKernel& other) = delete;
    CCKernel& operator=(CCKernel&& other) noexcept = delete;
    ~CCKernel() = default;

    CCResult run();

private:
    BVSS* m_matrix;
};

CCKernel::CCKernel(BVSS* matrix)
: m_matrix(matrix)
{

}

CCResult CCKernel::run()
{
    gpuErrchk(cudaSetDevice(0))

    BVSS* bvss = m_matrix;
    unsigned n = bvss->getN();
    unsigned sliceSize = bvss->getSliceSize();
    unsigned noMasks = bvss->getNoMasks();
    unsigned noRealSliceSets = bvss->getNoRealSliceSets();
    SLICE_TYPE noSlices = bvss->getNoSlices();
    unsigned noSliceSets = bvss->getNoVirtualSliceSets();
    SLICE_TYPE* sliceSetPtrs = bvss->getSliceSetPtrs();
    unsigned* virtualToReal = bvss->getVirtualToReal();
    unsigned* realPtrs = bvss->getRealPtrs();
    unsigned* rowIds = bvss->getRowIds();
    MASK* masks = bvss->getMasks();

    CCResult result;
    result.components = new unsigned[n];
    for (unsigned i = 0; i < n; ++i)
    {
        result.components[i] = i;
    }

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    void* kernelPtr;
    if (sliceSize == 8)
    {
        if (FULL_PADDING)
        {
            kernelPtr = (void*)BVSSCCKernels::BVSSCC8EnhancedSliceSize8NoMasks4LazyFullPad;
        }
        else
        {
            kernelPtr = (void*)BVSSCCKernels::BVSSCC8EnhancedSliceSize8NoMasks4Lazy;
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
    SLICE_TYPE* d_SliceSetPtrs;
    unsigned* d_VirtualToReal;
    unsigned* d_RealPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    // data structure
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_VirtualToReal, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_RealPtrs, sizeof(unsigned) * (noRealSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VirtualToReal, virtualToReal, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RealPtrs, realPtrs, sizeof(unsigned) * (noRealSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    unsigned* d_NoWords;
    unsigned* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_FrontierCurrentSize;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_FrontierNextSize;
    unsigned* d_Marker;
    unsigned* d_Components;

    unsigned noWords = (n + 31) / 32;
    gpuErrchk(cudaMalloc(&d_NoWords, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Marker, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_Components, sizeof(unsigned) * n))

    gpuErrchk(cudaMemset(d_Frontier, 0xFF, sizeof(unsigned) * noWords))
    if (n & 31)
    {
        unsigned last = (1u << (n & 31)) - 1;
        cudaMemcpy(d_Frontier + (noWords - 1), &last, sizeof(unsigned), cudaMemcpyHostToDevice);
    }
    gpuErrchk(cudaMemset(d_Marker, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemcpy(d_NoWords, &noWords, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Components, result.components, sizeof(unsigned) * n, cudaMemcpyHostToDevice))

    std::vector<unsigned> initialVset;
    for (unsigned vset = 0; vset < noSliceSets; ++vset)
    {
        initialVset.emplace_back(vset);
    }
    unsigned initialFrontierSize = initialVset.size();
    gpuErrchk(cudaMemcpy(d_SparseFrontierIds, initialVset.data(), sizeof(unsigned) * initialFrontierSize, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &initialFrontierSize, sizeof(unsigned), cudaMemcpyHostToDevice))

    double start;
    std::array<void*, 13> argsB =
    {
        (void*)&d_SliceSetPtrs,
        (void*)&d_VirtualToReal,
        (void*)&d_RealPtrs,
        (void*)&d_RowIds,
        (void*)&d_Masks,
        (void*)&d_NoWords,
        (void*)&d_Components,
        (void*)&d_Frontier,
        (void*)&d_Marker,
        (void*)&d_SparseFrontierIds,
        (void*)&d_FrontierCurrentSize,
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
    gpuErrchk(cudaPeekAtLastError())
    gpuErrchk(cudaDeviceSynchronize())
    double end = omp_get_wtime();
    result.time = (end - start);

    gpuErrchk(cudaMemcpy(result.components, d_Components, sizeof(unsigned) * n, cudaMemcpyDeviceToHost))

    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_VirtualToReal))
    gpuErrchk(cudaFree(d_RealPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_NoWords))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_SparseFrontierIds))
    gpuErrchk(cudaFree(d_FrontierCurrentSize))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Marker))
    gpuErrchk(cudaFree(d_Components))

    return result;
}
