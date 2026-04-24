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

#include "BVSSBFSKernels.cuh"
#include "BVSS.cuh"
#include <array>

struct BFSResult
{
    double time;
    unsigned sourceVertex;
    unsigned* levels;
};

class BFSKernel
{
public:
    BFSKernel(BitMatrix* matrix);
    BFSKernel(const BFSKernel& other) = delete;
    BFSKernel(BFSKernel&& other) noexcept = delete;
    BFSKernel& operator=(const BFSKernel& other) = delete;
    BFSKernel& operator=(BFSKernel&& other) noexcept = delete;
    ~BFSKernel() = default;

    std::vector<BFSResult> multiSourceRun(const std::vector<unsigned>& sources);
    BFSResult singleSourceRun(unsigned sourceVertex, bool switching);

private:
    BitMatrix* m_matrix;
};

BFSKernel::BFSKernel(BitMatrix* matrix)
: m_matrix(matrix)
{

}

std::vector<BFSResult> BFSKernel::multiSourceRun(const std::vector<unsigned>& sources)
{
    if (sources.empty())
    {
        return {};
    }

    gpuErrchk(cudaSetDevice(0))

    BVSS* bvss = dynamic_cast<BVSS*>(m_matrix);
    bool lazyKernel = (bvss->getUpdateDivergence() > LAZY_KERNEL_THRESHOLD);
    
    // tune
    bool switching = false;
    if (lazyKernel)
    {
        BFSResult noSwitch = this->singleSourceRun(sources[0], false);
        BFSResult withSwitch = this->singleSourceRun(sources[0], true);
        if (withSwitch.time < noSwitch.time)
        {
            switching = true;
            std::cout << "Switching-based kernel is enabled." << std::endl;
        }
        delete[] noSwitch.levels;
        delete[] withSwitch.levels;
    }
    //
    
    std::vector<BFSResult> results;

    CSC* csc = bvss->getCSR();
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

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    void* kernelPtr;
    if (sliceSize == 8)
    {
        if (lazyKernel)
        {
            if (switching)
            {
                if (FULL_PADDING)
                {
                    kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4LazyFullPadSwitching;
                }
                else
                {
                    kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4LazySwitching;
                }
            }
            else
            {
                if (FULL_PADDING)
                {
                    kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4LazyFullPad;
                }
                else
                {
                    kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4Lazy;
                }
            }
        }
        else
        {
            if (FULL_PADDING)
            {
                kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4FullPad;
            }
            else
            {
                kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4;
            }
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
    std::cout << "Total number of threads: " << gridSize * blockSize << std::endl;

    unsigned* d_RowPtrs;
    unsigned* d_ColIds;
    unsigned* d_N;
    SLICE_TYPE* d_SliceSetPtrs;
    unsigned* d_VirtualToReal;
    unsigned* d_RealPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    unsigned* d_NoWords;
    unsigned* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_UnvisitedCurrentSize;
    unsigned* d_FrontierCurrentSize;
    unsigned* d_VisitedNext;
    unsigned* d_FrontierNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_UnvisitedNextSize;
    unsigned* d_FrontierNextSize;
    unsigned* d_Visited;
    unsigned* d_Levels;

    // data structure
    unsigned noWords = (n + 31) / 32;
    gpuErrchk(cudaMalloc(&d_N, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_NoWords, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_RowPtrs, sizeof(unsigned) * (csc->getN() + 1)))
    gpuErrchk(cudaMalloc(&d_ColIds, sizeof(unsigned) * csc->getNNZ()))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_VirtualToReal, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_RealPtrs, sizeof(unsigned) * (noRealSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_N, &csc->getN(), sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoWords, &noWords, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowPtrs, csc->getColPtrs(), sizeof(unsigned) * (csc->getN() + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_ColIds, csc->getRows(), sizeof(unsigned) * csc->getNNZ(), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VirtualToReal, virtualToReal, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RealPtrs, realPtrs, sizeof(unsigned) * (noRealSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_UnvisitedCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_VisitedNext, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_UnvisitedNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_Levels, sizeof(unsigned) * n))

    for (const auto& sourceVertex: sources)
    {
        BFSResult result;
        result.sourceVertex = sourceVertex;
        result.levels = new unsigned[n];
        std::fill(result.levels, result.levels + n, UNSIGNED_MAX);
        result.levels[sourceVertex] = 0;

        gpuErrchk(cudaMemcpy(d_Levels, result.levels, sizeof(unsigned) * n, cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(unsigned) * noWords))
        gpuErrchk(cudaMemset(d_VisitedNext, 0, sizeof(unsigned) * noWords))
        gpuErrchk(cudaMemset(d_Visited, 0, sizeof(unsigned) * noWords))
        gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(unsigned) * noWords))

        std::vector<unsigned> initialVset;
        for (unsigned vset = realPtrs[sourceVertex / sliceSize]; vset < realPtrs[sourceVertex / sliceSize + 1]; ++vset)
        {
            initialVset.emplace_back(vset);
        }
        unsigned initialFrontierSize = initialVset.size();
        gpuErrchk(cudaMemcpy(d_SparseFrontierIds, initialVset.data(), sizeof(unsigned) * initialFrontierSize, cudaMemcpyHostToDevice))
        unsigned unvisitedSize = UNSIGNED_MAX;
        gpuErrchk(cudaMemcpy(d_UnvisitedCurrentSize, &unvisitedSize, sizeof(unsigned), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &initialFrontierSize, sizeof(unsigned), cudaMemcpyHostToDevice))

        unsigned word = sourceVertex >> 5;
        unsigned bit = sourceVertex & 31;
        unsigned temp = (1u << bit);
        gpuErrchk(cudaMemcpy(d_Frontier + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_Visited + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_VisitedNext + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))

        /*
        // profiling
        unsigned long long* levelTime = new unsigned long long[n];
        std::fill(levelTime, levelTime + n, UNSIGNED_LONG_MAX);
        unsigned long long* d_LevelTime;
        gpuErrchk(cudaMalloc(&d_LevelTime, sizeof(unsigned long long) * n))
        gpuErrchk(cudaMemcpy(d_LevelTime, levelTime, sizeof(unsigned long long) * n, cudaMemcpyHostToDevice))
        //
        */

        double start;
        if (!lazyKernel)
        {
            std::array<void*, 13> args =
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
                args.data(),
                allocateSharedMemory(blockSize),
                0))
        }
        else
        {
            if (switching)
            {
                std::array<void*, 19> args =
                {
                    (void*)&d_RowPtrs,
                    (void*)&d_ColIds,
                    (void*)&d_N,
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
                    (void*)&d_UnvisitedCurrentSize,
                    (void*)&d_FrontierCurrentSize,
                    (void*)&d_VisitedNext,
                    (void*)&d_SparseFrontierNextIds,
                    (void*)&d_UnvisitedNextSize,
                    (void*)&d_FrontierNextSize
                    /*
                    // profiling
                    (void*)&d_LevelTime
                    //
                    */
                };

                start = omp_get_wtime();
                gpuErrchk(cudaLaunchCooperativeKernel(
                    kernelPtr,
                    gridSize,
                    blockSize,
                    args.data(),
                    allocateSharedMemory(blockSize),
                    0))
            }
            else
            {
                std::array<void*, 14> args =
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
                    args.data(),
                    allocateSharedMemory(blockSize),
                    0))
            }
        }
        gpuErrchk(cudaPeekAtLastError())
        gpuErrchk(cudaDeviceSynchronize())
        double end = omp_get_wtime();

        result.time = (end - start);

        gpuErrchk(cudaMemcpy(result.levels, d_Levels, sizeof(unsigned) * n, cudaMemcpyDeviceToHost))
        results.emplace_back(result);

        /*
        // profiling
        gpuErrchk(cudaMemcpy(levelTime, d_LevelTime, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost))
        for (unsigned i = 1; i < n; ++i)
        {
            if (levelTime[i] == UNSIGNED_LONG_MAX) break;
            std::cout << "Level " << i - 1 << " took: " << levelTime[i] - levelTime[i - 1] << std::endl;
        }
        delete[] levelTime;
        gpuErrchk(cudaFree(d_LevelTime))
        //
        */
    }

    gpuErrchk(cudaFree(d_RowPtrs))
    gpuErrchk(cudaFree(d_ColIds))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_VirtualToReal))
    gpuErrchk(cudaFree(d_RealPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_N))
    gpuErrchk(cudaFree(d_NoWords))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_SparseFrontierIds))
    gpuErrchk(cudaFree(d_UnvisitedCurrentSize))
    gpuErrchk(cudaFree(d_FrontierCurrentSize))
    gpuErrchk(cudaFree(d_VisitedNext))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_UnvisitedNextSize))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_Levels))
    gpuErrchk(cudaFree(d_FrontierNext))

    return results;
}

BFSResult BFSKernel::singleSourceRun(unsigned sourceVertex, bool switching)
{
    gpuErrchk(cudaSetDevice(0))

    BVSS* bvss = dynamic_cast<BVSS*>(m_matrix);
    CSC* csc = bvss->getCSR();
    unsigned n = bvss->getN();
    unsigned sliceSize = bvss->getSliceSize();
    unsigned noMasks = bvss->getNoMasks();
    unsigned noRealSliceSets = bvss->getNoRealSliceSets();
    SLICE_TYPE noSlices = bvss->getNoSlices();
    bool lazyKernel = (bvss->getUpdateDivergence() > LAZY_KERNEL_THRESHOLD);
    unsigned noSliceSets = bvss->getNoVirtualSliceSets();
    SLICE_TYPE* sliceSetPtrs = bvss->getSliceSetPtrs();
    unsigned* virtualToReal = bvss->getVirtualToReal();
    unsigned* realPtrs = bvss->getRealPtrs();
    unsigned* rowIds = bvss->getRowIds();
    MASK* masks = bvss->getMasks();

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
        if (lazyKernel)
        {
            if (switching)
            {
                if (FULL_PADDING)
                {
                    kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4LazyFullPadSwitching;
                }
                else
                {
                    kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4LazySwitching;
                }
            }
            else
            {
                if (FULL_PADDING)
                {
                    kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4LazyFullPad;
                }
                else
                {
                    kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4Lazy;
                }
            }
        }
        else
        {
            if (FULL_PADDING)
            {
                kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4FullPad;
            }
            else
            {
                kernelPtr = (void*)BVSSBFSKernels::BVSSBFS8EnhancedSliceSize8NoMasks4;
            }
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
    std::cout << "Total number of threads: " << gridSize * blockSize << std::endl;

    unsigned* d_RowPtrs;
    unsigned* d_ColIds;
    unsigned* d_N;
    SLICE_TYPE* d_SliceSetPtrs;
    unsigned* d_VirtualToReal;
    unsigned* d_RealPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    unsigned* d_NoWords;
    unsigned* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_UnvisitedCurrentSize;
    unsigned* d_FrontierCurrentSize;
    unsigned* d_VisitedNext;
    unsigned* d_FrontierNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_UnvisitedNextSize;
    unsigned* d_FrontierNextSize;
    unsigned* d_Visited;
    unsigned* d_Levels;

    // data structure
    unsigned noWords = (n + 31) / 32;
    gpuErrchk(cudaMalloc(&d_N, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_NoWords, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_RowPtrs, sizeof(unsigned) * (csc->getN() + 1)))
    gpuErrchk(cudaMalloc(&d_ColIds, sizeof(unsigned) * csc->getNNZ()))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_VirtualToReal, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_RealPtrs, sizeof(unsigned) * (noRealSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_N, &csc->getN(), sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoWords, &noWords, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowPtrs, csc->getColPtrs(), sizeof(unsigned) * (csc->getN() + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_ColIds, csc->getRows(), sizeof(unsigned) * csc->getNNZ(), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VirtualToReal, virtualToReal, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RealPtrs, realPtrs, sizeof(unsigned) * (noRealSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_UnvisitedCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_VisitedNext, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_UnvisitedNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_FrontierNext, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMalloc(&d_Levels, sizeof(unsigned) * n))

    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_VisitedNext, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemset(d_FrontierNext, 0, sizeof(unsigned) * noWords))
    gpuErrchk(cudaMemcpy(d_Levels, result.levels, sizeof(unsigned) * n, cudaMemcpyHostToDevice))

    std::vector<unsigned> initialVset;
    for (unsigned vset = realPtrs[sourceVertex / sliceSize]; vset < realPtrs[sourceVertex / sliceSize + 1]; ++vset)
    {
        initialVset.emplace_back(vset);
    }
    unsigned initialFrontierSize = initialVset.size();
    gpuErrchk(cudaMemcpy(d_SparseFrontierIds, initialVset.data(), sizeof(unsigned) * initialFrontierSize, cudaMemcpyHostToDevice))
    unsigned unvisitedSize = UNSIGNED_MAX;
    gpuErrchk(cudaMemcpy(d_UnvisitedCurrentSize, &unvisitedSize, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &initialFrontierSize, sizeof(unsigned), cudaMemcpyHostToDevice))

    unsigned word = sourceVertex >> 5;
    unsigned bit = sourceVertex & 31;
    unsigned temp = (1u << bit);
    gpuErrchk(cudaMemcpy(d_Frontier + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Visited + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VisitedNext + word, &temp, sizeof(unsigned), cudaMemcpyHostToDevice))

    /*
    // profiling
    unsigned long long* levelTime = new unsigned long long[n];
    std::fill(levelTime, levelTime + n, UNSIGNED_LONG_MAX);
    unsigned long long* d_LevelTime;
    gpuErrchk(cudaMalloc(&d_LevelTime, sizeof(unsigned long long) * n))
    gpuErrchk(cudaMemcpy(d_LevelTime, levelTime, sizeof(unsigned long long) * n, cudaMemcpyHostToDevice))
    //
    */

    double start;
    if (!lazyKernel)
    {
        std::array<void*, 13> args =
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
            args.data(),
            allocateSharedMemory(blockSize),
            0))
    }
    else
    {
        if (switching)
        {
            std::array<void*, 19> args =
            {
                (void*)&d_RowPtrs,
                (void*)&d_ColIds,
                (void*)&d_N,
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
                (void*)&d_UnvisitedCurrentSize,
                (void*)&d_FrontierCurrentSize,
                (void*)&d_VisitedNext,
                (void*)&d_SparseFrontierNextIds,
                (void*)&d_UnvisitedNextSize,
                (void*)&d_FrontierNextSize
                /*
                // profiling
                (void*)&d_LevelTime
                //
                */
            };

            start = omp_get_wtime();
            gpuErrchk(cudaLaunchCooperativeKernel(
                kernelPtr,
                gridSize,
                blockSize,
                args.data(),
                allocateSharedMemory(blockSize),
                0))
        }
        else
        {
            std::array<void*, 14> args =
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
                args.data(),
                allocateSharedMemory(blockSize),
                0))
        }
    }
    gpuErrchk(cudaPeekAtLastError())
    gpuErrchk(cudaDeviceSynchronize())
    double end = omp_get_wtime();

    result.time = (end - start);

    gpuErrchk(cudaFree(d_RowPtrs))
    gpuErrchk(cudaFree(d_ColIds))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_VirtualToReal))
    gpuErrchk(cudaFree(d_RealPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_N))
    gpuErrchk(cudaFree(d_NoWords))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_SparseFrontierIds))
    gpuErrchk(cudaFree(d_UnvisitedCurrentSize))
    gpuErrchk(cudaFree(d_FrontierCurrentSize))
    gpuErrchk(cudaFree(d_VisitedNext))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_UnvisitedNextSize))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_Levels))
    gpuErrchk(cudaFree(d_FrontierNext))

    /*
    // profiling
    gpuErrchk(cudaMemcpy(levelTime, d_LevelTime, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost))
    for (unsigned i = 1; i < n; ++i)
    {
        if (levelTime[i] == UNSIGNED_LONG_MAX) break;
        std::cout << "Level: " << i - 1 << " took: " << levelTime[i] - levelTime[i - 1] << std::endl;
    }
    delete[] levelTime;
    gpuErrchk(cudaFree(d_LevelTime))
    //
    */

    return result;
}
