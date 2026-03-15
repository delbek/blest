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

#include "BVSS.cuh"
#include "BVSSMBFSKernels.cuh"
#include <array>

struct MBFSResult
{
    double time;
    std::vector<unsigned> sources;
    unsigned* levels;
};

class MBFSKernel
{
public:
    MBFSKernel(BitMatrix* m_matrix);
    MBFSKernel(const MBFSKernel& other) = delete;
    MBFSKernel(MBFSKernel&& other) noexcept = delete;
    MBFSKernel& operator=(const MBFSKernel& other) = delete;
    MBFSKernel& operator=(MBFSKernel&& other) noexcept = delete;
    ~MBFSKernel() = default;

    MBFSResult run(const std::vector<unsigned>& sourceVertices);

private:
    MBFSResult run32(const std::vector<unsigned>& sourceVertices);
    MBFSResult run64(const std::vector<unsigned>& sourceVertices);
    MBFSResult run128(const std::vector<unsigned>& sourceVertices);
    MBFSResult run256(const std::vector<unsigned>& sourceVertices);

private:
    BitMatrix* m_matrix;
};

MBFSKernel::MBFSKernel(BitMatrix* matrix)
: m_matrix(matrix)
{

}

MBFSResult MBFSKernel::run(const std::vector<unsigned>& sourceVertices)
{
    MBFSResult result;
    unsigned noBFS = sourceVertices.size();
    if (noBFS <= 32)
    {
        result = this->run32(sourceVertices);
    }
    else if (noBFS <= 64)
    {
        result = this->run64(sourceVertices);
    }
    else if (noBFS <= 128)
    {
        result = this->run128(sourceVertices);
    }
    else if (noBFS <= 256)
    {
        result = this->run256(sourceVertices);
    }
    else
    {
        throw std::runtime_error("At most 256 BFSs can be executed at the same time.");
    }
    return result;
}

MBFSResult MBFSKernel::run32(const std::vector<unsigned>& sourceVertices)
{
    gpuErrchk(cudaSetDevice(0))

    BVSS* bvss = dynamic_cast<BVSS*>(m_matrix);
    CSC* csc = bvss->getCSR();
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

    unsigned long long n = static_cast<unsigned long long>(bvss->getN());
    unsigned long long paddedN = static_cast<unsigned long long>(std::ceil(static_cast<double>(n) / 8) * 8);
    unsigned partitionSize = paddedN / 8;

    MBFSResult result;
    result.sources = sourceVertices;
    result.levels = new unsigned[paddedN * sourceVertices.size()];
    std::fill(result.levels, result.levels + paddedN * sourceVertices.size(), UNSIGNED_MAX);
    for (unsigned f = 0; f < sourceVertices.size(); ++f)
    {
        result.levels[f * paddedN + getVertexIndex(sourceVertices[f], partitionSize)] = 0;
    }

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    void* kernelPtr = (void*)BVSSMBFSKernels::BVSSMBFS8EnhancedSliceSize8NoMasks4Lazy32BFS;

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
    unsigned long long* d_N;
    unsigned long long* d_PaddedN;
    unsigned* d_NoSliceSets;
    SLICE_TYPE* d_SliceSetPtrs;
    unsigned* d_VirtualToReal;
    unsigned* d_RealPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    unsigned* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_FrontierCurrentSize;
    unsigned* d_VisitedNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_FrontierNextSize;
    unsigned* d_Visited;
    unsigned* d_Levels;
    unsigned* d_ActiveRSets;

    // data structure
    gpuErrchk(cudaMalloc(&d_N, sizeof(unsigned long long)))
    gpuErrchk(cudaMalloc(&d_PaddedN, sizeof(unsigned long long)))
    gpuErrchk(cudaMalloc(&d_RowPtrs, sizeof(unsigned) * (csc->getN() + 1)))
    gpuErrchk(cudaMalloc(&d_ColIds, sizeof(unsigned) * csc->getNNZ()))
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_VirtualToReal, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_RealPtrs, sizeof(unsigned) * (noRealSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_N, &n, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_PaddedN, &paddedN, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowPtrs, csc->getColPtrs(), sizeof(unsigned) * (csc->getN() + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_ColIds, csc->getRows(), sizeof(unsigned) * csc->getNNZ(), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VirtualToReal, virtualToReal, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RealPtrs, realPtrs, sizeof(unsigned) * (noRealSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(unsigned) * paddedN))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_VisitedNext, sizeof(unsigned) * paddedN))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(unsigned) * paddedN))
    gpuErrchk(cudaMalloc(&d_Levels, sizeof(unsigned) * paddedN * sourceVertices.size()))
    gpuErrchk(cudaMalloc(&d_ActiveRSets, sizeof(unsigned) * noRealSliceSets))
    
    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(unsigned) * paddedN))
    gpuErrchk(cudaMemset(d_VisitedNext, 0, sizeof(unsigned) * paddedN))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(unsigned) * paddedN))
    gpuErrchk(cudaMemset(d_ActiveRSets, 0, sizeof(unsigned) * noRealSliceSets))
    gpuErrchk(cudaMemcpy(d_Levels, result.levels, sizeof(unsigned) * paddedN * sourceVertices.size(), cudaMemcpyHostToDevice))

    std::unordered_map<unsigned, unsigned> rsets;
    std::vector<unsigned> initialVset;
    std::unordered_map<unsigned, unsigned> vertices;

    for (unsigned f = 0; f < sourceVertices.size(); ++f)
    {
        unsigned sourceVertex = sourceVertices[f];

        if (vertices.contains(sourceVertex))
        {
            vertices[sourceVertex] |= (1u << f);
        }
        else
        {
            vertices[sourceVertex] = (1u << f);
        }

        unsigned rset = sourceVertex / sliceSize;
        if (!rsets.contains(rset))
        {
            for (unsigned vset = realPtrs[rset]; vset < realPtrs[rset + 1]; ++vset)
            {
                initialVset.emplace_back(vset);
            }
            unsigned activeBit = (1u << f);
            rsets[rset] = activeBit;
        }
        else
        {
            unsigned activeBit = (1u << f);
            rsets[rset] |= activeBit;
        }
    }

    unsigned initialFrontierSize = initialVset.size();
    gpuErrchk(cudaMemcpy(d_SparseFrontierIds, initialVset.data(), sizeof(unsigned) * initialFrontierSize, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &initialFrontierSize, sizeof(unsigned), cudaMemcpyHostToDevice))

    for (const auto& rset: rsets)
    {
        gpuErrchk(cudaMemcpy(d_ActiveRSets + rset.first, &rset.second, sizeof(unsigned), cudaMemcpyHostToDevice))
    }

    for (const auto& vertex: vertices)
    {
        gpuErrchk(cudaMemcpy(d_Frontier + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(unsigned), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_Visited + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(unsigned), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_VisitedNext + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(unsigned), cudaMemcpyHostToDevice))
    }

    std::array<void*, 18> args =
    {
        (void*)&d_N,
        (void*)&d_PaddedN,
        (void*)&d_RowPtrs,
        (void*)&d_ColIds,
        (void*)&d_SliceSetPtrs,
        (void*)&d_VirtualToReal,
        (void*)&d_RealPtrs,
        (void*)&d_RowIds,
        (void*)&d_Masks,
        (void*)&d_Levels,
        (void*)&d_ActiveRSets,
        (void*)&d_Frontier,
        (void*)&d_Visited,
        (void*)&d_SparseFrontierIds,
        (void*)&d_FrontierCurrentSize,
        (void*)&d_VisitedNext,
        (void*)&d_SparseFrontierNextIds,
        (void*)&d_FrontierNextSize
    };

    double start = omp_get_wtime();
    gpuErrchk(cudaLaunchCooperativeKernel(
        kernelPtr,
        gridSize,
        blockSize,
        args.data(),
        allocateSharedMemory(blockSize),
        0))
    gpuErrchk(cudaPeekAtLastError())
    gpuErrchk(cudaDeviceSynchronize())
    double end = omp_get_wtime();

    result.time = (end - start);

    gpuErrchk(cudaMemcpy(result.levels, d_Levels, sizeof(unsigned) * paddedN * sourceVertices.size(), cudaMemcpyDeviceToHost))

    gpuErrchk(cudaFree(d_N))
    gpuErrchk(cudaFree(d_PaddedN))
    gpuErrchk(cudaFree(d_RowPtrs))
    gpuErrchk(cudaFree(d_ColIds))
    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_VirtualToReal))
    gpuErrchk(cudaFree(d_RealPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_SparseFrontierIds))
    gpuErrchk(cudaFree(d_FrontierCurrentSize))
    gpuErrchk(cudaFree(d_VisitedNext))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_Levels))
    gpuErrchk(cudaFree(d_ActiveRSets))

    return result;
}

MBFSResult MBFSKernel::run64(const std::vector<unsigned>& sourceVertices)
{
    gpuErrchk(cudaSetDevice(0))

    BVSS* bvss = dynamic_cast<BVSS*>(m_matrix);
    CSC* csc = bvss->getCSR();
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

    unsigned long long n = static_cast<unsigned long long>(bvss->getN());
    unsigned long long paddedN = static_cast<unsigned long long>(std::ceil(static_cast<double>(n) / 8) * 8);
    unsigned partitionSize = paddedN / 8;

    MBFSResult result;
    result.sources = sourceVertices;
    result.levels = new unsigned[paddedN * sourceVertices.size()];
    std::fill(result.levels, result.levels + paddedN * sourceVertices.size(), UNSIGNED_MAX);
    for (unsigned f = 0; f < sourceVertices.size(); ++f)
    {
        result.levels[f * paddedN + getVertexIndex(sourceVertices[f], partitionSize)] = 0;
    }

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    void* kernelPtr = (void*)BVSSMBFSKernels::BVSSMBFS8EnhancedSliceSize8NoMasks4Lazy64BFS;

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
    unsigned long long* d_N;
    unsigned long long* d_PaddedN;
    unsigned* d_NoSliceSets;
    SLICE_TYPE* d_SliceSetPtrs;
    unsigned* d_VirtualToReal;
    unsigned* d_RealPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    unsigned long long* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_FrontierCurrentSize;
    unsigned long long* d_VisitedNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_FrontierNextSize;
    unsigned long long* d_Visited;
    unsigned* d_Levels;
    unsigned long long* d_ActiveRSets;

    // data structure
    gpuErrchk(cudaMalloc(&d_N, sizeof(unsigned long long)))
    gpuErrchk(cudaMalloc(&d_PaddedN, sizeof(unsigned long long)))
    gpuErrchk(cudaMalloc(&d_RowPtrs, sizeof(unsigned) * (csc->getN() + 1)))
    gpuErrchk(cudaMalloc(&d_ColIds, sizeof(unsigned) * csc->getNNZ()))
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_VirtualToReal, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_RealPtrs, sizeof(unsigned) * (noRealSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_N, &n, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_PaddedN, &paddedN, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowPtrs, csc->getColPtrs(), sizeof(unsigned) * (csc->getN() + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_ColIds, csc->getRows(), sizeof(unsigned) * csc->getNNZ(), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VirtualToReal, virtualToReal, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RealPtrs, realPtrs, sizeof(unsigned) * (noRealSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(unsigned long long) * paddedN))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_VisitedNext, sizeof(unsigned long long) * paddedN))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(unsigned long long) * paddedN))
    gpuErrchk(cudaMalloc(&d_Levels, sizeof(unsigned) * paddedN * sourceVertices.size()))
    gpuErrchk(cudaMalloc(&d_ActiveRSets, sizeof(unsigned long long) * noRealSliceSets))
    
    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(unsigned long long) * paddedN))
    gpuErrchk(cudaMemset(d_VisitedNext, 0, sizeof(unsigned long long) * paddedN))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(unsigned long long) * paddedN))
    gpuErrchk(cudaMemset(d_ActiveRSets, 0, sizeof(unsigned long long) * noRealSliceSets))
    gpuErrchk(cudaMemcpy(d_Levels, result.levels, sizeof(unsigned) * paddedN * sourceVertices.size(), cudaMemcpyHostToDevice))

    std::unordered_map<unsigned, unsigned long long> rsets;
    std::vector<unsigned> initialVset;
    std::unordered_map<unsigned, unsigned long long> vertices;
    
    for (unsigned f = 0; f < sourceVertices.size(); ++f)
    {
        unsigned sourceVertex = sourceVertices[f];
        if (vertices.contains(sourceVertex))
        {
            vertices[sourceVertex] |= (1ull << f);
        }
        else
        {
            vertices[sourceVertex] = (1ull << f);
        }
        unsigned rset = sourceVertex / sliceSize;
        if (!rsets.contains(rset))
        {
            for (unsigned vset = realPtrs[rset]; vset < realPtrs[rset + 1]; ++vset)
            {
                initialVset.emplace_back(vset);
            }
            unsigned long long activeBit = (1ull << f);
            rsets[rset] = activeBit;
        }
        else
        {
            unsigned long long current = rsets[rset];
            unsigned long long activeBit = (1ull << f);
            current |= activeBit;
            rsets[rset] = current;
        }
    }
    unsigned initialFrontierSize = initialVset.size();
    gpuErrchk(cudaMemcpy(d_SparseFrontierIds, initialVset.data(), sizeof(unsigned) * initialFrontierSize, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &initialFrontierSize, sizeof(unsigned), cudaMemcpyHostToDevice))

    for (const auto& rset: rsets)
    {
        gpuErrchk(cudaMemcpy(d_ActiveRSets + rset.first, &rset.second, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    }

    for (const auto& vertex: vertices)
    {
        gpuErrchk(cudaMemcpy(d_Frontier + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(unsigned long long), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_Visited + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(unsigned long long), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_VisitedNext + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    }

    std::array<void*, 18> args =
    {
        (void*)&d_N,
        (void*)&d_PaddedN,
        (void*)&d_RowPtrs,
        (void*)&d_ColIds,
        (void*)&d_SliceSetPtrs,
        (void*)&d_VirtualToReal,
        (void*)&d_RealPtrs,
        (void*)&d_RowIds,
        (void*)&d_Masks,
        (void*)&d_Levels,
        (void*)&d_ActiveRSets,
        (void*)&d_Frontier,
        (void*)&d_Visited,
        (void*)&d_SparseFrontierIds,
        (void*)&d_FrontierCurrentSize,
        (void*)&d_VisitedNext,
        (void*)&d_SparseFrontierNextIds,
        (void*)&d_FrontierNextSize
    };

    double start = omp_get_wtime();
    gpuErrchk(cudaLaunchCooperativeKernel(
        kernelPtr,
        gridSize,
        blockSize,
        args.data(),
        allocateSharedMemory(blockSize),
        0))
    gpuErrchk(cudaPeekAtLastError())
    gpuErrchk(cudaDeviceSynchronize())
    double end = omp_get_wtime();

    result.time = (end - start);

    gpuErrchk(cudaMemcpy(result.levels, d_Levels, sizeof(unsigned) * paddedN * sourceVertices.size(), cudaMemcpyDeviceToHost))

    gpuErrchk(cudaFree(d_N))
    gpuErrchk(cudaFree(d_PaddedN))
    gpuErrchk(cudaFree(d_RowPtrs))
    gpuErrchk(cudaFree(d_ColIds))
    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_VirtualToReal))
    gpuErrchk(cudaFree(d_RealPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_SparseFrontierIds))
    gpuErrchk(cudaFree(d_FrontierCurrentSize))
    gpuErrchk(cudaFree(d_VisitedNext))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_Levels))
    gpuErrchk(cudaFree(d_ActiveRSets))

    return result;
}

MBFSResult MBFSKernel::run128(const std::vector<unsigned>& sourceVertices)
{
    gpuErrchk(cudaSetDevice(0))

    BVSS* bvss = dynamic_cast<BVSS*>(m_matrix);
    CSC* csc = bvss->getCSR();
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

    unsigned long long n = static_cast<unsigned long long>(bvss->getN());
    unsigned long long paddedN = static_cast<unsigned long long>(std::ceil(static_cast<double>(n) / 8) * 8);
    unsigned partitionSize = paddedN / 8;

    MBFSResult result;
    result.sources = sourceVertices;
    result.levels = new unsigned[paddedN * sourceVertices.size()];
    std::fill(result.levels, result.levels + paddedN * sourceVertices.size(), UNSIGNED_MAX);
    for (unsigned f = 0; f < sourceVertices.size(); ++f)
    {
        result.levels[f * paddedN + getVertexIndex(sourceVertices[f], partitionSize)] = 0;
    }

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    void* kernelPtr = (void*)BVSSMBFSKernels::BVSSMBFS8EnhancedSliceSize8NoMasks4Lazy128BFS;

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
    unsigned long long* d_N;
    unsigned long long* d_PaddedN;
    unsigned* d_NoSliceSets;
    SLICE_TYPE* d_SliceSetPtrs;
    unsigned* d_VirtualToReal;
    unsigned* d_RealPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    ulonglong2* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_FrontierCurrentSize;
    ulonglong2* d_VisitedNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_FrontierNextSize;
    ulonglong2* d_Visited;
    unsigned* d_Levels;
    ulonglong2* d_ActiveRSets;

    // data structure
    gpuErrchk(cudaMalloc(&d_N, sizeof(unsigned long long)))
    gpuErrchk(cudaMalloc(&d_PaddedN, sizeof(unsigned long long)))
    gpuErrchk(cudaMalloc(&d_RowPtrs, sizeof(unsigned) * (csc->getN() + 1)))
    gpuErrchk(cudaMalloc(&d_ColIds, sizeof(unsigned) * csc->getNNZ()))
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_VirtualToReal, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_RealPtrs, sizeof(unsigned) * (noRealSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_N, &n, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_PaddedN, &paddedN, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowPtrs, csc->getColPtrs(), sizeof(unsigned) * (csc->getN() + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_ColIds, csc->getRows(), sizeof(unsigned) * csc->getNNZ(), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VirtualToReal, virtualToReal, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RealPtrs, realPtrs, sizeof(unsigned) * (noRealSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(ulonglong2) * paddedN))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_VisitedNext, sizeof(ulonglong2) * paddedN))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(ulonglong2) * paddedN))
    gpuErrchk(cudaMalloc(&d_Levels, sizeof(unsigned) * paddedN * sourceVertices.size()))
    gpuErrchk(cudaMalloc(&d_ActiveRSets, sizeof(ulonglong2) * noRealSliceSets))
    
    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(ulonglong2) * paddedN))
    gpuErrchk(cudaMemset(d_VisitedNext, 0, sizeof(ulonglong2) * paddedN))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(ulonglong2) * paddedN))
    gpuErrchk(cudaMemset(d_ActiveRSets, 0, sizeof(ulonglong2) * noRealSliceSets))
    gpuErrchk(cudaMemcpy(d_Levels, result.levels, sizeof(unsigned) * paddedN * sourceVertices.size(), cudaMemcpyHostToDevice))

    std::unordered_map<unsigned, ulonglong2> rsets;
    std::vector<unsigned> initialVset;
    std::unordered_map<unsigned, ulonglong2> vertices;

    for (unsigned f = 0; f < sourceVertices.size(); ++f)
    {
        unsigned sourceVertex = sourceVertices[f];

        unsigned relativeF = f;
        if (f < 64)
        {
            if (vertices.contains(sourceVertex))
            {
                vertices[sourceVertex].x |= (1ull << f);
            }
            else
            {
                vertices[sourceVertex] = {(1ull << f), 0ull};
            }

            unsigned rset = sourceVertex / sliceSize;
            if (!rsets.contains(rset))
            {
                for (unsigned vset = realPtrs[rset]; vset < realPtrs[rset + 1]; ++vset)
                {
                    initialVset.emplace_back(vset);
                }
                unsigned long long activeBit = (1ull << f);
                rsets[rset] = {activeBit, 0ull};
            }
            else
            {
                unsigned long long activeBit = (1ull << f);
                rsets[rset].x |= activeBit;
            }
        }
        else
        {
            relativeF -= 64;

            if (vertices.contains(sourceVertex))
            {
                vertices[sourceVertex].y |= (1ull << relativeF);
            }
            else
            {
                vertices[sourceVertex] = {0ull, (1ull << relativeF)};
            }

            unsigned rset = sourceVertex / sliceSize;
            if (!rsets.contains(rset))
            {
                for (unsigned vset = realPtrs[rset]; vset < realPtrs[rset + 1]; ++vset)
                {
                    initialVset.emplace_back(vset);
                }
                unsigned long long activeBit = (1ull << relativeF);
                rsets[rset] = {0ull, activeBit};
            }
            else
            {
                unsigned long long activeBit = (1ull << relativeF);
                rsets[rset].y |= activeBit;
            }
        }
    }

    unsigned initialFrontierSize = initialVset.size();
    gpuErrchk(cudaMemcpy(d_SparseFrontierIds, initialVset.data(), sizeof(unsigned) * initialFrontierSize, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &initialFrontierSize, sizeof(unsigned), cudaMemcpyHostToDevice))

    for (const auto& rset: rsets)
    {
        gpuErrchk(cudaMemcpy(d_ActiveRSets + rset.first, &rset.second, sizeof(ulonglong2), cudaMemcpyHostToDevice))
    }

    for (const auto& vertex: vertices)
    {
        gpuErrchk(cudaMemcpy(d_Frontier + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(ulonglong2), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_Visited + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(ulonglong2), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_VisitedNext + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(ulonglong2), cudaMemcpyHostToDevice))
    }

    std::array<void*, 18> args =
    {
        (void*)&d_N,
        (void*)&d_PaddedN,
        (void*)&d_RowPtrs,
        (void*)&d_ColIds,
        (void*)&d_SliceSetPtrs,
        (void*)&d_VirtualToReal,
        (void*)&d_RealPtrs,
        (void*)&d_RowIds,
        (void*)&d_Masks,
        (void*)&d_Levels,
        (void*)&d_ActiveRSets,
        (void*)&d_Frontier,
        (void*)&d_Visited,
        (void*)&d_SparseFrontierIds,
        (void*)&d_FrontierCurrentSize,
        (void*)&d_VisitedNext,
        (void*)&d_SparseFrontierNextIds,
        (void*)&d_FrontierNextSize
    };

    double start = omp_get_wtime();
    gpuErrchk(cudaLaunchCooperativeKernel(
        kernelPtr,
        gridSize,
        blockSize,
        args.data(),
        allocateSharedMemory(blockSize),
        0))
    gpuErrchk(cudaPeekAtLastError())
    gpuErrchk(cudaDeviceSynchronize())
    double end = omp_get_wtime();

    result.time = (end - start);

    gpuErrchk(cudaMemcpy(result.levels, d_Levels, sizeof(unsigned) * paddedN * sourceVertices.size(), cudaMemcpyDeviceToHost))

    gpuErrchk(cudaFree(d_N))
    gpuErrchk(cudaFree(d_PaddedN))
    gpuErrchk(cudaFree(d_RowPtrs))
    gpuErrchk(cudaFree(d_ColIds))
    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_VirtualToReal))
    gpuErrchk(cudaFree(d_RealPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_SparseFrontierIds))
    gpuErrchk(cudaFree(d_FrontierCurrentSize))
    gpuErrchk(cudaFree(d_VisitedNext))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_Levels))
    gpuErrchk(cudaFree(d_ActiveRSets))

    return result;
}

MBFSResult MBFSKernel::run256(const std::vector<unsigned>& sourceVertices)
{
    gpuErrchk(cudaSetDevice(0))

    BVSS* bvss = dynamic_cast<BVSS*>(m_matrix);
    CSC* csc = bvss->getCSR();
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

    unsigned long long n = static_cast<unsigned long long>(bvss->getN());
    unsigned long long paddedN = static_cast<unsigned long long>(std::ceil(static_cast<double>(n) / 8) * 8);
    unsigned partitionSize = paddedN / 8;

    MBFSResult result;
    result.sources = sourceVertices;
    result.levels = new unsigned[paddedN * sourceVertices.size()];
    std::fill(result.levels, result.levels + paddedN * sourceVertices.size(), UNSIGNED_MAX);
    for (unsigned f = 0; f < sourceVertices.size(); ++f)
    {
        result.levels[f * paddedN + getVertexIndex(sourceVertices[f], partitionSize)] = 0;
    }

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    void* kernelPtr = (void*)BVSSMBFSKernels::BVSSMBFS8EnhancedSliceSize8NoMasks4Lazy256BFS;

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
    unsigned long long* d_N;
    unsigned long long* d_PaddedN;
    unsigned* d_NoSliceSets;
    SLICE_TYPE* d_SliceSetPtrs;
    unsigned* d_VirtualToReal;
    unsigned* d_RealPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    ulonglong4_32a* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_FrontierCurrentSize;
    ulonglong4_32a* d_VisitedNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_FrontierNextSize;
    ulonglong4_32a* d_Visited;
    unsigned* d_Levels;
    ulonglong4_32a* d_ActiveRSets;

    // data structure
    gpuErrchk(cudaMalloc(&d_N, sizeof(unsigned long long)))
    gpuErrchk(cudaMalloc(&d_PaddedN, sizeof(unsigned long long)))
    gpuErrchk(cudaMalloc(&d_RowPtrs, sizeof(unsigned) * (csc->getN() + 1)))
    gpuErrchk(cudaMalloc(&d_ColIds, sizeof(unsigned) * csc->getNNZ()))
    gpuErrchk(cudaMalloc(&d_NoSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_VirtualToReal, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_RealPtrs, sizeof(unsigned) * (noRealSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_N, &n, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_PaddedN, &paddedN, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowPtrs, csc->getColPtrs(), sizeof(unsigned) * (csc->getN() + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_ColIds, csc->getRows(), sizeof(unsigned) * csc->getNNZ(), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoSliceSets, &noSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VirtualToReal, virtualToReal, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RealPtrs, realPtrs, sizeof(unsigned) * (noRealSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(ulonglong4_32a) * paddedN))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_VisitedNext, sizeof(ulonglong4_32a) * paddedN))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(ulonglong4_32a) * paddedN))
    gpuErrchk(cudaMalloc(&d_Levels, sizeof(unsigned) * paddedN * sourceVertices.size()))
    gpuErrchk(cudaMalloc(&d_ActiveRSets, sizeof(ulonglong4_32a) * noRealSliceSets))
    
    gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(ulonglong4_32a) * paddedN))
    gpuErrchk(cudaMemset(d_VisitedNext, 0, sizeof(ulonglong4_32a) * paddedN))
    gpuErrchk(cudaMemset(d_Visited, 0, sizeof(ulonglong4_32a) * paddedN))
    gpuErrchk(cudaMemset(d_ActiveRSets, 0, sizeof(ulonglong4_32a) * noRealSliceSets))
    gpuErrchk(cudaMemcpy(d_Levels, result.levels, sizeof(unsigned) * paddedN * sourceVertices.size(), cudaMemcpyHostToDevice))

    std::unordered_map<unsigned, ulonglong4_32a> rsets;
    std::vector<unsigned> initialVset;
    std::unordered_map<unsigned, ulonglong4_32a> vertices;

    for (unsigned f = 0; f < sourceVertices.size(); ++f)
    {
        unsigned sourceVertex = sourceVertices[f];

        unsigned relativeF = f;
        if (f < 64)
        {
            if (vertices.contains(sourceVertex))
            {
                vertices[sourceVertex].x |= (1ull << f);
            }
            else
            {
                vertices[sourceVertex] = {(1ull << f), 0ull, 0ull, 0ull};
            }

            unsigned rset = sourceVertex / sliceSize;
            if (!rsets.contains(rset))
            {
                for (unsigned vset = realPtrs[rset]; vset < realPtrs[rset + 1]; ++vset)
                {
                    initialVset.emplace_back(vset);
                }
                unsigned long long activeBit = (1ull << f);
                rsets[rset] = {activeBit, 0ull, 0ull, 0ull};
            }
            else
            {
                unsigned long long activeBit = (1ull << f);
                rsets[rset].x |= activeBit;
            }
        }
        else if (f < 128)
        {
            relativeF -= 64;

            if (vertices.contains(sourceVertex))
            {
                vertices[sourceVertex].y |= (1ull << relativeF);
            }
            else
            {
                vertices[sourceVertex] = {0ull, (1ull << relativeF), 0ull, 0ull};
            }

            unsigned rset = sourceVertex / sliceSize;
            if (!rsets.contains(rset))
            {
                for (unsigned vset = realPtrs[rset]; vset < realPtrs[rset + 1]; ++vset)
                {
                    initialVset.emplace_back(vset);
                }
                unsigned long long activeBit = (1ull << relativeF);
                rsets[rset] = {0ull, activeBit, 0ull, 0ull};
            }
            else
            {
                unsigned long long activeBit = (1ull << relativeF);
                rsets[rset].y |= activeBit;
            }
        }
        else if (f < 192)
        {
            relativeF -= 128;

            if (vertices.contains(sourceVertex))
            {
                vertices[sourceVertex].z |= (1ull << relativeF);
            }
            else
            {
                vertices[sourceVertex] = {0ull, 0ull, (1ull << relativeF), 0ull};
            }

            unsigned rset = sourceVertex / sliceSize;
            if (!rsets.contains(rset))
            {
                for (unsigned vset = realPtrs[rset]; vset < realPtrs[rset + 1]; ++vset)
                {
                    initialVset.emplace_back(vset);
                }
                unsigned long long activeBit = (1ull << relativeF);
                rsets[rset] = {0ull, 0ull, activeBit, 0ull};
            }
            else
            {
                unsigned long long activeBit = (1ull << relativeF);
                rsets[rset].z |= activeBit;
            }
        }
        else
        {
            relativeF -= 192;

            if (vertices.contains(sourceVertex))
            {
                vertices[sourceVertex].w |= (1ull << relativeF);
            }
            else
            {
                vertices[sourceVertex] = {0ull, 0ull, 0ull, (1ull << relativeF)};
            }

            unsigned rset = sourceVertex / sliceSize;
            if (!rsets.contains(rset))
            {
                for (unsigned vset = realPtrs[rset]; vset < realPtrs[rset + 1]; ++vset)
                {
                    initialVset.emplace_back(vset);
                }
                unsigned long long activeBit = (1ull << relativeF);
                rsets[rset] = {0ull, 0ull, 0ull, activeBit};
            }
            else
            {
                unsigned long long activeBit = (1ull << relativeF);
                rsets[rset].w |= activeBit;
            }
        }
    }

    unsigned initialFrontierSize = initialVset.size();
    gpuErrchk(cudaMemcpy(d_SparseFrontierIds, initialVset.data(), sizeof(unsigned) * initialFrontierSize, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &initialFrontierSize, sizeof(unsigned), cudaMemcpyHostToDevice))

    for (const auto& rset: rsets)
    {
        gpuErrchk(cudaMemcpy(d_ActiveRSets + rset.first, &rset.second, sizeof(ulonglong4_32a), cudaMemcpyHostToDevice))
    }

    for (const auto& vertex: vertices)
    {
        gpuErrchk(cudaMemcpy(d_Frontier + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(ulonglong4_32a), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_Visited + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(ulonglong4_32a), cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_VisitedNext + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(ulonglong4_32a), cudaMemcpyHostToDevice))
    }

    std::array<void*, 18> args =
    {
        (void*)&d_N,
        (void*)&d_PaddedN,
        (void*)&d_RowPtrs,
        (void*)&d_ColIds,
        (void*)&d_SliceSetPtrs,
        (void*)&d_VirtualToReal,
        (void*)&d_RealPtrs,
        (void*)&d_RowIds,
        (void*)&d_Masks,
        (void*)&d_Levels,
        (void*)&d_ActiveRSets,
        (void*)&d_Frontier,
        (void*)&d_Visited,
        (void*)&d_SparseFrontierIds,
        (void*)&d_FrontierCurrentSize,
        (void*)&d_VisitedNext,
        (void*)&d_SparseFrontierNextIds,
        (void*)&d_FrontierNextSize
    };

    double start = omp_get_wtime();
    gpuErrchk(cudaLaunchCooperativeKernel(
        kernelPtr,
        gridSize,
        blockSize,
        args.data(),
        allocateSharedMemory(blockSize),
        0))
    gpuErrchk(cudaPeekAtLastError())
    gpuErrchk(cudaDeviceSynchronize())
    double end = omp_get_wtime();

    result.time = (end - start);

    gpuErrchk(cudaMemcpy(result.levels, d_Levels, sizeof(unsigned) * paddedN * sourceVertices.size(), cudaMemcpyDeviceToHost))

    gpuErrchk(cudaFree(d_N))
    gpuErrchk(cudaFree(d_PaddedN))
    gpuErrchk(cudaFree(d_RowPtrs))
    gpuErrchk(cudaFree(d_ColIds))
    gpuErrchk(cudaFree(d_NoSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_VirtualToReal))
    gpuErrchk(cudaFree(d_RealPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_SparseFrontierIds))
    gpuErrchk(cudaFree(d_FrontierCurrentSize))
    gpuErrchk(cudaFree(d_VisitedNext))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_Levels))
    gpuErrchk(cudaFree(d_ActiveRSets))

    return result;
}
