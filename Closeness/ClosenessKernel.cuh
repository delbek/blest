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
#include "BVSSClosenessKernels.cuh"
#include <array>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#ifdef MPI_AVAILABLE
#include "mpi.h"
#endif

struct ClosenessResult
{
    double time;
    unsigned long long* distances;
};

class ClosenessKernel
{
public:
    ClosenessKernel(BVSS* m_matrix);
    ClosenessKernel(const ClosenessKernel& other) = delete;
    ClosenessKernel(ClosenessKernel&& other) noexcept = delete;
    ClosenessKernel& operator=(const ClosenessKernel& other) = delete;
    ClosenessKernel& operator=(ClosenessKernel&& other) noexcept = delete;
    ~ClosenessKernel() = default;

    ClosenessResult run();

private:
    BVSS* m_matrix;
};

ClosenessKernel::ClosenessKernel(BVSS* matrix)
: m_matrix(matrix)
{

}

ClosenessResult ClosenessKernel::run()
{
    int rank = 0;
    int worldSize = 1;

    #ifdef MPI_AVAILABLE
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    #endif
 
    gpuErrchk(cudaSetDevice(0));

    BVSS* bvss = m_matrix;
    CSC* csc = bvss->getCSR();
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
    bool lazyKernel = (bvss->getUpdateDivergence() > LAZY_KERNEL_THRESHOLD);

    unsigned n = bvss->getN();
    unsigned paddedN = std::ceil(static_cast<double>(n) / 8) * 8;
    unsigned partitionSize = paddedN / 8;

    unsigned noChunks = (n + 255) / 256;
    unsigned baseChunksPerRank = noChunks / worldSize;
    unsigned remainderChunks = noChunks % worldSize;
    unsigned localChunkCount = baseChunksPerRank + (rank < remainderChunks ? 1u : 0u);
    unsigned localChunkBegin = rank * baseChunksPerRank + std::min(static_cast<unsigned>(rank), remainderChunks);
    unsigned localChunkEnd = localChunkBegin + localChunkCount;
    constexpr unsigned taskSize = TASK_SIZE;
    unsigned noTasks = std::ceil(static_cast<double>(localChunkEnd - localChunkBegin) / taskSize);

    ClosenessResult result;
    result.distances = new unsigned long long[paddedN];

    auto allocateSharedMemory = [](int blockSize) -> size_t
    {
        return 0;
    };

    // tune
    bool switching = false;
    if (lazyKernel)
    {
        BFSKernel* kernel = new BFSKernel(m_matrix);
        switching = kernel->shouldSwitch(20);
        if (switching)
        {
            std::cout << "Switching-based kernel is enabled." << std::endl;
        }
        delete kernel;
    }
    //

    void* kernelPtr;
    if (sliceSize == 8)
    {
        if (switching)
        {
            kernelPtr = (void*)BVSSClosenessKernels::BVSSCloseness8EnhancedSliceSize8NoMasks4LazyChunkFusionSwitching;
        }
        else
        {
            kernelPtr = (void*)BVSSClosenessKernels::BVSSCloseness8EnhancedSliceSize8NoMasks4LazyChunkFusion;
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
                                                    
    unsigned* d_RowPtrs;
    unsigned* d_ColIds;
    unsigned* d_N;
    unsigned* d_PaddedN;
    unsigned* d_NoRealSliceSets;
    SLICE_TYPE* d_SliceSetPtrs;
    unsigned* d_VirtualToReal;
    unsigned* d_RealPtrs;
    unsigned* d_RowIds;
    MASK* d_Masks;

    ulonglong4_32a* d_Frontier;
    unsigned* d_SparseFrontierIds;
    unsigned* d_UnvisitedCurrentSize;
    unsigned* d_FrontierCurrentSize;
    ulonglong4_32a* d_VisitedNext;
    unsigned* d_SparseFrontierNextIds;
    unsigned* d_UnvisitedNextSize;
    unsigned* d_FrontierNextSize;
    ulonglong4_32a* d_Visited;
    unsigned long long* d_Far;
    ulonglong4_32a* d_ActiveRSets;
    bool* d_DirtyRSets;

    // data structure
    gpuErrchk(cudaMalloc(&d_N, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_PaddedN, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_RowPtrs, sizeof(unsigned) * (csc->getN() + 1)))
    gpuErrchk(cudaMalloc(&d_ColIds, sizeof(unsigned) * csc->getNNZ()))
    gpuErrchk(cudaMalloc(&d_NoRealSliceSets, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_SliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_VirtualToReal, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_RealPtrs, sizeof(unsigned) * (noRealSliceSets + 1)))
    gpuErrchk(cudaMalloc(&d_RowIds, sizeof(unsigned) * noSlices))
    gpuErrchk(cudaMalloc(&d_Masks, sizeof(MASK) * (noSlices / noMasks)))

    gpuErrchk(cudaMemcpy(d_N, &n, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_PaddedN, &paddedN, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowPtrs, csc->getColPtrs(), sizeof(unsigned) * (csc->getN() + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_ColIds, csc->getRows(), sizeof(unsigned) * csc->getNNZ(), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_NoRealSliceSets, &noRealSliceSets, sizeof(unsigned), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_SliceSetPtrs, sliceSetPtrs, sizeof(SLICE_TYPE) * (noSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_VirtualToReal, virtualToReal, sizeof(unsigned) * noSliceSets, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RealPtrs, realPtrs, sizeof(unsigned) * (noRealSliceSets + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_RowIds, rowIds, sizeof(unsigned) * noSlices, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_Masks, masks, sizeof(MASK) * (noSlices / noMasks), cudaMemcpyHostToDevice))

    // algorithm
    gpuErrchk(cudaMalloc(&d_Frontier, sizeof(ulonglong4_32a) * paddedN * taskSize))
    gpuErrchk(cudaMalloc(&d_SparseFrontierIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_UnvisitedCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierCurrentSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_VisitedNext, sizeof(ulonglong4_32a) * paddedN * taskSize))
    gpuErrchk(cudaMalloc(&d_SparseFrontierNextIds, sizeof(unsigned) * noSliceSets))
    gpuErrchk(cudaMalloc(&d_UnvisitedNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_FrontierNextSize, sizeof(unsigned)))
    gpuErrchk(cudaMalloc(&d_Visited, sizeof(ulonglong4_32a) * paddedN * taskSize))
    gpuErrchk(cudaMalloc(&d_Far, sizeof(unsigned long long) * paddedN))
    gpuErrchk(cudaMalloc(&d_ActiveRSets, sizeof(ulonglong4_32a) * noRealSliceSets * taskSize))
    gpuErrchk(cudaMalloc(&d_DirtyRSets, sizeof(bool) * noRealSliceSets))

    gpuErrchk(cudaMemset(d_Far, 0, sizeof(unsigned long long) * paddedN))

    std::vector<unsigned> initialVset;
    std::unordered_map<unsigned, ulonglong4_32a> rsets;
    std::unordered_map<unsigned, ulonglong4_32a> vertices;

    double next = 0.1;

    double start = omp_get_wtime();
    for (unsigned taskNo = 0; taskNo < noTasks; ++taskNo)
    {
        // task reset
        gpuErrchk(cudaMemset(d_Frontier, 0, sizeof(ulonglong4_32a) * paddedN * taskSize))
        gpuErrchk(cudaMemset(d_Visited, 0, sizeof(ulonglong4_32a) * paddedN * taskSize))
        gpuErrchk(cudaMemset(d_VisitedNext, 0, sizeof(ulonglong4_32a) * paddedN * taskSize))
        gpuErrchk(cudaMemset(d_ActiveRSets, 0, sizeof(ulonglong4_32a) * noRealSliceSets * taskSize))
        gpuErrchk(cudaMemset(d_DirtyRSets, 0, sizeof(bool) * noRealSliceSets))
        rsets.clear();
        rsets.reserve(256 * taskSize);
        initialVset.clear();
        initialVset.reserve(256 * taskSize);
        //

        unsigned chunkStart = localChunkBegin + taskNo * taskSize;
        unsigned chunkEnd = std::min(localChunkEnd, chunkStart + taskSize);
        for (unsigned chunk = chunkStart; chunk < chunkEnd; ++chunk)
        {
            // chunk reset
            for (auto& rset: rsets)
            {
                rset.second = {0, 0, 0, 0};
            }
            vertices.clear();
            vertices.reserve(256);
            //

            unsigned vertexStart = chunk * 256;
            unsigned vertexEnd = std::min(n, vertexStart + 256);
            for (unsigned sourceVertex = vertexStart; sourceVertex < vertexEnd; ++sourceVertex)
            {
                unsigned f = sourceVertex - vertexStart;
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
                        rsets[rset] = {(1ull << relativeF), 0ull, 0ull, 0ull};
                    }
                    else
                    {
                        rsets[rset].x |= (1ull << relativeF);
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
                        rsets[rset] = {0ull, (1ull << relativeF), 0ull, 0ull};
                    }
                    else
                    {
                        rsets[rset].y |= (1ull << relativeF);
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
                        rsets[rset] = {0ull, 0ull, (1ull << relativeF), 0ull};
                    }
                    else
                    {
                        rsets[rset].z |= (1ull << relativeF);
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
                        rsets[rset] = {0ull, 0ull, 0ull, (1ull << relativeF)};
                    }
                    else
                    {
                        rsets[rset].w |= (1ull << relativeF);
                    }
                }
            }

            unsigned chunkNo = chunk - chunkStart;
            for (const auto& vertex: vertices)
            {
                gpuErrchk(cudaMemcpy(d_Frontier + chunkNo * paddedN + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(ulonglong4_32a), cudaMemcpyHostToDevice))
                gpuErrchk(cudaMemcpy(d_Visited + chunkNo * paddedN + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(ulonglong4_32a), cudaMemcpyHostToDevice))
                gpuErrchk(cudaMemcpy(d_VisitedNext + chunkNo * paddedN + getVertexIndex(vertex.first, partitionSize), &vertex.second, sizeof(ulonglong4_32a), cudaMemcpyHostToDevice))
            }

            for (const auto& rset: rsets)
            {
                if (rset.second.x != 0 || rset.second.y != 0 || rset.second.z != 0 || rset.second.w != 0)
                {
                    gpuErrchk(cudaMemcpy(d_ActiveRSets + chunkNo * noRealSliceSets + rset.first, &rset.second, sizeof(ulonglong4_32a), cudaMemcpyHostToDevice))
                }
            }
        }

        unsigned initialFrontierSize = initialVset.size();
        gpuErrchk(cudaMemcpy(d_SparseFrontierIds, initialVset.data(), sizeof(unsigned) * initialFrontierSize, cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_FrontierCurrentSize, &initialFrontierSize, sizeof(unsigned), cudaMemcpyHostToDevice))
        unsigned initialUnvisitedSize = UNSIGNED_MAX;
        gpuErrchk(cudaMemcpy(d_UnvisitedCurrentSize, &initialUnvisitedSize, sizeof(unsigned), cudaMemcpyHostToDevice))
        
        std::array<void*, 22> args =
        {
            (void*)&d_N,
            (void*)&d_PaddedN,
            (void*)&d_NoRealSliceSets,
            (void*)&d_RowPtrs,
            (void*)&d_ColIds,
            (void*)&d_SliceSetPtrs,
            (void*)&d_VirtualToReal,
            (void*)&d_RealPtrs,
            (void*)&d_RowIds,
            (void*)&d_Masks,
            (void*)&d_Far,
            (void*)&d_ActiveRSets,
            (void*)&d_DirtyRSets,
            (void*)&d_Frontier,
            (void*)&d_Visited,
            (void*)&d_SparseFrontierIds,
            (void*)&d_UnvisitedCurrentSize,
            (void*)&d_FrontierCurrentSize,
            (void*)&d_VisitedNext,
            (void*)&d_SparseFrontierNextIds,
            (void*)&d_UnvisitedNextSize,
            (void*)&d_FrontierNextSize
        };

        double kernelStart = omp_get_wtime();
        gpuErrchk(cudaLaunchCooperativeKernel(
            kernelPtr,
            gridSize,
            blockSize,
            args.data(),
            allocateSharedMemory(blockSize),
            0))
        gpuErrchk(cudaPeekAtLastError())
        gpuErrchk(cudaDeviceSynchronize())

        double pct = static_cast<double>(taskNo + 1) / noTasks;
        if (pct > next)
        {
            next += 0.1;          
            std::cout << "Rank: " << rank << " - Completed: " << pct * 100 << "%" << std::endl << std::flush;
        }
    }
    double end = omp_get_wtime();

    double localTime = (end - start);
    result.time = localTime;

    gpuErrchk(cudaMemcpy(result.distances, d_Far, sizeof(unsigned long long) * paddedN, cudaMemcpyDeviceToHost))
    #ifdef MPI_AVAILABLE
    if (rank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, result.distances, static_cast<int>(paddedN), MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE, &result.time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Reduce(result.distances, nullptr, static_cast<int>(paddedN), MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localTime, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    if (rank != 0)
    {
        std::exit(0);
    }
    #endif

    gpuErrchk(cudaFree(d_N))
    gpuErrchk(cudaFree(d_PaddedN))
    gpuErrchk(cudaFree(d_RowPtrs))
    gpuErrchk(cudaFree(d_ColIds))
    gpuErrchk(cudaFree(d_NoRealSliceSets))
    gpuErrchk(cudaFree(d_SliceSetPtrs))
    gpuErrchk(cudaFree(d_VirtualToReal))
    gpuErrchk(cudaFree(d_RealPtrs))
    gpuErrchk(cudaFree(d_RowIds))
    gpuErrchk(cudaFree(d_Masks))
    gpuErrchk(cudaFree(d_Frontier))
    gpuErrchk(cudaFree(d_SparseFrontierIds))
    gpuErrchk(cudaFree(d_UnvisitedCurrentSize))
    gpuErrchk(cudaFree(d_FrontierCurrentSize))
    gpuErrchk(cudaFree(d_VisitedNext))
    gpuErrchk(cudaFree(d_SparseFrontierNextIds))
    gpuErrchk(cudaFree(d_UnvisitedNextSize))
    gpuErrchk(cudaFree(d_FrontierNextSize))
    gpuErrchk(cudaFree(d_Visited))
    gpuErrchk(cudaFree(d_Far))
    gpuErrchk(cudaFree(d_ActiveRSets))
    gpuErrchk(cudaFree(d_DirtyRSets))

    return result;
}
