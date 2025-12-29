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

#include "BitMatrix.cuh"

struct BFSResult
{
    double time;
    unsigned sourceVertex;
    unsigned* levels;
    unsigned totalLevels;
    unsigned noVisited;
};

class BFSKernel
{
public:
    BFSKernel(BitMatrix* matrix);
    BFSKernel(const BFSKernel& other) = delete;
    BFSKernel(BFSKernel&& other) noexcept = delete;
    BFSKernel& operator=(const BFSKernel& other) = delete;
    BFSKernel& operator=(BFSKernel&& other) noexcept = delete;
    virtual ~BFSKernel() = default;

    virtual BFSResult hostCode(unsigned sourceVertex) = 0;
    BFSResult runBFS(unsigned sourceVertex);

protected:
    BitMatrix* matrix;
};

BFSKernel::BFSKernel(BitMatrix* matrix)
: matrix(matrix)
{

}

BFSResult BFSKernel::runBFS(unsigned sourceVertex)
{
    return hostCode(sourceVertex);
}
