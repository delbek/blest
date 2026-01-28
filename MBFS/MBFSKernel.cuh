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

struct MBFSResult
{
    double time;
    std::vector<unsigned*> levelArrays;
};

class MBFSKernel
{
public:
    MBFSKernel(BitMatrix* matrix);
    MBFSKernel(const MBFSKernel& other) = delete;
    MBFSKernel(MBFSKernel&& other) noexcept = delete;
    MBFSKernel& operator=(const MBFSKernel& other) = delete;
    MBFSKernel& operator=(MBFSKernel&& other) noexcept = delete;
    virtual ~MBFSKernel() = default;

    virtual MBFSResult hostCode() = 0;
    MBFSResult run();

protected:
    BitMatrix* matrix;
};

MBFSKernel::MBFSKernel(BitMatrix* matrix)
: matrix(matrix)
{

}

MBFSResult MBFSKernel::run()
{
    return hostCode();
}
