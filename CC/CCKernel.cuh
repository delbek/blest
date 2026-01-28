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

struct CCResult
{
    double time;
    unsigned* components;
    unsigned noComponents;
};

class CCKernel
{
public:
    CCKernel(BitMatrix* matrix);
    CCKernel(const CCKernel& other) = delete;
    CCKernel(CCKernel&& other) noexcept = delete;
    CCKernel& operator=(const CCKernel& other) = delete;
    CCKernel& operator=(CCKernel&& other) noexcept = delete;
    virtual ~CCKernel() = default;

    virtual CCResult hostCode() = 0;
    CCResult run();

protected:
    BitMatrix* matrix;
};

CCKernel::CCKernel(BitMatrix* matrix)
: matrix(matrix)
{

}

CCResult CCKernel::run()
{
    return hostCode();
}
