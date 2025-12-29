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

#include "Common.cuh"

class BitMatrix
{
public:
    BitMatrix() = default;
    BitMatrix(const BitMatrix& other) = delete;
    BitMatrix(BitMatrix&& other) noexcept = delete;
    BitMatrix& operator=(const BitMatrix& other) = delete;
    BitMatrix& operator=(BitMatrix&& other) noexcept = delete;
    virtual ~BitMatrix() = default;
};
