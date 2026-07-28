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
