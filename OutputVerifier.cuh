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

#include "BFSKernel.cuh"
#include "MBFSKernel.cuh"
#include "ClosenessKernel.cuh"

class OutputVerifier
{
public:
    OutputVerifier() = default;
    OutputVerifier(const OutputVerifier& other) = delete;
    OutputVerifier(OutputVerifier&& other) noexcept = delete;
    OutputVerifier& operator=(const OutputVerifier& other) = delete;
    OutputVerifier& operator=(OutputVerifier&& other) noexcept = delete;
    ~OutputVerifier() = default;

    void verifyBFSOutput(CSC* csc, const BFSResult& result);
    void verifyMBFSOutput(CSC* csc, const MBFSResult& result);
    void verifyClosenessOutput(CSC* csc, const ClosenessResult& result);

private:
    unsigned* cpuBFS(CSC* csc, unsigned source);
    unsigned* cpuMBFS(CSC* csc, const std::vector<unsigned>& sources);
};

void OutputVerifier::verifyBFSOutput(CSC* csc, const BFSResult& result)
{
    std::cout << "VERIFYING BFS OUTPUT..." << std::endl;
    unsigned* cpuResult = this->cpuBFS(csc, result.sourceVertex);
    for (unsigned i = 0; i < csc->getN(); ++i)
    {
        if (cpuResult[i] != result.levels[i])
        {
            throw std::runtime_error("BFS algorithm is incorrect.");
        }
    }
    std::cout << "Kernel is correct." << std::endl;

    delete[] cpuResult;
}

void OutputVerifier::verifyMBFSOutput(CSC* csc, const MBFSResult& result)
{
    std::cout << "VERIFYING MBFS OUTPUT..." << std::endl;
    unsigned long long paddedN = static_cast<unsigned long long>(std::ceil(static_cast<double>(csc->getN()) / 8) * 8);
    unsigned long long total = (result.sources.size() * paddedN);
    unsigned* cpuResult = this->cpuMBFS(csc, result.sources);
    for (unsigned long long i = 0; i < total; ++i)
    {
        if (result.levels[i] != cpuResult[i])
        {
            throw std::runtime_error("MBFS algorithm is incorrect.");
        }
    }
    std::cout << "Kernel is correct." << std::endl;
    
    delete[] cpuResult;
}

void OutputVerifier::verifyClosenessOutput(CSC* csc, const ClosenessResult& result)
{
    unsigned paddedN = static_cast<unsigned>(std::ceil(static_cast<double>(csc->getN()) / 8) * 8);
    unsigned partitionSize = paddedN / 8;

    unsigned long long* far = new unsigned long long[paddedN];
    std::fill(far, far + paddedN, 0);
    
    std::vector<unsigned> q(csc->getN());
    std::vector<bool> visited(csc->getN());

    for (unsigned f = 0; f < csc->getN(); ++f)
    {
        std::fill(visited.begin(), visited.end(), false);
        unsigned qs = 0;
        unsigned qe = 0;
        q[qe++] = f;
        visited[f] = true;
        
        unsigned currentLevel = 0;
        unsigned levelEnd = 1;

        while (qs < qe)
        {
            unsigned u = q[qs++];
            far[getVertexIndex(u, partitionSize)] += currentLevel;
            for (unsigned nnz = csc->getColPtrs()[u]; nnz < csc->getColPtrs()[u + 1]; ++nnz)
            {
                unsigned v = csc->getRows()[nnz];
                if (!visited[v])
                {
                    q[qe++] = v;
                    visited[v] = true;
                }
            }
            if (qs == levelEnd)
            {
                levelEnd = qe;
                ++currentLevel;
            }
        }
    }
    
    for (unsigned i = 0; i < csc->getN(); ++i)
    {
        if (result.distances[i] != far[i])
        {
            throw std::runtime_error("Closeness centrality algorithm is incorrect.");
        }
    }
    std::cout << "Kernel is correct" << std::endl;

    delete[] far;
}

unsigned* OutputVerifier::cpuBFS(CSC* csc, unsigned source)
{
    unsigned* levels = new unsigned[csc->getN()];
    std::fill(levels, levels + csc->getN(), UNSIGNED_MAX);

    std::vector<unsigned> q(csc->getN());
    std::vector<bool> visited(csc->getN(), false);
    unsigned qs = 0;
    unsigned qe = 0;
    q[qe++] = source;
    visited[source] = true;
    
    unsigned currentLevel = 0;
    unsigned levelEnd = 1;

    while (qs < qe)
    {
        unsigned u = q[qs++];
        levels[u] = currentLevel;
        for (unsigned nnz = csc->getColPtrs()[u]; nnz < csc->getColPtrs()[u + 1]; ++nnz)
        {
            unsigned v = csc->getRows()[nnz];
            if (!visited[v])
            {
                q[qe++] = v;
                visited[v] = true;
            }
        }
        if (qs == levelEnd)
        {
            levelEnd = qe;
            ++currentLevel;
        }
    }

    return levels;
}

unsigned* OutputVerifier::cpuMBFS(CSC* csc, const std::vector<unsigned>& sources)
{
    unsigned long long paddedN = static_cast<unsigned long long>(std::ceil(static_cast<double>(csc->getN()) / 8) * 8);
    unsigned partitionSize = paddedN / 8;

    unsigned* levels = new unsigned[sources.size() * paddedN];
    std::fill(levels, levels + paddedN * sources.size(), UNSIGNED_MAX);

    std::vector<unsigned> q(csc->getN());
    std::vector<bool> visited(csc->getN());

    for (unsigned long long f = 0; f < sources.size(); ++f)
    {
        std::fill(visited.begin(), visited.end(), false);
        unsigned qs = 0;
        unsigned qe = 0;
        q[qe++] = sources[f];
        visited[sources[f]] = true;
        
        unsigned currentLevel = 0;
        unsigned levelEnd = 1;

        while (qs < qe)
        {
            unsigned u = q[qs++];
            levels[f * paddedN + getVertexIndex(u, partitionSize)] = currentLevel;
            for (unsigned nnz = csc->getColPtrs()[u]; nnz < csc->getColPtrs()[u + 1]; ++nnz)
            {
                unsigned v = csc->getRows()[nnz];
                if (!visited[v])
                {
                    q[qe++] = v;
                    visited[v] = true;
                }
            }
            if (qs == levelEnd)
            {
                levelEnd = qe;
                ++currentLevel;
            }
        }
    }

    return levels;
}
