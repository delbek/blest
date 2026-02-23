#pragma once

#include "BFSKernel.cuh"

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

private:
    unsigned* cpuBFS(CSC* csc, unsigned source);
};

void OutputVerifier::verifyBFSOutput(CSC* csc, const BFSResult& result)
{
    unsigned* cpuResult = this->cpuBFS(csc, result.sourceVertex);
    for (unsigned i = 0; i < csc->getN(); ++i)
    {
        if (cpuResult[i] != result.levels[i])
        {
            throw std::runtime_error("BFS algorithm is incorrect.");
        }
    }
    delete[] cpuResult;
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
