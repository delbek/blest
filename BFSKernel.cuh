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
