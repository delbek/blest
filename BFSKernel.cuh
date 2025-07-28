#ifndef BFSKERNEL_CUH
#define BFSKERNEL_CUH

#include "BitMatrix.cuh"

class BFSKernel
{
public:
    BFSKernel(BitMatrix* matrix);
    BFSKernel(const BFSKernel& other) = delete;
    BFSKernel(BFSKernel&& other) noexcept = delete;
    BFSKernel& operator=(const BFSKernel& other) = delete;
    BFSKernel& operator=(BFSKernel&& other) noexcept = delete;
    virtual ~BFSKernel() = default;

    void runBFS(unsigned sourceVertex, unsigned nRun, unsigned nIgnore);
    virtual double hostCode(unsigned sourceVertex) = 0;

protected:
    BitMatrix* matrix;
};

BFSKernel::BFSKernel(BitMatrix* matrix)
: matrix(matrix)
{

}

void BFSKernel::runBFS(unsigned sourceVertex, unsigned nRun, unsigned nIgnore)
{
    double total = 0;

    for (unsigned i = 0; i < nRun; ++i)
    {
        double time = hostCode(sourceVertex);
        if (i >= nIgnore)
        {
            total += time;
        }
    }

    total /= (nRun - nIgnore);

    std::cout << "Average time took per BFS iteration: " << total * 1000 << " ms." << std::endl;
}

#endif
