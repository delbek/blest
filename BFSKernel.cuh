#ifndef BFSKERNEL_CUH
#define BFSKERNEL_CUH

#include "BitMatrix.cuh"

class BFSKernel
{
public:
    BFSKernel(BitMatrix* matrix) = default;
    BFSKernel(const BFSKernel& other) = delete;
    BFSKernel(BFSKernel&& other) noexcept = delete;
    BFSKernel& operator=(const BFSKernel& other) = delete;
    BFSKernel& operator=(BFSKernel&& other) noexcept = delete;
    virtual ~BFSKernel() = default;

    void runBFS(unsigned nRun, unsigned nIgnore);
    void hostCode() = 0;

protected:
    BitMatrix* matrix;
};

BFSKernel::BFSKernel()
: matrix(matrix)
{

}

void BFSKernel::runBFS(unsigned nRun, unsigned nIgnore)
{
    double total = 0;

    for (unsigned i = 0; i < nRun; ++i)
    {
        double start = omp_get_wtime();
        hostCode();
        double end = omp_get_wtime();
        if (i >= nIgnore)
        {
            total += (end - start);
        }
    }

    total /= (nRun - nIgnore);

    std::cout << "Average time took per BFS iteration: " << total << std::endl;
}

#endif
