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

    virtual double hostCode(unsigned sourceVertex) = 0;
    double runBFS(const std::vector<unsigned>& sources, unsigned nRun, unsigned nIgnore);

protected:
    BitMatrix* matrix;
};

BFSKernel::BFSKernel(BitMatrix* matrix)
: matrix(matrix)
{

}

double BFSKernel::runBFS(const std::vector<unsigned>& sources, unsigned nRun, unsigned nIgnore)
{
    double total = 0;
    unsigned iter = 0;
    for (const auto source: sources)
    {
        double run = 0;
        for (unsigned i = 0; i < nRun; ++i)
        {
            double time = hostCode(source);
            if (i >= nIgnore)
            {
                run += time;
            }
        }
    
        run /= (nRun - nIgnore);
        total += run;
        ++iter;
    }
    total /= iter;

    return total * 1000;
}

#endif
