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
    void runBFS(std::string sourceVerticesFilename, unsigned nRun, unsigned nIgnore, unsigned* inversePermutation);

protected:
    BitMatrix* matrix;
};

BFSKernel::BFSKernel(BitMatrix* matrix)
: matrix(matrix)
{

}

void BFSKernel::runBFS(std::string sourceVerticesFilename, unsigned nRun, unsigned nIgnore, unsigned* inversePermutation)
{
    std::ifstream file(sourceVerticesFilename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file from which to read source vertices.");
    }

    double total = 0;
    unsigned iter = 0;
    unsigned sourceVertex;
    while (file >> sourceVertex)
    {
        double run = 0;
        for (unsigned i = 0; i < nRun; ++i)
        {
            if (inversePermutation != nullptr)
            {
                sourceVertex = inversePermutation[sourceVertex];
            }
            double time = hostCode(sourceVertex);
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

    std::cout << "Average time took per BFS iteration: " << total * 1000 << " ms." << std::endl;

    file.close();
}

#endif
