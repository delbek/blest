#include "CSC.cuh"
#include "HBRS.cuh"
#include "HBRSBFSKernel.cuh"

int main()
{
    bool undirected = false;
    bool binary = true;
    CSC* csc = new CSC("/arf/home/delbek/sutensor/wikipedia-20070206.mtx", undirected, binary);
    unsigned k;
    unsigned* inversePermutation = csc->hubPartition(k);

    HBRS* hbrs = new HBRS(8, 8);
    hbrs->constructFromCSCMatrix(csc, k);

    HBRSBFSKernel kernel(dynamic_cast<BitMatrix*>(hbrs));
    kernel.runBFS("/arf/home/delbek/sutensor/wikipedia-20070206.txt", 10, 5, inversePermutation);

    delete csc;
    delete hbrs;
    if (inversePermutation != nullptr)
    {
        delete[] inversePermutation;
    }

    return 0;
}
