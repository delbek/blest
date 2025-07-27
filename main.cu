#include "BitMatrix.cuh"
#include "BRS.cuh"
#include "CSC.cuh"
#include "BRSBFSKernel.cuh"

int main()
{
    CSC csc("/arf/home/delbek/sutensor/delaunay_n20.mtx", true, true);

    BRS* brs = new BRS;
    brs->readFromCSCMatrix(&csc);
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS(0, 5, 4);

    delete brs;

    return 0;
}
