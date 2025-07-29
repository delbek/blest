#include "BitMatrix.cuh"
#include "BRS.cuh"
#include "CSC.cuh"
#include "BRSBFSKernel.cuh"

/*
int main()
{
    bool undirected = true;
    bool binary = true;
    CSC csc("/arf/home/delbek/sutensor/delaunay_n20.mtx", undirected, binary);

    BRS* brs = new BRS;
    brs->constructFromCSCMatrix(&csc);
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS(0, 10, 5);

    brs->save("/arf/home/delbek/sutensor/delaunay_n20.bin");

    delete brs;

    return 0;
}
*/

int main()
{
    BRS* brs = new BRS("/arf/home/delbek/sutensor/delaunay_n20.bin");
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS(0, 10, 5);

    delete brs;

    return 0;
}
