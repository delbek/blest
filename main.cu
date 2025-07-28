#include "BitMatrix.cuh"
#include "BRS.cuh"
#include "CSC.cuh"
#include "BRSBFSKernel.cuh"

int main()
{
    bool undirected = true;
    bool binary = true;
    CSC csc("/arf/home/delbek/sutensor/144.mtx", undirected, binary);

    BRS* brs = new BRS;
    brs->constructFromCSCMatrix(&csc);
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS("/arf/home/delbek/sutensor/144.txt", 5, 4);

    brs->save("/arf/home/delbek/sutensor/144.bin");

    delete brs;

    return 0;
}

/*
int main()
{
    BRS* brs = new BRS("/arf/home/delbek/sutensor/eu-2005.bin");
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS("/arf/home/delbek/sutensor/eu-2005.txt", 5, 4);

    delete brs;

    return 0;
}
*/
