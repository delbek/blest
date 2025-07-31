#include "BitMatrix.cuh"
#include "BRS.cuh"
#include "CSC.cuh"
#include "BRSBFSKernel.cuh"

int main()
{
    bool undirected = true;
    bool binary = true;
    CSC csc("/arf/home/delbek/sutensor/delaunay_n20.mtx", undirected, binary);
    //csc.reorderFromFile("/arf/home/delbek/sutensor/PTH-8-CN-RCM_delaunay_n20.bin");

    BRS* brs = new BRS(32);
    brs->constructFromCSCMatrix(&csc);
    brs->save("/arf/home/delbek/sutensor/delaunay_n20.bin");
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS("/arf/home/delbek/sutensor/delaunay_n20.txt", 10, 5);

    delete brs;

    return 0;
}

/*
int main()
{
    BRS* brs = new BRS("/arf/home/delbek/sutensor/delaunay_n20.bin");
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS("/arf/home/delbek/sutensor/delaunay_n20.txt", 1, 0);

    delete brs;

    return 0;
}
*/
