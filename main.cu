#include "BitMatrix.cuh"
#include "BRS.cuh"
#include "CSC.cuh"
#include "BRSBFSKernel.cuh"

#define FULL_READ

#ifdef FULL_READ
int main()
{
    bool undirected = true;
    bool binary = true;
    CSC* csc = new CSC("/arf/home/delbek/sutensor/com-LiveJournal.mtx", undirected, binary);
    unsigned* inversePermutation = csc->reorderFromFile("/arf/home/delbek/sutensor/ordering_gorder8_com-LiveJournal.bin");

    BRS* brs = new BRS(8);
    brs->constructFromCSCMatrix(csc);
    brs->save("/arf/home/delbek/sutensor/save_gorder8_com-LiveJournal.bin");
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS("/arf/home/delbek/sutensor/com-LiveJournal.txt", 10, 7, inversePermutation);

    delete csc;
    delete brs;
    if (inversePermutation != nullptr)
    {
        delete[] inversePermutation;
    }

    return 0;
}
#else
int main()
{
    BRS* brs = new BRS("/arf/home/delbek/sutensor/gorder8-com-LiveJournal.bin");
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS("/arf/home/delbek/sutensor/com-LiveJournal.txt", 10, 7);

    delete brs;

    return 0;
}
#endif
