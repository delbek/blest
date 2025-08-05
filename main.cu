#include "CSC.cuh"
#include "BRS.cuh"
#include "BRSBFSKernel.cuh"

int main()
{
    bool undirected = true;
    bool binary = true;
    CSC* csc = new CSC("/arf/home/delbek/sutensor/delaunay_n14.mtx", undirected, binary);
    unsigned* inversePermutation = nullptr; //csc->reorderFromFile("ordering_gorder8_com-LiveJournal.bin");

    BRS* brs = new BRS(8);
    brs->constructFromCSCMatrix(csc);
    //brs->save("/arf/home/delbek/sutensor/save_gorder8_delaunay_n20.bin");
    brs->printBRSData();

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    kernel.runBFS("/arf/home/delbek/sutensor/delaunay_n14.txt", 10, 5, inversePermutation);

    delete csc;
    delete brs;
    if (inversePermutation != nullptr)
    {
        delete[] inversePermutation;
    }

    return 0;
}
