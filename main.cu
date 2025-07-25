#include "BitMatrix.cuh"
#include "BRS.cuh"
#include "CSC.cuh"

int main()
{
    CSC csc("/arf/home/delbek/sutensor/delaunay_n20.mtx", true, true);

    BitMatrix* matrix = new BRS();
    BRS* brs = dynamic_cast<BRS*>(matrix);
    brs->readFromCSCMatrix(&csc);
    brs->printBRSData();

    delete matrix;

    return 0;
}
