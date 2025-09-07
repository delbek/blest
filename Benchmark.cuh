#pragma once

#include "CSC.cuh"
#include "BRS.cuh"
#include "BRSBFSKernel.cuh"
#include "HBRS.cuh"
#include "HBRSBFSKernel.cuh"

struct Matrix
{
    Matrix(std::string filename, std::string sourceFile, bool undirected, bool binary)
    : filename(filename),
      sourceFile(sourceFile),
      undirected(undirected),
      binary(binary) {}

    std::string filename;
    std::string sourceFile;
    bool undirected;
    bool binary;
};

class Benchmark
{
public:
    Benchmark() = default;
    Benchmark(const Benchmark& other) = delete;
    Benchmark(Benchmark&& other) noexcept = delete;
    Benchmark& operator=(const Benchmark& other) = delete;
    Benchmark& operator=(Benchmark&& other) noexcept = delete;
    
    void main();
    double runBRS(const Matrix& matrix);
    double runHBRS(const Matrix& matrix);
};

void Benchmark::main()
{
    std::vector<Matrix> matrices = 
    {
        {"/arf/home/delbek/sutensor/wikipedia-20070206.mtx", "/arf/home/delbek/sutensor/wikipedia-20070206.txt", false, true}
        /*
        {"/arf/home/delbek/sutensor/com-LiveJournal.mtx", "/arf/home/delbek/sutensor/com-LiveJournal.txt", true, true},
        {"/arf/home/delbek/sutensor/wb-edu.mtx", "/arf/home/delbek/sutensor/wb-edu.txt", false, true},
        {"/arf/home/delbek/sutensor/eu-2005.mtx", "/arf/home/delbek/sutensor/eu-2005.txt", false, true},
        {"/arf/home/delbek/sutensor/indochina-2004.mtx", "/arf/home/delbek/sutensor/indochina-2004.txt", false, true},
        {"/arf/home/delbek/sutensor/GAP-road.mtx", "/arf/home/delbek/sutensor/GAP-road.txt", true, false}
        */

    };

    for (const auto& matrix: matrices)
    {
        std::cout << "Matrix: " << matrix.filename << std::endl;
        double brs = runBRS(matrix);
        double hbrs = runHBRS(matrix);
        std::cout << "BRS time: " << brs << std::endl;
        std::cout << "HBRS time: " << hbrs << std::endl;
        std::cout << "Speedup: " << brs / hbrs << 'x' << std::endl;
        std::cout << "******************************" << std::endl;
    }
}

double Benchmark::runBRS(const Matrix& matrix)
{
    CSC* csc = new CSC(matrix.filename, matrix.undirected, matrix.binary);
    unsigned* inversePermutation = nullptr;

    BRS* brs = new BRS(8);
    brs->constructFromCSCMatrix(csc);

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    double time = kernel.runBFS(matrix.sourceFile, 10, 5, inversePermutation);

    delete csc;
    delete brs;
    if (inversePermutation != nullptr)
    {
        delete[] inversePermutation;
    }

    return time;
}

double Benchmark::runHBRS(const Matrix& matrix)
{
    CSC* csc = new CSC(matrix.filename, matrix.undirected, matrix.binary);
    unsigned k;
    unsigned* inversePermutation = csc->hubPartition(k);

    HBRS* hbrs = new HBRS(8, 8);
    hbrs->constructFromCSCMatrix(csc, k);

    HBRSBFSKernel kernel(dynamic_cast<BitMatrix*>(hbrs));
    double time = kernel.runBFS(matrix.sourceFile, 10, 5, inversePermutation);

    delete csc;
    delete hbrs;
    if (inversePermutation != nullptr)
    {
        delete[] inversePermutation;
    }

    return time;
}
