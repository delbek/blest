#pragma once

#include "CSC.cuh"
#include "BRS.cuh"
#include "BRSBFSKernel.cuh"

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
        {"/arf/scratch/delbek/wikipedia-20070206.mtx", "/arf/scratch/delbek/wikipedia-20070206.txt", false, true},
        {"/arf/scratch/delbek/com-LiveJournal.mtx", "/arf/scratch/delbek/com-LiveJournal.txt", true, true},
        {"/arf/scratch/delbek/wb-edu.mtx", "/arf/scratch/delbek/wb-edu.txt", false, true},
        {"/arf/scratch/delbek/eu-2005.mtx", "/arf/scratch/delbek/eu-2005.txt", false, true},
        {"/arf/scratch/delbek/indochina-2004.mtx", "/arf/scratch/delbek/indochina-2004.txt", false, true},
        {"/arf/scratch/delbek/GAP-road.mtx", "/arf/scratch/delbek/GAP-road.txt", true, false},
        {"/arf/scratch/delbek/uk-2005.mtx", "/arf/scratch/delbek/uk-2005.txt", false, true},
        {"/arf/scratch/delbek/amazon-2008.mtx", "/arf/scratch/delbek/amazon-2008.txt", false, true},
        {"/arf/scratch/delbek/roadNet-CA.mtx", "/arf/scratch/delbek/roadNet-CA.txt", true, true},
        {"/arf/scratch/delbek/rgg_n_2_24_s0.mtx", "/arf/scratch/delbek/rgg_n_2_24_s0.txt", true, true},
        {"/arf/scratch/delbek/GAP-twitter.mtx", "/arf/scratch/delbek/GAP-twitter.txt", false, false},
        {"/arf/scratch/delbek/GAP-web.mtx", "/arf/scratch/delbek/GAP-web.txt", false, false},
        {"/arf/scratch/delbek/GAP-kron.mtx", "/arf/scratch/delbek/GAP-kron.txt", true, false}
        //{"/arf/scratch/delbek/GAP-urand.mtx", "/arf/scratch/delbek/GAP-urand.txt", true, false}
    };

    for (const auto& matrix: matrices)
    {
        std::cout << "Matrix: " << matrix.filename << std::endl;
        double brs = runBRS(matrix);
        std::cout << "BRS time: " << brs << std::endl;
        std::cout << "******************************" << std::endl;
    }
}

double Benchmark::runBRS(const Matrix& matrix)
{
    CSC* csc = new CSC(matrix.filename, matrix.undirected, matrix.binary);
    unsigned sliceSize = 8;
    unsigned* inversePermutation = csc->gorder(sliceSize);

    BRS* brs = new BRS(sliceSize);
    brs->constructFromCSCMatrix(csc);

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    double time = kernel.runBFS(matrix.sourceFile, 15, 5, inversePermutation);

    delete csc;
    delete brs;
    if (inversePermutation != nullptr)
    {
        delete[] inversePermutation;
    }

    return time;
}
