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
    std::vector<unsigned> constructSourceVertices(std::string filename, unsigned* inversePermutation);
};

std::vector<unsigned> Benchmark::constructSourceVertices(std::string filename, unsigned* inversePermutation)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file from which to read source vertices.");
    }

    std::vector<unsigned> sources;

    unsigned sourceVertex;
    while (file >> sourceVertex)
    {
        --sourceVertex;
        sources.emplace_back(inversePermutation[sourceVertex]);   
    }
    file.close();

    return sources;
}

void Benchmark::main()
{
    std::vector<Matrix> matrices = 
    {
        {"/arf/scratch/delbek/GAP-road.mtx", "/arf/scratch/delbek/GAP-road.txt", true, false},
        {"/arf/scratch/delbek/roadNet-CA.mtx", "/arf/scratch/delbek/roadNet-CA.txt", true, true}
        /*
        {"/arf/scratch/delbek/rgg_n_2_24_s0.mtx", "/arf/scratch/delbek/rgg_n_2_24_s0.txt", true, true},
        {"/arf/scratch/delbek/eu-2005.mtx", "/arf/scratch/delbek/eu-2005.txt", false, true},
        {"/arf/scratch/delbek/indochina-2004.mtx", "/arf/scratch/delbek/indochina-2004.txt", false, true},
        {"/arf/scratch/delbek/uk-2005.mtx", "/arf/scratch/delbek/uk-2005.txt", false, true},
        {"/arf/scratch/delbek/wb-edu.mtx", "/arf/scratch/delbek/wb-edu.txt", false, true},
        {"/arf/scratch/delbek/wikipedia-20070206.mtx", "/arf/scratch/delbek/wikipedia-20070206.txt", false, true},
        {"/arf/scratch/delbek/com-LiveJournal.mtx", "/arf/scratch/delbek/com-LiveJournal.txt", true, true},
        {"/arf/scratch/delbek/amazon-2008.mtx", "/arf/scratch/delbek/amazon-2008.txt", false, true},
        {"/arf/scratch/delbek/GAP-web.mtx", "/arf/scratch/delbek/GAP-web.txt", false, false},
        {"/arf/scratch/delbek/GAP-twitter.mtx", "/arf/scratch/delbek/GAP-twitter.txt", false, false},
        {"/arf/scratch/delbek/GAP-kron.mtx", "/arf/scratch/delbek/GAP-kron.txt", true, false},
        {"/arf/scratch/delbek/GAP-urand.mtx", "/arf/scratch/delbek/GAP-urand.txt", true, false}
        */
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
    unsigned sliceSize = 8;
    bool fullPadding = false;
    
    CSC* csc = new CSC(matrix.filename, matrix.undirected, matrix.binary);
    
    std::cout << "Average Bandwidth before ordering: " << csc->averageBandwidth() << std::endl;
    std::cout << "Max Bandwidth before ordering: " << csc->maxBandwidth() << std::endl;
    std::cout << "Average Profile before ordering: " << csc->averageProfile() << std::endl;
    std::cout << "Max Profile before ordering: " << csc->maxProfile() << std::endl;
    unsigned* inversePermutation = nullptr;
    if (csc->checkSymmetry())
    {
        inversePermutation = csc->rcm();
    }
    else
    {
        inversePermutation = csc->gorderWithJackard(sliceSize);
    }
    std::cout << "------" << std::endl;
    std::cout << "Average Bandwidth after ordering: " << csc->averageBandwidth() << std::endl;
    std::cout << "Max Bandwidth after ordering: " << csc->maxBandwidth() << std::endl;
    std::cout << "Average Profile after ordering: " << csc->averageProfile() << std::endl;
    std::cout << "Max Profile after ordering: " << csc->maxProfile() << std::endl;

    std::ofstream file(matrix.filename + ".csv");
    BRS* brs = new BRS(sliceSize, fullPadding, file);
    brs->constructFromCSCMatrix(csc);

    if (inversePermutation == nullptr)
    {
        inversePermutation = new unsigned[csc->getN()];
        for (unsigned i = 0; i < csc->getN(); ++i)
        {
            inversePermutation[i] = i;
        }
    }
    std::vector<unsigned> sources = this->constructSourceVertices(matrix.sourceFile, inversePermutation);

    BRSBFSKernel kernel(dynamic_cast<BitMatrix*>(brs));
    double time = kernel.runBFS(sources, 1, 0);

    file.close();
    delete csc;
    delete brs;
    delete[] inversePermutation;

    return time;
}
