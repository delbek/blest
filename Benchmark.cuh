#pragma once

#include "CSC.cuh"
#include "BRS.cuh"
#include "BRSBFSKernel.cuh"
#include <filesystem>

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
    double run(const Matrix& matrix);
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
        {"/arf/scratch/delbek/roadNet-CA.mtx", "/arf/scratch/delbek/roadNet-CA.txt", true, true},
        {"/arf/scratch/delbek/rgg_n_2_24_s0.mtx", "/arf/scratch/delbek/rgg_n_2_24_s0.txt", true, true},
        {"/arf/scratch/delbek/indochina-2004.mtx", "/arf/scratch/delbek/indochina-2004.txt", false, true},
        {"/arf/scratch/delbek/wikipedia-20070206.mtx", "/arf/scratch/delbek/wikipedia-20070206.txt", false, true},
        {"/arf/scratch/delbek/eu-2005.mtx", "/arf/scratch/delbek/eu-2005.txt", false, true},
        {"/arf/scratch/delbek/wb-edu.mtx", "/arf/scratch/delbek/wb-edu.txt", false, true},
        {"/arf/scratch/delbek/uk-2005.mtx", "/arf/scratch/delbek/uk-2005.txt", false, true},
        {"/arf/scratch/delbek/GAP-twitter.mtx", "/arf/scratch/delbek/GAP-twitter.txt", false, false},
        {"/arf/scratch/delbek/GAP-web.mtx", "/arf/scratch/delbek/GAP-web.txt", false, false},
        {"/arf/scratch/delbek/GAP-kron.mtx", "/arf/scratch/delbek/GAP-kron.txt", true, false},
        {"/arf/scratch/delbek/GAP-urand.mtx", "/arf/scratch/delbek/GAP-urand.txt", true, false}
    };

    for (const auto& matrix: matrices)
    {
        std::cout << "Matrix: " << matrix.filename << std::endl;
        double time = run(matrix);
        std::cout << "Time: " << time << std::endl;
        std::cout << "******************************" << std::endl;
    }
}

double Benchmark::run(const Matrix& matrix)
{
    constexpr unsigned sliceSize = 8;

    // csc
    CSC* csc = new CSC(matrix.filename, matrix.undirected, matrix.binary);
    std::cout << "Is symmetric: " << csc->checkSymmetry() << std::endl;
    //

    // csc ordering
    unsigned* inversePermutation = nullptr;
    if (csc->isRoadNetwork())
    {
        inversePermutation = csc->rcm();
    }
    else
    {
        inversePermutation = csc->gorderWithJackard(sliceSize);
    }
    
    if (inversePermutation == nullptr)
    {
        inversePermutation = new unsigned[csc->getN()];
        for (unsigned i = 0; i < csc->getN(); ++i)
        {
            inversePermutation[i] = i;
        }
    }
    //

    // brs
    std::ofstream file(matrix.filename + ".csv");
    bool fullPadding;
    std::string binaryName;
    if (csc->isRoadNetwork())
    {
        fullPadding = true;
        binaryName = matrix.filename + "_fullpad_natural.bin";
    }
    else
    {
        fullPadding = false;
        binaryName = matrix.filename + "_natural.bin";
    }
    BRS* brs = new BRS(sliceSize, fullPadding, file);
    if (!std::filesystem::exists(std::filesystem::path(binaryName)))
    {
        brs->constructFromCSCMatrix(csc);
        brs->saveToBinary(binaryName);
    }
    else
    {
        brs->constructFromBinary(binaryName);
    }
    //

    // kernel run
    BRSBFSKernel* kernel = new BRSBFSKernel(dynamic_cast<BitMatrix*>(brs));
    std::vector<unsigned> sources = this->constructSourceVertices(matrix.sourceFile, inversePermutation);
    unsigned nRun = 5;
    unsigned nIgnore = 2;
    double total = 0;
    unsigned iter = 0;
    std::vector<BFSResult> results;
    for (const auto source: sources)
    {
        double run = 0;
        for (unsigned i = 0; i < nRun; ++i)
        {
            BFSResult result = kernel->runBFS(source);
            if (i >= nIgnore)
            {
                results.emplace_back(result);
                run += result.time;
            }
        }
    
        run /= (nRun - nIgnore);
        total += run;
        ++iter;
    }
    total /= iter;
    //

    unsigned* permutation = new unsigned[csc->getN()];
    for (unsigned i = 0; i < csc->getN(); ++i)
    {
        permutation[inversePermutation[i]] = i;
    }

    // result save
    for (auto& result: results)
    {
        result.sourceVertex = permutation[result.sourceVertex];
        unsigned* newLevels = new unsigned[csc->getN()];
        for (unsigned old = 0; old < csc->getN(); ++old)
        {
            newLevels[old] = result.levels[inversePermutation[old]];
        }
        delete[] result.levels;
        result.levels = newLevels;
        brs->kernelAnalysis(result.sourceVertex, result.totalLevels, result.noVisited, result.time);
        delete[] result.levels;
    }
    //

    // cleanup
    file.close();
    delete csc;
    delete brs;
    delete kernel;
    delete[] inversePermutation;
    delete[] permutation;
    //

    return total * 1000;
}
