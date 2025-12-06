#pragma once

#include "CSC.cuh"
#include "BRS.cuh"
#include "BRSBFSKernel.cuh"
#include <filesystem>
#include "SuiteSparseMatrixDownloader.hpp"

#define MATRIX_DIRECTORY "/arf/scratch/delbek/"

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
    SuiteSparseDownloader downloader;
    SuiteSparseDownloader::MatrixFilter filter;

    /*
    "GAP-kron"
    "mawi_201512020330"
    */

    /* FULL EXPERIMENT SET */
    filter.names = {
        "webbase-2001",
        "GAP-web",
        "GAP-road",
        "GAP-twitter",
        "uk-2005",
        "europe_osm",
        "road_usa",
        "sk-2005",
        "it-2004",
        "com-Friendster",
        "GAP-urand",
        "kmer_V1r"
    };

    /* COMPRESSION EXPERIMENTS
    filter.names = {
        "web-BerkStan"
    };
    */

    /* UPDATE DIVERGENCE EXPERIMENTS
    filter.names = {
        "GAP-road",
        "europe_osm",
        "delaunay_n24",
        "rgg_n_2_24_s0"
    };
    */

    std::vector<SuiteSparseDownloader::MatrixInfo> matrices = downloader.getMatrices(filter);
    downloader.downloadMatrices(MATRIX_DIRECTORY, matrices);

    for (const auto& matrix: matrices)
    {
        if (!matrix.isValid) continue;

        std::cout
            << "id: "               << matrix.id               << '\n'
            << "groupName: "        << matrix.groupName        << '\n'
            << "name: "             << matrix.name             << '\n'
            << "rows: "             << matrix.rows             << '\n'
            << "cols: "             << matrix.cols             << '\n'
            << "nonzeros: "         << matrix.nonzeros         << '\n'
            << "isReal: "           << (matrix.isReal ? "true" : "false") << '\n'
            << "isBinary: "         << (matrix.isBinary ? "true" : "false") << '\n'
            << "is2d3d: "           << (matrix.is2d3d ? "true" : "false") << '\n'
            << "isPosDef: "         << (matrix.isPosDef ? "true" : "false") << '\n'
            << "patternSymmetry: "  << (matrix.patternSymmetry ? "true" : "false")  << '\n'
            << "numericSymmetry: "  << (matrix.numericSymmetry ? "true" : "false")  << '\n'
            << "kind: "             << matrix.kind             << '\n'
            << "downloadLink: "     << matrix.downloadLink     << '\n'
            << "installationPath: " << matrix.installationPath << '\n'
            << "----------------------------------------" << std::endl;

        Matrix currentMatrix(matrix.installationPath, MATRIX_DIRECTORY + matrix.name + ".txt", matrix.numericSymmetry, matrix.isBinary);
        double time = run(currentMatrix);
        std::cout << "Time: " << time << std::endl;
        std::cout << "******************************" << std::endl;
    }
}

double Benchmark::run(const Matrix& matrix)
{
    constexpr unsigned sliceSize = 8;
    constexpr unsigned noMasks = 32 / sliceSize;
    constexpr bool orderingSave = false;
    constexpr bool orderingLoad = true;
    constexpr bool brsSave = false;
    constexpr bool brsLoad = false;

    // csc
    CSC* csc = new CSC(matrix.filename, matrix.undirected, matrix.binary);
    std::cout << "Is symmetric: " << csc->checkSymmetry() << std::endl;
    //
    if (csc->isSocialNetwork()) FULL_PADDING = false;

    // binary names
    std::string brsBinaryName = matrix.filename + "_brs.bin";
    std::string orderingBinaryName = matrix.filename + "_ordering.bin";

    // csc ordering
    unsigned* inversePermutation = nullptr;
    if (std::filesystem::exists(std::filesystem::path(orderingBinaryName)) && orderingLoad)
    {
        inversePermutation = csc->orderFromBinary(orderingBinaryName);
    }
    else
    {
        inversePermutation = csc->reorder(sliceSize);
        if (orderingSave)
        {
            csc->saveOrderingToBinary(orderingBinaryName, inversePermutation);
        }
    }
    //

    // brs
    std::ofstream file(matrix.filename + ".csv");
    BRS* brs = new BRS(sliceSize, noMasks, csc->isSocialNetwork(), file);
    if (std::filesystem::exists(std::filesystem::path(brsBinaryName)) && brsLoad)
    {
        brs->constructFromBinary(brsBinaryName);
    }
    else
    {
        brs->constructFromCSCMatrix(csc);
        if (brsSave)
        {
            brs->saveToBinary(brsBinaryName);
        }
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
    for (const auto& source: sources)
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
    for (unsigned old = 0; old < csc->getN(); ++old)
    {
        permutation[inversePermutation[old]] = old;
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
    }
    //

    // cleanup
    file.close();
    for (auto& result: results)
    {
        delete[] result.levels;
    }
    delete csc;
    delete brs;
    delete kernel;
    delete[] inversePermutation;
    delete[] permutation;
    //

    return total * 1000;
}
