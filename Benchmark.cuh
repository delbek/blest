#pragma once

#include "CSC.cuh"
#include "BVSS.cuh"
#include "BVSSBFSKernel.cuh"
#include <filesystem>
#include "SuiteSparseMatrixDownloader.hpp"
#include <unordered_set>
#include <random>

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
    void generateSourceVertices(std::string filename, unsigned n, unsigned k);
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

void Benchmark::generateSourceVertices(std::string filename, unsigned n, unsigned k)
{
    std::ofstream file(filename);
    std::unordered_set<unsigned> set;

    auto rand = [](unsigned min, unsigned max)
    {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<unsigned> dist(min, max);
        return dist(gen);
    };

    for (unsigned i = 0; i < k; ++i)
    {
        unsigned vertex;
        do
        {
            vertex = rand(0, n - 1);
        } while (set.contains(vertex));
        set.insert(vertex);
        file << vertex << std::endl;
    }
    file.close();
}

void Benchmark::main()
{
    SuiteSparseDownloader downloader;
    SuiteSparseDownloader::MatrixFilter filter;

    /* FULL EXPERIMENT SET */
    filter.names = {
        "GAP-urand"
        /*
        "Spielman_k600",
        "com-Friendster",
        "GAP-kron",
        "mawi_201512020330"
        */
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
        std::cout << "Graph valid: " << matrix.isValid << std::endl;
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
    constexpr bool orderingSave = true;
    constexpr bool orderingLoad = false;
    constexpr bool bvssSave = false;
    constexpr bool bvssLoad = false;

    // csc
    CSC* csc = new CSC(matrix.filename, matrix.undirected, matrix.binary);
    std::cout << "Is symmetric: " << csc->checkSymmetry() << std::endl;
    //
    if (csc->isSocialNetwork()) 
    {
        FULL_PADDING = false;
    }
    else
    {
        FULL_PADDING = true;
    }
    std::cout << "Full Padding: " << FULL_PADDING << std::endl;

    // binary names
    std::string bvssBinaryName = matrix.filename + "_bvss.bin";
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

    // bvss
    std::ofstream file(matrix.filename + ".csv");
    BVSS* bvss = new BVSS(sliceSize, noMasks, file);
    if (std::filesystem::exists(std::filesystem::path(bvssBinaryName)) && bvssLoad)
    {
        bvss->constructFromBinary(bvssBinaryName);
    }
    else
    {
        bvss->constructFromCSCMatrix(csc);
        if (bvssSave)
        {
            bvss->saveToBinary(bvssBinaryName);
        }
    }
    //

    // kernel run
    BVSSBFSKernel* kernel = new BVSSBFSKernel(dynamic_cast<BitMatrix*>(bvss));
    std::vector<unsigned> sources = this->constructSourceVertices(matrix.sourceFile, inversePermutation);
    unsigned nRun = 3;
    unsigned nIgnore = 1;
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
            else
            {
                delete[] result.levels;
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
        bvss->kernelAnalysis(result.sourceVertex, result.totalLevels, result.noVisited, result.time);
    }
    //

    // cleanup
    for (auto& result: results)
    {
        delete[] result.levels;
    }
    delete[] permutation;
    delete kernel;
    
    delete bvss;
    file.close();
    delete[] inversePermutation;
    delete csc;
    //

    return total * 1000;
}
