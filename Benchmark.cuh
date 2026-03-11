/*
 * This file is part of the BLEST repository: https://github.com/delbek/blest
 * Author: Deniz Elbek
 *
 * Please see the paper:
 * 
 * @article{Elbek2025BLEST,
 *   title   = {BLEST: Blazingly Efficient BFS using Tensor Cores},
 *   author  = {Elbek, Deniz and Kaya, Kamer},
 *   journal = {arXiv preprint arXiv:2512.21967},
 *   year    = {2025},
 *   doi     = {10.48550/arXiv.2512.21967},
 *   url     = {https://www.arxiv.org/abs/2512.21967}
 * }
 */

#pragma once

#include <filesystem>
#include <unordered_set>

// Data Structures
#include "CSC.cuh"
#include "BVSS.cuh"
//

// Kernels
#include "BFSKernel.cuh"
#include "MBFSKernel.cuh"
#include "ClosenessKernel.cuh"
#include "CCKernel.cuh"
//

#include "SuiteSparseMatrixDownloader.hpp"
#include "OutputVerifier.cuh"

struct Config
{
    Config(std::string directory, std::string matrixName, bool jackardEnabled, unsigned windowSize, std::string kernelName)
    : directory(directory), matrixName(matrixName), jackardEnabled(jackardEnabled), windowSize(windowSize), kernelName(kernelName) {}

    std::string directory;
    std::string matrixName;
    bool jackardEnabled;
    unsigned windowSize;
    std::string kernelName;
};

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
    ~Benchmark() = default;
    
    void main(const Config& config);
    double run(const Matrix& matrix, const Config& config);
    std::vector<unsigned> constructSourceVertices(std::string filename, unsigned* inversePermutation);
    std::vector<unsigned> constructRandomSourceVertices(std::string filename, unsigned no, unsigned lower, unsigned upper);
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

std::vector<unsigned> Benchmark::constructRandomSourceVertices(std::string filename, unsigned no, unsigned lower, unsigned upper)
{
    std::vector<unsigned> ret;
    for (unsigned i = 0; i < no; ++i)
    {
        ret.emplace_back(rand(lower, upper));
    }
    return ret;
}

void Benchmark::main(const Config& config)
{
    SuiteSparseDownloader downloader;
    SuiteSparseDownloader::MatrixFilter filter;

    filter.names = {
        config.matrixName
    };
    JACKARD_ON = config.jackardEnabled;
    WINDOW_SIZE = config.windowSize;

    std::vector<SuiteSparseDownloader::MatrixInfo> matrices = downloader.getMatrices(filter);
    downloader.downloadMatrices(config.directory, matrices);

    std::cout << "************************************************************************************************************************" << std::endl;
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

        Matrix currentMatrix(matrix.installationPath, config.directory + matrix.name + ".txt", matrix.numericSymmetry, matrix.isBinary);
        double time = run(currentMatrix, config);
        std::cout << "Time: " << time << " ms." << std::endl;
        std::cout << "************************************************************************************************************************" << std::endl;
    }
}

double Benchmark::run(const Matrix& matrix, const Config& config)
{
    #ifdef MPI_AVAILABLE
    MPI_Init(nullptr, nullptr);
    #endif

    constexpr unsigned sliceSize = 8;
    constexpr unsigned noMasks = 32 / sliceSize;
    constexpr bool cscSave = true;
    constexpr bool cscLoad = true;
    constexpr bool orderingSave = true;
    constexpr bool orderingLoad = true;
    constexpr bool bvssSave = true;
    constexpr bool bvssLoad = true;

    std::string kernelName = config.kernelName;

    // binary names
    std::string cscBinaryName = matrix.filename + "_csc.bin";
    std::string orderingBinaryName = matrix.filename + "_ordering.bin";
    std::string bvssBinaryName = matrix.filename + "_bvss.bin";
    //

    // csc
    double startCSC = omp_get_wtime();
    std::cout << "CSC construction started." << std::endl; 
    CSC* csc;
    if (std::filesystem::exists(std::filesystem::path(cscBinaryName)) && cscLoad)
    {
        csc = new CSC;
        csc->constructFromBinary(cscBinaryName);
    }
    else
    {
        csc = new CSC(matrix.filename, matrix.undirected, matrix.binary);
        if (cscSave)
        {
            csc->saveToBinary(cscBinaryName);
        }
    }
    double endCSC = omp_get_wtime();
    std::cout << "CSC construction completed in: " << endCSC - startCSC << " seconds." << std::endl;
    //

    if (csc->isSocialNetwork()) 
    {
        FULL_PADDING = false;
    }
    else
    {
        FULL_PADDING = true;
    }

    // csc ordering
    double startOrder = omp_get_wtime();
    std::cout << "Reordering started." << std::endl;
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
    double endOrder = omp_get_wtime();
    std::cout << "Reordering completed in: " << endOrder - startOrder << " seconds." << std::endl;
    //

    // bvss
    double startBVSS = omp_get_wtime();
    std::cout << "BVSS construction started." << std::endl;
    std::ofstream file(matrix.filename + ".csv");
    BVSS* bvss = new BVSS(sliceSize, noMasks, file);
    if (std::filesystem::exists(std::filesystem::path(bvssBinaryName)) && bvssLoad)
    {
        bvss->constructFromBinary(bvssBinaryName, csc);
    }
    else
    {
        bvss->constructFromCSCMatrix(csc);
        if (bvssSave)
        {
            bvss->saveToBinary(bvssBinaryName);
        }
    }
    double endBVSS = omp_get_wtime();
    std::cout << "BVSS construction completed in: " << endBVSS - startBVSS << " seconds." << std::endl;
    //

    unsigned* permutation = new unsigned[csc->getN()];
    for (unsigned old = 0; old < csc->getN(); ++old)
    {
        permutation[inversePermutation[old]] = old;
    }

    //OutputVerifier verifier;
    double total = 0;

    if (kernelName == "BFS") // BFS
    {
        std::vector<unsigned> sources = this->constructSourceVertices(matrix.sourceFile, inversePermutation);

        // kernel run
        std::cout << "************* RUNNING BFS KERNELS *************" << std::endl;
        BFSKernel* kernel = new BFSKernel(dynamic_cast<BitMatrix*>(bvss));
        std::vector<BFSResult> results = kernel->multiSourceRun(sources);
        //
        
        // processing results
        for (auto& result: results)
        {
            result.sourceVertex = permutation[result.sourceVertex];
            total += result.time;
            unsigned* newLevels = new unsigned[csc->getN()];
            unsigned visitedCount = 0;
            unsigned levelCount = 0;
            for (unsigned old = 0; old < csc->getN(); ++old)
            {
                newLevels[old] = result.levels[inversePermutation[old]];
                if (newLevels[old] != UNSIGNED_MAX)
                {
                    ++visitedCount;
                    levelCount = std::max(levelCount, newLevels[old]);
                }
            }
            std::cout << "Source: " << result.sourceVertex << " - Visited: " << visitedCount << " - Level: " << levelCount << " - Time (ms): " << result.time * 1000 << std::endl;
            delete[] result.levels;
            result.levels = newLevels;
        }
        total /= results.size();
        //

        // kernel cleanup
        for (auto& result: results)
        {
            delete[] result.levels;
        }
        delete kernel;
        //
    }
    else if (kernelName == "MBFS") // MBFS
    {
        std::vector<unsigned> sources = this->constructSourceVertices(matrix.sourceFile, inversePermutation);

        // kernel run
        std::cout << "************* RUNNING MBFS KERNELS *************" << std::endl;
        MBFSKernel* kernel = new MBFSKernel(dynamic_cast<BitMatrix*>(bvss));
        MBFSResult result = kernel->run(sources);
        //

        total = result.time;

        // kernel cleanup
        delete[] result.levels;
        delete kernel;
        //
    }
    else if (kernelName == "Closeness")
    {
        // kernel run
        std::cout << "************* RUNNING CLOSENESS CENTRALITY KERNELS *************" << std::endl;
        ClosenessKernel* kernel = new ClosenessKernel(dynamic_cast<BitMatrix*>(bvss));
        #ifdef MPI_AVAILABLE
        ClosenessResult result = kernel->runMPI();
        #else
        ClosenessResult result = kernel->run();
        #endif
        //

        total = result.time;

        // kernel cleanup
        delete[] result.distances;
        delete kernel;
        //
    }
    else if (kernelName == "CC")
    {
        if (csc->checkSymmetry() == false)
        {
            throw std::runtime_error("Connected components are not well-defined for directed graphs. Try weakly connected components.");
        }

        // kernel run
        std::cout << "************* RUNNING CONNECTED COMPONENT KERNELS *************" << std::endl;
        CCKernel* kernel = new CCKernel(dynamic_cast<BitMatrix*>(bvss));
        CCResult result = kernel->run();
        //

        // finding number of connected components
        unsigned* marker = new unsigned[csc->getN()];
        std::fill(marker, marker + csc->getN(), 0);
        for (unsigned i = 0; i < csc->getN(); ++i)
        {
            ++marker[result.components[i]];
        }
        unsigned noComponents = 0;
        for (unsigned i = 0; i < csc->getN(); ++i)
        {
            if (marker[i] != 0)
            {
                ++noComponents;
                if (marker[i] > 100)
                {
                    std::cout << "A large component found - Size: " << marker[i] << std::endl;
                }
            }
        }
        std::cout << "Number of connected components: " << noComponents << std::endl;
        delete[] marker;
        total = result.time;
        //

        // kernel cleanup
        delete[] result.components;
        delete kernel;
        //
    }
    else if (kernelName == "WCC")
    {
        std::cout << "************* RUNNING WEAKLY CONNECTED COMPONENT KERNELS *************" << std::endl;
        CSC* csc_s = csc->symmetrize();
        std::ofstream file(matrix.filename + "_s.csv");
        BVSS* bvss_s = new BVSS(sliceSize, noMasks, file);
        bvss_s->constructFromCSCMatrix(csc_s);

        // kernel run
        CCKernel* kernel = new CCKernel(dynamic_cast<BitMatrix*>(bvss_s));
        CCResult result = kernel->run();
        //

        // finding number of weakly connected components
        unsigned* marker = new unsigned[csc->getN()];
        std::fill(marker, marker + csc->getN(), 0);
        for (unsigned i = 0; i < csc->getN(); ++i)
        {
            ++marker[result.components[i]];
        }
        unsigned noComponents = 0;
        for (unsigned i = 0; i < csc->getN(); ++i)
        {
            if (marker[i] != 0)
            {
                ++noComponents;
                if (marker[i] > 100)
                {
                    std::cout << "A large component found - Size: " << marker[i] << std::endl;
                }
            }
        }
        std::cout << "Number of weakly connected components: " << noComponents << std::endl;
        delete[] marker;
        total = result.time;
        //

        // kernel cleanup
        delete csc_s;
        delete bvss_s;
        delete kernel;
        //
    }
    else
    {
        throw std::runtime_error("No kernel found.");
    }
    
    // full cleanup
    delete bvss;
    file.close();
    delete[] inversePermutation;
    delete[] permutation;
    delete csc;
    //

    return total * 1000;
}
