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

#include "CSC.cuh"
#include "BVSS.cuh"
#include "BVSSBFSKernel.cuh"
#include "BVSSCCKernel.cuh"
#include <filesystem>
#include "SuiteSparseMatrixDownloader.hpp"
#include <unordered_set>

struct Config
{
    Config(std::string directory, std::string matrixName, bool jackardEnabled, unsigned windowSize)
    : directory(directory), matrixName(matrixName), jackardEnabled(jackardEnabled), windowSize(windowSize) {}

    std::string directory;
    std::string matrixName;
    bool jackardEnabled;
    unsigned windowSize;
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
    
    void main(const Config& config);
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
        double time = run(currentMatrix);
        std::cout << "Time: " << time << " ms." << std::endl;
        std::cout << "******************************" << std::endl;
    }
}

double Benchmark::run(const Matrix& matrix)
{
    constexpr unsigned sliceSize = 8;
    constexpr unsigned noMasks = 32 / sliceSize;
    constexpr bool cscSave = true;
    constexpr bool cscLoad = true;
    constexpr bool orderingSave = true;
    constexpr bool orderingLoad = true;
    constexpr bool bvssSave = false; // never set true atm, it is not saving CSR
    constexpr bool bvssLoad = false; // never set true atm, it is not saving CSR
    
    std::string kernelName = "BFS";

    // binary names
    std::string cscBinaryName = matrix.filename + "_csc.bin";
    std::string orderingBinaryName = matrix.filename + "_ordering.bin";
    std::string bvssBinaryName = matrix.filename + "_bvss.bin";
    //

    // csc
    CSC* csc;
    if (std::filesystem::exists(std::filesystem::path(cscBinaryName)) && cscLoad)
    {
        csc = new CSC;
        csc->constructFromBinary(cscBinaryName);
    }
    else
    {
        std::cout << "CSC construction started." << std::endl;
        csc = new CSC(matrix.filename, matrix.undirected, matrix.binary);
        if (cscSave)
        {
            csc->saveToBinary(cscBinaryName);
        }
        std::cout << "CSC constructed." << std::endl;
    }
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
        std::cout << "BVSS construction started." << std::endl;
        bvss->constructFromCSCMatrix(csc);
        if (bvssSave)
        {
            bvss->saveToBinary(bvssBinaryName);
        }
        std::cout << "BVSS constructed." << std::endl;
    }
    //

    double total = 0;
    if (kernelName == "BFS")
    {
        std::cout << "BFS kernels launching..." << std::endl;
        // kernel run
        BVSSBFSKernel* kernel = new BVSSBFSKernel(dynamic_cast<BitMatrix*>(bvss));
        std::vector<unsigned> sources = this->constructSourceVertices(matrix.sourceFile, inversePermutation);
        unsigned* permutation = new unsigned[csc->getN()];
        for (unsigned old = 0; old < csc->getN(); ++old)
        {
            permutation[inversePermutation[old]] = old;
        }
        unsigned nRun = 1;
        unsigned nIgnore = 0;
        unsigned iter = 0;
        std::vector<BFSResult> results;
        for (const auto& source: sources)
        {
            double run = 0;
            for (unsigned i = 0; i < nRun; ++i)
            {
                BFSResult result = kernel->run(source);
                result.sourceVertex = permutation[result.sourceVertex];
                std::cout << "Source: " << result.sourceVertex << " - Number of levels processed: " << result.totalLevels << " - Total visited: " << result.noVisited << " - Time took: " << result.time * 1000 << " ms." << std::endl;
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

        // result save
        for (auto& result: results)
        {
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

        // kernel cleanup
        for (auto& result: results)
        {
            delete[] result.levels;
        }
        delete[] permutation;
        delete kernel;
        //
    }
    else if (kernelName == "CC")
    {
        std::cout << "CC kernels launching..." << std::endl;
        // kernel run
        BVSSCCKernel* kernel = new BVSSCCKernel(dynamic_cast<BitMatrix*>(bvss));
        CCResult result = kernel->run();
        total = result.time;
        std::cout << "Number of components identified: " << result.noComponents << std::endl;
        unsigned* newComponents = new unsigned[csc->getN()];
        for (unsigned old = 0; old < csc->getN(); ++old)
        {
            newComponents[old] = result.components[inversePermutation[old]];
        }
        delete[] result.components;
        result.components = newComponents;
        //

        // kernel cleanup
        delete[] result.components;
        delete kernel;
        //
    }
    else
    {
        std::cout << "No kernel found." << std::endl;
    }
    
    // full cleanup
    delete bvss;
    file.close();
    delete[] inversePermutation;
    delete csc;
    //

    return total * 1000;
}
