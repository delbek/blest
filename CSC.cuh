#ifndef CSC_CUH
#define CSC_CUH

#include "Common.cuh"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>

// Gorder
#include "Graph.h"
// Gorder

class CSC
{
public:
    CSC() = default;
    CSC(std::string filename, bool undirected, bool binary);
    CSC(const CSC& other) = delete;
    CSC(CSC&& other) noexcept = delete;
    CSC& operator=(const CSC& other) = delete;
    CSC& operator=(CSC&& other) noexcept = delete;
    ~CSC();

    [[nodiscard]] inline unsigned& getN() {return m_N;}
    [[nodiscard]] inline unsigned& getNNZ() {return m_NNZ;}
    [[nodiscard]] inline unsigned*& getColPtrs() {return m_ColPtrs;}
    [[nodiscard]] inline unsigned*& getRows() {return m_Rows;}

    unsigned* random();
    unsigned* jackard(unsigned sliceSize);
    unsigned* gorder(unsigned sliceSize);
    unsigned* jackardWithWindow(unsigned sliceSize, unsigned windowSize);
    unsigned* degreeSort();
    void applyPermutation(unsigned* inversePermutation);

private:
    unsigned m_N;
    unsigned m_NNZ;
    unsigned* m_ColPtrs;
    unsigned* m_Rows;
};

CSC::CSC(std::string filename, bool undirected, bool binary)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file from which to construct CSC.");
    }

    while (file.peek() == '%')
    {
        file.ignore(2048, '\n');
    }

    double trash;
    unsigned noLines;

    file >> m_N >> m_N >> noLines;

    std::vector<std::pair<unsigned, unsigned>> nnzs;

    unsigned i, j;
    for (unsigned iter = 0; iter < noLines; ++iter)
    {
        file >> i >> j;
        if (!binary) file >> trash;

        --i;
        --j;

        nnzs.emplace_back(i, j);

        if (undirected && i != j)
        {
            nnzs.emplace_back(j, i);
        }
    }

    file.close();

    std::cout << "Number of vertices: " << m_N << std::endl;
    std::cout << "Number of edges: " << nnzs.size() << std::endl;
    std::cout << "Sparsity: " << double(nnzs.size()) / double(m_N * m_N) << std::endl;

    std::sort(nnzs.begin(), nnzs.end(), [](const auto& a, const auto& b) 
    {
        if (a.second == b.second)
        {
            return a.first < b.first;
        }
        else
        {
            return a.second < b.second;
        }
    });

    m_NNZ = nnzs.size();
    m_ColPtrs = new unsigned[m_N + 1];
    m_Rows = new unsigned[m_NNZ];

    std::fill(m_ColPtrs, m_ColPtrs + m_N + 1, 0);

    for (unsigned iter = 0; iter < m_NNZ; ++iter)
    {
        ++m_ColPtrs[nnzs[iter].second + 1];
        m_Rows[iter] = nnzs[iter].first;
    }

    for (unsigned j = 0; j < m_N; ++j)
    {
        m_ColPtrs[j + 1] += m_ColPtrs[j];
    }
}

CSC::~CSC()
{
    delete[] m_ColPtrs;
    delete[] m_Rows;
}

unsigned* CSC::random()
{
    auto rng = [&]()
    {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    };
    auto* inversePermutation = new unsigned[m_N];
    std::vector<unsigned> perm(m_N);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng());

    for (unsigned i = 0; i < m_N; ++i)
    {
        inversePermutation[perm[i]] = i;
    }
    applyPermutation(inversePermutation);

    return inversePermutation;
}

unsigned* CSC::jackard(unsigned sliceSize)
{
    unsigned* inversePermutation = new unsigned[m_N];

    std::vector<bool> permuted(m_N, false);

    std::vector<unsigned> rowMark(m_N, 0);
    unsigned epoch = 1;
    unsigned unionSize = 0;

    auto nextEpoch = [&]()
    {
        ++epoch;
        unionSize = 0;
        if (epoch == std::numeric_limits<unsigned>::max())
        {
            std::fill(rowMark.begin(), rowMark.end(), 0);
            epoch = 1;
        }
    };
    auto isMarked = [&](unsigned r)
    {
        return rowMark[r] == epoch;
    };
    auto markRow = [&](unsigned r)
    {
        if (rowMark[r] != epoch)
        {
            rowMark[r] = epoch;
            ++unionSize;
        }
    };

    unsigned noSliceSets = (m_N + sliceSize - 1) / sliceSize;
    for (unsigned sliceSet = 0; sliceSet < noSliceSets; ++sliceSet)
    {
        nextEpoch();

        unsigned sliceStart = sliceSet * sliceSize;
        unsigned sliceEnd = std::min(sliceStart + sliceSize, m_N);

        unsigned highestDegree = 0;
        unsigned col = UNSIGNED_MAX;
        for (unsigned j = 0; j < m_N; ++j)
        {
            if (!permuted[j])
            {
                unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
                if (deg >= highestDegree)
                {
                    highestDegree = deg;
                    col = j;
                }
            }
        }
        if (col == UNSIGNED_MAX) break;

        for (unsigned nnz = m_ColPtrs[col]; nnz < m_ColPtrs[col + 1]; ++nnz)
        {
            unsigned r = m_Rows[nnz];
            markRow(r);
        }
        inversePermutation[col] = sliceStart;
        permuted[col] = true;

        for (unsigned newCol = sliceStart + 1; newCol < sliceEnd; ++newCol)
        {
            double bestJackard = -1;
            unsigned bestCol;
            #pragma omp parallel num_threads(omp_get_max_threads())
            {
                double myBest = -1;
                unsigned myCol;
                #pragma omp for schedule(static)
                for (unsigned j = 0; j < m_N; ++j)
                {
                    if (permuted[j]) continue;
    
                    unsigned inter = 0;
                    for (unsigned nnz = m_ColPtrs[j]; nnz < m_ColPtrs[j + 1]; ++nnz)
                    {
                        inter += isMarked(m_Rows[nnz]);
                    }
                    unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
                    unsigned uni = unionSize + deg - inter;
                    double currentJackard = (uni == 0) ? 1 : static_cast<double>(inter) / static_cast<double>(uni);
    
                    if (currentJackard > myBest)
                    {
                        myBest = currentJackard;
                        myCol = j;
                    }
                }
                #pragma omp critical
                {
                    if (myBest > bestJackard)
                    {
                        bestJackard = myBest;
                        bestCol = myCol;
                    }
                }
            }
            if (bestJackard == -1) break;

            for (unsigned nnz = m_ColPtrs[bestCol]; nnz < m_ColPtrs[bestCol + 1]; ++nnz)
            {
                unsigned r = m_Rows[nnz];
                markRow(r);
            }
            inversePermutation[bestCol] = newCol;
            permuted[bestCol] = true;
        }
    }
    applyPermutation(inversePermutation);

    return inversePermutation;
}

unsigned* CSC::gorder(unsigned sliceSize)
{
    unsigned windowSize = sliceSize * 8192;

	Gorder::Graph g;
	g.setFilename("gorder");
	g.readGraph(m_N, m_NNZ, m_ColPtrs, m_Rows);
	g.Transform();
	std::vector<int> order;
	g.GorderGreedy(order, windowSize);

    unsigned* inversePermutation = new unsigned[m_N];
    std::copy(order.data(), order.data() + m_N, inversePermutation);
    applyPermutation(inversePermutation);
    unsigned* inversePermutation2 = this->jackardWithWindow(sliceSize, windowSize);

    unsigned* mergedPermutation = new unsigned[m_N];
    // merge permutations
    for (unsigned i = 0; i < m_N; ++i)
    {
        mergedPermutation[i] = inversePermutation2[inversePermutation[i]];
    }
    delete[] inversePermutation;
    delete[] inversePermutation2;
    //

    return mergedPermutation;
}

unsigned* CSC::jackardWithWindow(unsigned sliceSize, unsigned windowSize)
{
    assert(windowSize % sliceSize == 0);

    unsigned* inversePermutation = new unsigned[m_N];

    unsigned noWindows = (m_N + windowSize - 1) / windowSize;
    #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
    for (unsigned window = 0; window < noWindows; ++window)
    {
        unsigned windowStart = window * windowSize;
        unsigned windowEnd = std::min(windowStart + windowSize, m_N);
        unsigned windowLength = windowEnd - windowStart;

        std::vector<bool> permuted(m_N, false);

        std::vector<unsigned> rowMark(m_N, 0);
        unsigned epoch = 1;
        unsigned unionSize = 0;
    
        auto nextEpoch = [&]()
        {
            ++epoch;
            unionSize = 0;
            if (epoch == std::numeric_limits<unsigned>::max())
            {
                std::fill(rowMark.begin(), rowMark.end(), 0);
                epoch = 1;
            }
        };
        auto isMarked = [&](unsigned r)
        {
            return rowMark[r] == epoch;
        };
        auto markRow = [&](unsigned r)
        {
            if (rowMark[r] != epoch)
            {
                rowMark[r] = epoch;
                ++unionSize;
            }
        };

        unsigned noSliceSets = (windowLength + sliceSize - 1) / sliceSize;
        for (unsigned sliceSet = 0; sliceSet < noSliceSets; ++sliceSet)
        {
            nextEpoch();
    
            unsigned sliceStart = windowStart + sliceSet * sliceSize;
            unsigned sliceEnd = std::min(sliceStart + sliceSize, windowEnd);
    
            unsigned highestDegree = 0;
            unsigned col = UNSIGNED_MAX;
            for (unsigned j = windowStart; j < windowEnd; ++j)
            {
                if (!permuted[j])
                {
                    unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
                    if (deg >= highestDegree)
                    {
                        highestDegree = deg;
                        col = j;
                    }
                }
            }
            if (col == UNSIGNED_MAX) break;
    
            for (unsigned nnz = m_ColPtrs[col]; nnz < m_ColPtrs[col + 1]; ++nnz)
            {
                unsigned r = m_Rows[nnz];
                markRow(r);
            }
            inversePermutation[col] = sliceStart;
            permuted[col] = true;
    
            for (unsigned newCol = sliceStart + 1; newCol < sliceEnd; ++newCol)
            {
                double bestJackard = -1;
                unsigned bestCol;
                for (unsigned j = windowStart; j < windowEnd; ++j)
                {
                    if (permuted[j]) continue;

                    unsigned inter = 0;
                    for (unsigned nnz = m_ColPtrs[j]; nnz < m_ColPtrs[j + 1]; ++nnz)
                    {
                        inter += isMarked(m_Rows[nnz]);
                    }
                    unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
                    unsigned uni = unionSize + deg - inter;
                    double currentJackard = (uni == 0) ? 1 : static_cast<double>(inter) / static_cast<double>(uni);

                    if (currentJackard > bestJackard)
                    {
                        bestJackard = currentJackard;
                        bestCol = j;
                    }
                }
                if (bestJackard == -1) break;
    
                for (unsigned nnz = m_ColPtrs[bestCol]; nnz < m_ColPtrs[bestCol + 1]; ++nnz)
                {
                    unsigned r = m_Rows[nnz];
                    markRow(r);
                }
                inversePermutation[bestCol] = newCol;
                permuted[bestCol] = true;
            }
        }
    }
    applyPermutation(inversePermutation);

    return inversePermutation;
}

unsigned* CSC::degreeSort()
{
    std::vector<std::pair<unsigned, unsigned>> degrees(m_N);

    for (unsigned j = 0; j < m_N; ++j)
    {
        degrees[j].first = j;
        degrees[j].second = m_ColPtrs[j + 1] - m_ColPtrs[j];
    }
    std::sort(degrees.begin(), degrees.end(),
    [](const auto& a, const auto& b)
    {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    unsigned* inversePermutation = new unsigned[m_N];

    for (unsigned j = 0; j < m_N; ++j)
    {
        inversePermutation[degrees[j].first] = j;
    }
    applyPermutation(inversePermutation);

    return inversePermutation;
}

void CSC::applyPermutation(unsigned* inversePermutation)
{
    unsigned* newColPtrs = new unsigned[m_N + 1];
    std::fill(newColPtrs, newColPtrs + m_N + 1, 0);

    for (unsigned oldCol = 0; oldCol < m_N; ++oldCol)
    {
        unsigned newCol = inversePermutation[oldCol];
        newColPtrs[newCol + 1] = m_ColPtrs[oldCol + 1] - m_ColPtrs[oldCol];
    }

    for (unsigned j = 0; j < m_N; ++j)
    {
        newColPtrs[j + 1] += newColPtrs[j];
    }

    unsigned* newRows = new unsigned[m_NNZ];

    for (unsigned oldCol = 0; oldCol < m_N; ++oldCol)
    {
        unsigned newCol = inversePermutation[oldCol];
        unsigned start = m_ColPtrs[oldCol];
        unsigned end = m_ColPtrs[oldCol + 1];

        for (unsigned nnz = start; nnz < end; ++nnz)
        {
            unsigned oldRow = m_Rows[nnz];
            unsigned newRow = inversePermutation[oldRow];
            newRows[newColPtrs[newCol] + (nnz - start)] = newRow;
        }
    }

    for (unsigned col = 0; col < m_N; ++col)
    {
        std::sort(newRows + newColPtrs[col], newRows + newColPtrs[col + 1]);
    }

    delete[] m_ColPtrs;
    delete[] m_Rows;

    m_ColPtrs = newColPtrs;
    m_Rows = newRows;
}

#endif
