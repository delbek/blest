#ifndef CSC_CUH
#define CSC_CUH

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

class CSC
{
public:
    CSC(std::string filename, bool undirected, bool binary);
    CSC(const CSC& other) = delete;
    CSC(CSC&& other) noexcept = delete;
    CSC& operator=(const CSC& other) = delete;
    CSC& operator=(CSC&& other) noexcept = delete;
    ~CSC();

    unsigned* reorderFromFile(std::string filename);
    unsigned* degreeSort();
    unsigned* hubPartition(unsigned& k);

    [[nodiscard]] inline unsigned getN() {return m_N;}
    [[nodiscard]] inline unsigned* getColPtrs() {return m_ColPtrs;}
    [[nodiscard]] inline unsigned* getRows() {return m_Rows;}

private:
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

unsigned* CSC::reorderFromFile(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) 
    {
        std::cout << "Failed to open file from which to load reordering. Proceeding with natural ordering." << std::endl;
        return nullptr;
    }

    unsigned* inversePermutation = new unsigned[m_N];
    file.read(reinterpret_cast<char*>(inversePermutation), sizeof(unsigned) * m_N);
    file.close();

    unsigned* newColPtrs = new unsigned[m_N + 1];
    unsigned* newRows = new unsigned[m_NNZ];

    std::vector<std::pair<unsigned, unsigned>> nnzs;
    nnzs.reserve(m_NNZ);
    for (unsigned j = 0; j < m_N; ++j)
    {
        unsigned newCol = inversePermutation[j];
        for (unsigned ptr = m_ColPtrs[j]; ptr < m_ColPtrs[j + 1]; ++ptr)
        {
            unsigned newRow = inversePermutation[m_Rows[ptr]];
            nnzs.emplace_back(newRow, newCol);
        }
    }
    
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

    std::fill(newColPtrs, newColPtrs + m_N + 1, 0);

    for (unsigned iter = 0; iter < m_NNZ; ++iter)
    {
        ++newColPtrs[nnzs[iter].second + 1];
        newRows[iter] = nnzs[iter].first;
    }

    for (unsigned j = 0; j < m_N; ++j)
    {
        newColPtrs[j + 1] += newColPtrs[j];
    }

    delete[] m_ColPtrs;
    delete[] m_Rows;
    m_ColPtrs = newColPtrs;
    m_Rows = newRows;

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

unsigned* CSC::hubPartition(unsigned& k)
{
    unsigned* inversePermutation = this->degreeSort();

    k = m_N;
    for (unsigned j = 0; j < m_N; ++j)
    {
        if (m_ColPtrs[j + 1] - m_ColPtrs[j] < 512)
        {
            k = j;
            break;
        }
    }

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
