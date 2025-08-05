#ifndef CSR_CUH
#define CSR_CUH

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

class CSR
{
public:
    CSR(std::string filename, bool undirected, bool binary);
    CSR(const CSR& other) = delete;
    CSR(CSR&& other) noexcept = delete;
    CSR& operator=(const CSR& other) = delete;
    CSR& operator=(CSR&& other) noexcept = delete;
    ~CSR();
    
    unsigned* reorderFromFile(std::string filename);
    unsigned* degreeSort();
    void applyPermutation(unsigned* inversePermutation);

    [[nodiscard]] inline unsigned getN() {return m_N;}
    [[nodiscard]] inline unsigned* getRowPtrs() {return m_RowPtrs;}
    [[nodiscard]] inline unsigned* getCols() {return m_Cols;}

private:
    unsigned m_N;
    unsigned m_NNZ;
    unsigned* m_RowPtrs;
    unsigned* m_Cols;
};

CSR::CSR(std::string filename, bool undirected, bool binary)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file from which to construct CSR.");
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
        if (a.first == b.first)
        {
            return a.second < b.second;
        }
        else
        {
            return a.first < b.first;
        }
    });

    m_NNZ = nnzs.size();
    m_RowPtrs = new unsigned[m_N + 1];
    m_Cols = new unsigned[m_NNZ];

    std::fill(m_RowPtrs, m_RowPtrs + m_N + 1, 0);

    for (unsigned iter = 0; iter < m_NNZ; ++iter)
    {
        ++m_RowPtrs[nnzs[iter].first + 1];
        m_Cols[iter] = nnzs[iter].second;
    }

    for (unsigned j = 0; j < m_N; ++j)
    {
        m_RowPtrs[j + 1] += m_RowPtrs[j];
    }
}

unsigned* CSR::reorderFromFile(std::string filename)
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

    applyPermutation(inversePermutation);

    return inversePermutation;
}

unsigned* CSR::degreeSort()
{
    unsigned* inversePermutation = new unsigned[m_N];

    std::vector<std::pair<unsigned, unsigned>> frequencies(m_N);
    for (unsigned i = 0; i < m_N; ++i)
    {
        frequencies[i].first = i;
        frequencies[i].second = m_RowPtrs[i + 1] - m_RowPtrs[i];
    }
    std::sort(frequencies.begin(), frequencies.end(), [](const auto& a, const auto& b) 
    {
        return a.second < b.second;
    });

    for (unsigned i = 0; i < m_N; ++i)
    {
        inversePermutation[frequencies[i].first] = i;
    }

    applyPermutation(inversePermutation);

    return inversePermutation;
}

void CSR::applyPermutation(unsigned* inversePermutation)
{
    unsigned* newPtrs = new unsigned[m_N + 1];
    std::fill(newPtrs, newPtrs + m_N + 1, 0);
    unsigned* newCols = new unsigned[m_NNZ];

    for (unsigned i = 0; i < m_N; ++i)
    {
        unsigned newRow = inversePermutation[i];
        newPtrs[newRow + 1] = m_RowPtrs[i + 1] - m_RowPtrs[i];
    }
    
    for (unsigned i = 0; i < m_N; ++i)
    {
        newPtrs[i + 1] += newPtrs[i];
    }

    for (unsigned i = 0; i < m_N; ++i)
    {
        unsigned newRow = inversePermutation[i];
        for (unsigned idx = 0; idx < (m_RowPtrs[i + 1] - m_RowPtrs[i]); ++idx)
        {
            newCols[newPtrs[newRow] + idx] = inversePermutation[m_Cols[m_RowPtrs[i] + idx]];
        }
    }

    delete[] m_RowPtrs;
    delete[] m_Cols;

    m_RowPtrs = newPtrs;
    m_Cols = newCols;
}

CSR::~CSR()
{
    delete[] m_RowPtrs;
    delete[] m_Cols;
}

#endif
