#ifndef CSC_CUH
#define CSC_CUH

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

class CSC
{
public:
    CSC(std::string filename, bool undirected, bool binary);
    CSC(const CSC& other);
    CSC(CSC&& other) noexcept;
    CSC& operator=(const CSC& other);
    CSC& operator=(CSC&& other) noexcept;
    ~CSC();

    [[nodiscard]] inline unsigned getN() {return m_N;}
    [[nodiscard]] inline unsigned* getColPtrs() {return m_ColPtrs;}
    [[nodiscard]] inline unsigned* getRows() {return m_Rows;}

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
        throw std::runtime_error("Failed to open file");
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

CSC::CSC(const CSC& other)
    : m_N(other.m_N), m_NNZ(other.m_NNZ)
{
    m_ColPtrs = new unsigned[m_N + 1];
    std::copy(other.m_ColPtrs, other.m_ColPtrs + m_N + 1, m_ColPtrs);

    m_Rows = new unsigned[m_NNZ];
    std::copy(other.m_Rows, other.m_Rows + m_NNZ, m_Rows);
}

CSC::CSC(CSC&& other) noexcept
    : m_N(other.m_N), m_NNZ(other.m_NNZ), m_ColPtrs(other.m_ColPtrs), m_Rows(other.m_Rows)
{
    other.m_ColPtrs = nullptr;
    other.m_Rows = nullptr;
    other.m_N = 0;
    other.m_NNZ = 0;
}

CSC& CSC::operator=(const CSC& other)
{
    if (this != &other)
    {
        delete[] m_ColPtrs;
        delete[] m_Rows;

        m_N = other.m_N;
        m_NNZ = other.m_NNZ;

        m_ColPtrs = new unsigned[m_N + 1];
        std::copy(other.m_ColPtrs, other.m_ColPtrs + m_N + 1, m_ColPtrs);

        m_Rows = new unsigned[m_NNZ];
        std::copy(other.m_Rows, other.m_Rows + m_NNZ, m_Rows);
    }
    return *this;
}

CSC& CSC::operator=(CSC&& other) noexcept
{
    if (this != &other)
    {
        delete[] m_ColPtrs;
        delete[] m_Rows;

        m_N = other.m_N;
        m_NNZ = other.m_NNZ;
        m_ColPtrs = other.m_ColPtrs;
        m_Rows = other.m_Rows;

        other.m_ColPtrs = nullptr;
        other.m_Rows = nullptr;
        other.m_N = 0;
        other.m_NNZ = 0;
    }
    return *this;
}

CSC::~CSC()
{
    delete[] m_ColPtrs;
    delete[] m_Rows;
}

#endif
