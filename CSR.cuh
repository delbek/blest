#ifndef CSR_CUH
#define CSR_CUH

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

class CSR
{
public:
    CSR(std::string filename, bool undirected, bool binary);
    CSR(const CSR& other) = delete;
    CSR(CSR&& other) noexcept = delete;
    CSR& operator=(const CSR& other) = delete;
    CSR& operator=(CSR&& other) noexcept = delete;
    ~CSR();

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

CSR::~CSR()
{
    delete[] m_RowPtrs;
    delete[] m_Cols;
}

#endif
