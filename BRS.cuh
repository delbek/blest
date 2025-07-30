#ifndef BRS_CUH
#define BRS_CUH

#include "BitMatrix.cuh"
#include "CSC.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>

class BRS: public BitMatrix
{
public:
    BRS(std::string filename);
    BRS(unsigned sliceSize = 32);
    BRS(const BRS& other) = delete;
    BRS(BRS&& other) noexcept = delete;
    BRS& operator=(const BRS& other) = delete;
    BRS& operator=(BRS&& other) noexcept = delete;
    virtual ~BRS();

    virtual void save(std::string filename) final;
    void constructFromCSCMatrix(CSC* csc);
    void printBRSData();

    [[nodiscard]] inline unsigned getN() {return m_N;}
    [[nodiscard]] inline unsigned getSliceSize() {return m_SliceSize;}
    [[nodiscard]] inline unsigned getNoSliceSets() {return m_NoSliceSets;}
    [[nodiscard]] inline unsigned* getSliceSetPtrs() {return m_SliceSetPtrs;}
    [[nodiscard]] inline unsigned* getRowIds() {return m_RowIds;}
    [[nodiscard]] inline MASK* getMasks() {return m_Masks;}

private:
    unsigned m_N;
    unsigned m_SliceSize;
    unsigned m_NoSliceSets;

    unsigned* m_SliceSetPtrs;
    unsigned* m_RowIds;
    MASK* m_Masks;
};

BRS::BRS(std::string filename)
: BitMatrix()
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) 
    {
        throw std::runtime_error("Failed to open file from which to load BRS.");
    }

    // metadata
    file.read(reinterpret_cast<char*>(&m_N), sizeof(unsigned));
    file.read(reinterpret_cast<char*>(&m_SliceSize), sizeof(unsigned));
    file.read(reinterpret_cast<char*>(&m_NoSliceSets), sizeof(unsigned));

    // arrays
    m_SliceSetPtrs = new unsigned[m_NoSliceSets + 1];
    file.read(reinterpret_cast<char*>(m_SliceSetPtrs), sizeof(unsigned) * (m_NoSliceSets + 1));
    m_RowIds = new unsigned[m_SliceSetPtrs[m_NoSliceSets]];
    file.read(reinterpret_cast<char*>(m_RowIds), sizeof(unsigned) * m_SliceSetPtrs[m_NoSliceSets]);
    unsigned noMasks = m_SliceSize / MASK_BITS;
    m_Masks = new MASK[m_SliceSetPtrs[m_NoSliceSets] * noMasks];
    file.read(reinterpret_cast<char*>(m_Masks), sizeof(MASK) * m_SliceSetPtrs[m_NoSliceSets] * noMasks);

    file.close();
}

BRS::BRS(unsigned sliceSize)
: BitMatrix(),
  m_SliceSize(sliceSize)
{

}

BRS::~BRS()
{
    delete[] m_SliceSetPtrs;
    delete[] m_RowIds;
    delete[] m_Masks;
}

void BRS::save(std::string filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) 
    {
        throw std::runtime_error("Failed to open file in which to save BRS.");
    }
    
    // metadata
    file.write(reinterpret_cast<const char*>(&m_N), sizeof(unsigned));
    file.write(reinterpret_cast<const char*>(&m_SliceSize), sizeof(unsigned));
    file.write(reinterpret_cast<const char*>(&m_NoSliceSets), sizeof(unsigned));

    // arrays
    file.write(reinterpret_cast<const char*>(m_SliceSetPtrs), sizeof(unsigned) * (m_NoSliceSets + 1));
    file.write(reinterpret_cast<const char*>(m_RowIds), sizeof(unsigned) * m_SliceSetPtrs[m_NoSliceSets]);
    unsigned noMasks = m_SliceSize / MASK_BITS;
    file.write(reinterpret_cast<const char*>(m_Masks), sizeof(MASK) * m_SliceSetPtrs[m_NoSliceSets] * noMasks);

    file.close();
}

void BRS::constructFromCSCMatrix(CSC* csc)
{
    m_N = csc->getN();
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();

    m_NoSliceSets = (m_N + K - 1) / K;
    m_SliceSetPtrs = new unsigned[m_NoSliceSets + 1];
    std::fill(m_SliceSetPtrs, m_SliceSetPtrs + m_NoSliceSets + 1, 0);

    if (K % m_SliceSize != 0)
    {
        throw std::runtime_error("Tensor core instruction width, K, must be a multiple of the selected slice size.");
    }
    if (m_SliceSize < MASK_BITS)
    {
        throw std::runtime_error("Minimum supported slice size is equal to 32.");
    }
    unsigned noSlices = K / m_SliceSize;
    unsigned noMasks = m_SliceSize / MASK_BITS;

    std::vector<unsigned> sliceSetLengths(m_NoSliceSets, 0);
    std::vector<std::vector<std::vector<std::pair<unsigned, std::vector<MASK>>>>> sliceSetMasks(m_NoSliceSets, std::vector<std::vector<std::pair<unsigned, std::vector<MASK>>>>(noSlices));

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        unsigned sliceSetStart = sliceSet * K;
        unsigned sliceSetEnd = std::min(m_N, sliceSetStart + K);

        for (unsigned slice = 0; slice < noSlices; ++slice)
        {
            unsigned sliceStart = sliceSetStart + slice * m_SliceSize;
            unsigned sliceEnd = std::min(sliceSetEnd, sliceStart + m_SliceSize);

            std::vector<unsigned> ptrs(sliceEnd - sliceStart);
            for (unsigned j = sliceStart; j < sliceEnd; ++j)
            {
                ptrs[j - sliceStart] = colPtrs[j]; // do note that for this approach to work out, adjacency list must be sorted
            }

            for (unsigned i = 0; i < m_N; ++i)
            {
                std::vector<MASK> masks(noMasks, 0);
                for (unsigned j = sliceStart; j < sliceEnd; ++j)
                {
                    unsigned idx = j - sliceStart;
                    unsigned mask = idx / MASK_BITS;
                    unsigned bit = idx % MASK_BITS;
                    while (rows[ptrs[idx]] < i && ptrs[idx] < colPtrs[j + 1])
                    {
                        ++ptrs[idx];
                    }
                    if (rows[ptrs[idx]] == i && ptrs[idx] < colPtrs[j + 1])
                    {
                        masks[mask] |= (static_cast<MASK>(1) << (bit));
                    }
                }
                for (const auto& mask: masks)
                {
                    if (mask != 0)
                    {
                        sliceSetMasks[sliceSet][slice].emplace_back(i, masks);
                        break;
                    }
                }
            }
            sliceSetLengths[sliceSet] = std::max(sliceSetLengths[sliceSet], unsigned(sliceSetMasks[sliceSet][slice].size()));
        }
    }

    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet) 
    {
        m_SliceSetPtrs[sliceSet + 1] = m_SliceSetPtrs[sliceSet] + sliceSetLengths[sliceSet] * noSlices;
    }
    m_RowIds = new unsigned[m_SliceSetPtrs[m_NoSliceSets]];
    m_Masks = new MASK[m_SliceSetPtrs[m_NoSliceSets] * noMasks];

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        for (unsigned i = 0; i < sliceSetLengths[sliceSet]; ++i)
        {
            for (unsigned slice = 0; slice < noSlices; ++slice)
            {
                if (i < sliceSetMasks[sliceSet][slice].size())
                {
                    m_RowIds[m_SliceSetPtrs[sliceSet] + (i * noSlices) + slice] = sliceSetMasks[sliceSet][slice][i].first;
                    for (unsigned j = 0; j < noMasks; ++j)
                    {
                        m_Masks[(m_SliceSetPtrs[sliceSet] + (i * noSlices) + slice) * noMasks + j] = sliceSetMasks[sliceSet][slice][i].second[j];
                    }
                }
                else
                {
                    m_RowIds[m_SliceSetPtrs[sliceSet] + (i * noSlices) + slice] = UNSIGNED_MAX;
                    for (unsigned j = 0; j < noMasks; ++j)
                    {
                        m_Masks[(m_SliceSetPtrs[sliceSet] + (i * noSlices) + slice) * noMasks + j] = 0;
                    }
                }
            }
        }
    }
}

void BRS::printBRSData()
{
    unsigned noSlices = K / m_SliceSize;
    unsigned noMasks = m_SliceSize / MASK_BITS;

    std::cout << "MASK size: " << MASK_BITS << std::endl;
    std::cout << "Slice size: " << m_SliceSize << std::endl;
    std::cout << "Number of slice sets: " << m_NoSliceSets << std::endl;
    std::cout << "Number of slices in each set row: " << noSlices << std::endl;
    std::cout << "Number of masks in each slice: " << noMasks << std::endl;

    double average = 0;
    for (unsigned ss = 0; ss < m_NoSliceSets; ++ss)
    {
        average += (m_SliceSetPtrs[ss + 1] - m_SliceSetPtrs[ss]);
    }
    average /= m_NoSliceSets;
    std::cout << "Average number of slices in each set " << average << std::endl;

    double variance = 0;
    for (unsigned ss = 0; ss < m_NoSliceSets; ++ss)
    {
        unsigned length = (m_SliceSetPtrs[ss + 1] - m_SliceSetPtrs[ss]);
        double diff = length - average;
        variance += diff * diff;
    }
    variance /= m_NoSliceSets;
    double standardDeviation = std::sqrt(variance);
    std::cout << "Standard deviation of the number of slices in each set: " << standardDeviation << std::endl;

    unsigned noSetBits = 0;
    for (unsigned i = 0; i < m_SliceSetPtrs[m_NoSliceSets]; ++i)
    {
        for (unsigned j = 0; j < noMasks; ++j)
        {
            noSetBits += __builtin_popcount(m_Masks[i * noMasks + j]);
        }
    }
    std::cout << "Set bits: " << noSetBits << std::endl;
    double maskCompressionRatio = noSetBits;
    maskCompressionRatio /= (m_SliceSetPtrs[m_NoSliceSets] * noMasks * MASK_BITS);
    std::cout << "Mask compression ratio: " << maskCompressionRatio << std::endl;
}

#endif
