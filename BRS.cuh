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
    BRS(unsigned sliceSize = 32);
    BRS(const BRS& other);
    BRS(BRS&& other) noexcept;
    BRS& operator=(const BRS& other);
    BRS& operator=(BRS&& other) noexcept;
    ~BRS();

    void readFromCSCMatrix(CSC* csc);
    void printBRSData();

private:
    unsigned m_N;
    unsigned m_SliceSize;
    unsigned m_NoSliceSets;

    unsigned* m_SliceSetPtrs;
    unsigned* m_RowIds;
    MASK* m_Masks;
};

BRS::BRS(unsigned sliceSize)
: BitMatrix(),
  m_SliceSize(sliceSize)
{

}

BRS::BRS(const BRS& other)
: BitMatrix(other),
  m_N(other.m_N),
  m_SliceSize(other.m_SliceSize),
  m_NoSliceSets(other.m_NoSliceSets)
{
    unsigned m_NoSliceSets = std::ceil(m_N / K);
    unsigned noSlices = K / m_SliceSize;
    unsigned totalSlots = (other.m_SliceSetPtrs[m_NoSliceSets]) * noSlices;

    m_SliceSetPtrs = new unsigned[m_NoSliceSets + 1];
    std::copy(other.m_SliceSetPtrs, other.m_SliceSetPtrs + m_NoSliceSets + 1, m_SliceSetPtrs);

    m_RowIds = new unsigned[totalSlots];
    std::copy(other.m_RowIds, other.m_RowIds + totalSlots, m_RowIds);

    m_Masks = new MASK[totalSlots];
    std::copy(other.m_Masks, other.m_Masks + totalSlots, m_Masks);
}

BRS::BRS(BRS&& other) noexcept
: BitMatrix(std::move(other)),
  m_N(other.m_N),
  m_SliceSize(other.m_SliceSize),
  m_NoSliceSets(other.m_NoSliceSets),
  m_SliceSetPtrs(other.m_SliceSetPtrs),
  m_RowIds(other.m_RowIds),
  m_Masks(other.m_Masks)
{
    other.m_SliceSetPtrs = nullptr;
    other.m_RowIds = nullptr;
    other.m_Masks = nullptr;
    other.m_N = 0;
    other.m_SliceSize = 0;
}

BRS& BRS::operator=(const BRS& other)
{
    if (this != &other)
    {
        BitMatrix::operator=(other);
        delete[] m_SliceSetPtrs;
        delete[] m_RowIds;
        delete[] m_Masks;

        m_N = other.m_N;
        m_SliceSize = other.m_SliceSize;
        m_NoSliceSets = other.m_NoSliceSets;

        unsigned m_NoSliceSets = std::ceil(m_N / K);
        unsigned noSlices = K / m_SliceSize;
        unsigned totalSlots = (other.m_SliceSetPtrs[m_NoSliceSets]) * noSlices;

        m_SliceSetPtrs = new unsigned[m_NoSliceSets + 1];
        std::copy(other.m_SliceSetPtrs, other.m_SliceSetPtrs + m_NoSliceSets + 1, m_SliceSetPtrs);

        m_RowIds = new unsigned[totalSlots];
        std::copy(other.m_RowIds, other.m_RowIds + totalSlots, m_RowIds);

        m_Masks = new MASK[totalSlots];
        std::copy(other.m_Masks, other.m_Masks + totalSlots, m_Masks);
    }
    return *this;
}

BRS& BRS::operator=(BRS&& other) noexcept
{
    if (this != &other)
    {
        BitMatrix::operator=(std::move(other));
        delete[] m_SliceSetPtrs;
        delete[] m_RowIds;
        delete[] m_Masks;

        m_N = other.m_N;
        m_SliceSize = other.m_SliceSize;
        m_NoSliceSets = other.m_NoSliceSets;
        m_SliceSetPtrs = other.m_SliceSetPtrs;
        m_RowIds = other.m_RowIds;
        m_Masks = other.m_Masks;

        other.m_SliceSetPtrs = nullptr;
        other.m_RowIds = nullptr;
        other.m_Masks = nullptr;
        other.m_N = 0;
        other.m_SliceSize = 0;
    }
    return *this;
}

BRS::~BRS()
{
    delete[] m_SliceSetPtrs;
    delete[] m_RowIds;
    delete[] m_Masks;
}

void BRS::readFromCSCMatrix(CSC* csc)
{
    m_N = csc->getN();
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();

    m_NoSliceSets = std::ceil(m_N / K);
    m_SliceSetPtrs = new unsigned[m_NoSliceSets + 1];
    std::fill(m_SliceSetPtrs, m_SliceSetPtrs + m_NoSliceSets + 1, 0);

    if (K % m_SliceSize != 0)
    {
        throw std::runtime_error("Tensor core instruction width, K, must be a multiple of the selected slice size.");
    }
    unsigned noSlices = K / m_SliceSize;

    std::vector<unsigned> sliceSetLengths(m_NoSliceSets, 0);
    std::vector<std::vector<std::vector<std::pair<unsigned, MASK>>>> sliceSetMasks(m_NoSliceSets, std::vector<std::vector<std::pair<unsigned, MASK>>>(noSlices));

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
                ptrs[j - sliceStart] = colPtrs[j]; // do note that for this approach to work out, adjancecy list must be sorted
            }

            for (unsigned i = 0; i < m_N; ++i)
            {
                MASK mask = 0;
                for (unsigned j = sliceStart; j < sliceEnd; ++j)
                {
                    while (rows[ptrs[j - sliceStart]] < i && ptrs[j - sliceStart] < colPtrs[j + 1])
                    {
                        ++ptrs[j - sliceStart];
                    }
                    if (rows[ptrs[j - sliceStart]] == i && ptrs[j - sliceStart] < colPtrs[j + 1])
                    {
                        mask |= static_cast<MASK>(1) << ((MASK_BITS - 1) - (j - sliceStart));
                    }
                }
                if (mask != 0)
                {
                    sliceSetMasks[sliceSet][slice].emplace_back(i, mask);
                }
            }
            sliceSetLengths[sliceSet] = std::max(sliceSetLengths[sliceSet], unsigned(sliceSetMasks[sliceSet][slice].size()));
        }
    }

    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet) 
    {
        m_SliceSetPtrs[sliceSet + 1] = m_SliceSetPtrs[sliceSet] + sliceSetLengths[sliceSet];
    }
    m_RowIds = new unsigned[m_SliceSetPtrs[m_NoSliceSets] * noSlices];
    m_Masks = new MASK[m_SliceSetPtrs[m_NoSliceSets] * noSlices];

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        for (unsigned i = 0; i < sliceSetLengths[sliceSet]; ++i)
        {
            for (unsigned slice = 0; slice < noSlices; ++slice)
            {
                if (i < sliceSetMasks[sliceSet][slice].size())
                {
                    m_RowIds[(m_SliceSetPtrs[sliceSet] + i) * noSlices + slice] = sliceSetMasks[sliceSet][slice][i].first;
                    m_Masks[(m_SliceSetPtrs[sliceSet] + i) * noSlices + slice] = sliceSetMasks[sliceSet][slice][i].second;
                }
                else
                {
                    m_RowIds[(m_SliceSetPtrs[sliceSet] + i) * noSlices + slice] = UNSIGNED_MAX;
                    m_Masks[(m_SliceSetPtrs[sliceSet] + i) * noSlices + slice] = 0;
                }
            }
        }
    }
}

void BRS::printBRSData()
{
    unsigned noSlices = K / m_SliceSize;

    std::cout << "Number of slice sets: " << m_NoSliceSets << std::endl;
    std::cout << "Number of slices in each set: " << noSlices << std::endl;

    double averageSliceSetLength = 0;
    for (unsigned ss = 0; ss < m_NoSliceSets; ++ss)
    {
        averageSliceSetLength += (m_SliceSetPtrs[ss + 1] - m_SliceSetPtrs[ss]);
    }
    averageSliceSetLength /= m_NoSliceSets;
    std::cout << "Average slice set length: " << averageSliceSetLength << std::endl;

    unsigned noSetBits = 0;
    for (unsigned i = 0; i < m_SliceSetPtrs[m_NoSliceSets]; ++i)
    {
        for (unsigned j = 0; j < noSlices; ++j)
        {
            noSetBits += __builtin_popcount(m_Masks[i * noSlices + j]);
        }
    }
    double maskCompressionRatio = noSetBits;
    maskCompressionRatio /= (m_SliceSetPtrs[m_NoSliceSets] * noSlices * 32);
    std::cout << "Mask compression ratio: " << maskCompressionRatio << std::endl;
}

#endif
