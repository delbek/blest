#ifndef BCS_CUH
#define BCS_CUH

#include "BitMatrix.cuh"
#include "CSR.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>

class BCS: public BitMatrix
{
public:
    BCS(std::string filename);
    BCS(unsigned sliceSize = 32);
    BCS(const BCS& other) = delete;
    BCS(BCS&& other) noexcept = delete;
    BCS& operator=(const BCS& other) = delete;
    BCS& operator=(BCS&& other) noexcept = delete;
    virtual ~BCS();

    virtual void save(std::string filename) final;
    void constructFromCSRMatrix(CSR* csr);
    void printBCSData();

    [[nodiscard]] inline unsigned getN() {return m_N;}
    [[nodiscard]] inline unsigned getSliceSize() {return m_SliceSize;}
    [[nodiscard]] inline unsigned getNoSliceSets() {return m_NoSliceSets;}
    [[nodiscard]] inline unsigned* getSliceSetPtrs() {return m_SliceSetPtrs;}
    [[nodiscard]] inline unsigned* getColIds() {return m_ColIds;}
    [[nodiscard]] inline MASK* getMasks() {return m_Masks;}

private:
    unsigned m_N;
    unsigned m_SliceSize;
    unsigned m_NoSliceSets;

    unsigned* m_SliceSetPtrs;
    unsigned* m_ColIds;
    MASK* m_Masks;
};

BCS::BCS(std::string filename)
: BitMatrix()
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) 
    {
        throw std::runtime_error("Failed to open file from which to load BCS.");
    }

    // metadata
    file.read(reinterpret_cast<char*>(&m_N), sizeof(unsigned));
    file.read(reinterpret_cast<char*>(&m_SliceSize), sizeof(unsigned));
    file.read(reinterpret_cast<char*>(&m_NoSliceSets), sizeof(unsigned));

    // arrays
    m_SliceSetPtrs = new unsigned[m_NoSliceSets + 1];
    file.read(reinterpret_cast<char*>(m_SliceSetPtrs), sizeof(unsigned) * (m_NoSliceSets + 1));
    m_ColIds = new unsigned[m_SliceSetPtrs[m_NoSliceSets]];
    file.read(reinterpret_cast<char*>(m_ColIds), sizeof(unsigned) * m_SliceSetPtrs[m_NoSliceSets]);
    unsigned noMasks = MASK_BITS / m_SliceSize;
    m_Masks = new MASK[(m_SliceSetPtrs[m_NoSliceSets] / noMasks)];
    file.read(reinterpret_cast<char*>(m_Masks), sizeof(MASK) * (m_SliceSetPtrs[m_NoSliceSets] / noMasks));

    file.close();
}

BCS::BCS(unsigned sliceSize)
: BitMatrix(),
  m_SliceSize(sliceSize)
{

}

BCS::~BCS()
{
    delete[] m_SliceSetPtrs;
    delete[] m_ColIds;
    delete[] m_Masks;
}

void BCS::save(std::string filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) 
    {
        throw std::runtime_error("Failed to open file in which to save BCS.");
    }
    
    // metadata
    file.write(reinterpret_cast<const char*>(&m_N), sizeof(unsigned));
    file.write(reinterpret_cast<const char*>(&m_SliceSize), sizeof(unsigned));
    file.write(reinterpret_cast<const char*>(&m_NoSliceSets), sizeof(unsigned));

    // arrays
    file.write(reinterpret_cast<const char*>(m_SliceSetPtrs), sizeof(unsigned) * (m_NoSliceSets + 1));
    file.write(reinterpret_cast<const char*>(m_ColIds), sizeof(unsigned) * m_SliceSetPtrs[m_NoSliceSets]);
    unsigned noMasks = MASK_BITS / m_SliceSize;
    file.write(reinterpret_cast<const char*>(m_Masks), sizeof(MASK) * (m_SliceSetPtrs[m_NoSliceSets] / noMasks));

    file.close();
}

void BCS::constructFromCSRMatrix(CSR* csr)
{
    m_N = csr->getN();
    unsigned* rowPtrs = csr->getRowPtrs();
    unsigned* cols = csr->getCols();

    m_NoSliceSets = (m_N + M - 1) / M;
    m_SliceSetPtrs = new unsigned[m_NoSliceSets + 1];
    std::fill(m_SliceSetPtrs, m_SliceSetPtrs + m_NoSliceSets + 1, 0);

    if (m_SliceSize > MASK_BITS || K % m_SliceSize != 0)
    {
        throw std::runtime_error("Invalid slice size provided.");
    }
    unsigned noMasks = MASK_BITS / m_SliceSize;
    unsigned noSlices = K / m_SliceSize;

    std::vector<std::vector<unsigned>> colIds(m_NoSliceSets);
    std::vector<std::vector<MASK>> masks(m_NoSliceSets);

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        unsigned sliceSetRowStart = sliceSet * M;
        unsigned sliceSetRowEnd = std::min(m_N, sliceSetRowStart + M);

        std::vector<std::vector<unsigned>> colIdsPartial(m_SliceSize);
        std::vector<std::vector<MASK>> masksPartial(m_SliceSize);
        unsigned longest = noSlices;

        for (unsigned i = sliceSetRowStart; i < sliceSetRowEnd; ++i)
        {
            unsigned idx = i - sliceSetRowStart;
            MASK mask = 0;
            int counter = -1;
            bool first = true;
            unsigned prev = 0;
            for (unsigned ptr = rowPtrs[i]; ptr < rowPtrs[i + 1]; ++ptr)
            {
                unsigned col = cols[ptr];
                if (first || col - prev >= m_SliceSize)
                {
                    if (counter == (noMasks - 1))
                    {
                        masksPartial[idx].emplace_back(mask);
                        counter = -1;
                        mask = 0;
                    }
                    
                    prev = col;
                    while (prev % m_SliceSize != 0)
                    {
                        --prev;
                    }
                    colIdsPartial[idx].emplace_back(prev);
                    ++counter;
                    first = false;
                }

                mask |= (static_cast<MASK>(1) << (m_SliceSize * counter + (col - prev)));
            }
            if (mask != 0)
            {
                while (colIdsPartial[idx].size() % noMasks)
                {
                    colIdsPartial[idx].emplace_back(0);
                }
                masksPartial[idx].emplace_back(mask);
            }
            longest = std::max(longest, unsigned(colIdsPartial[idx].size()));
            assert(colIdsPartial[idx].size() == masksPartial[idx].size() * noMasks);
        }

        for (unsigned mma = 0; mma < longest; mma += noSlices)
        {
            for (unsigned i = sliceSetRowStart; i < sliceSetRowStart + M; ++i)
            {
                unsigned idx = i - sliceSetRowStart;
                for (unsigned iter = mma; iter < (mma + noSlices); ++iter)
                {
                    if (iter < colIdsPartial[idx].size())
                    {
                        colIds[sliceSet].emplace_back(colIdsPartial[idx][iter]);
                        if (iter % noMasks == 0)
                        {
                            masks[sliceSet].emplace_back(masksPartial[idx][iter / noMasks]);
                        }
                    }
                    else
                    {
                        colIds[sliceSet].emplace_back(0);
                        if (iter % noMasks == 0)
                        {
                            masks[sliceSet].emplace_back(0);
                        }
                    }
                }
            }
        }
    }

    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet) 
    {
        m_SliceSetPtrs[sliceSet + 1] = m_SliceSetPtrs[sliceSet] + colIds[sliceSet].size();
    }

    m_ColIds = new unsigned[m_SliceSetPtrs[m_NoSliceSets]];
    unsigned idx = 0;
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        for (unsigned i = 0; i < colIds[sliceSet].size(); ++i)
        {
            m_ColIds[idx++] = colIds[sliceSet][i];
        }
    }

    m_Masks = new MASK[(m_SliceSetPtrs[m_NoSliceSets] / noMasks)];
    idx = 0;
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        for (unsigned i = 0; i < masks[sliceSet].size(); ++i)
        {
            m_Masks[idx++] = masks[sliceSet][i];
        }
    }
}

void BCS::printBCSData()
{
    unsigned noMasks = MASK_BITS / m_SliceSize;
    unsigned noSlices = K / m_SliceSize;

    std::cout << "MASK size: " << MASK_BITS << std::endl;
    std::cout << "Slice size: " << m_SliceSize << std::endl;
    std::cout << "Number of slice sets: " << m_NoSliceSets << std::endl;
    std::cout << "Number of slices: " << m_SliceSetPtrs[m_NoSliceSets] << std::endl;
    std::cout << "Number of slices in each MMA row: " << noSlices << std::endl;
    std::cout << "Number of slices in each mask: " << noMasks << std::endl;

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
    unsigned total = (m_SliceSetPtrs[m_NoSliceSets] / noMasks);
    for (unsigned i = 0; i < total; ++i)
    {
        noSetBits += __builtin_popcount(m_Masks[i]);
    }
    double compressionEff = noSetBits;
    compressionEff /= (total * MASK_BITS);
    std::cout << "Total - Set Bits: " << noSetBits << " - Compression Efficiency: " << compressionEff << std::endl;
}

#endif
