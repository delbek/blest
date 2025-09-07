#ifndef BRS_CUH
#define BRS_CUH

#include "BitMatrix.cuh"
#include "CSC.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <unordered_map>

class BRS: public BitMatrix
{
public:
    BRS(unsigned sliceSize = 32);
    BRS(const BRS& other) = delete;
    BRS(BRS&& other) noexcept = delete;
    BRS& operator=(const BRS& other) = delete;
    BRS& operator=(BRS&& other) noexcept = delete;
    virtual ~BRS();

    void constructFromCSCMatrix(CSC* csc);
    void printBRSData();
    void coalescingTest();
    void pattern8Test();

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

void BRS::constructFromCSCMatrix(CSC* csc)
{
    m_N = csc->getN();
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();

    m_NoSliceSets = (m_N + m_SliceSize - 1) / m_SliceSize;
    m_SliceSetPtrs = new unsigned[m_NoSliceSets + 1];
    std::fill(m_SliceSetPtrs, m_SliceSetPtrs + m_NoSliceSets + 1, 0);

    if (m_SliceSize > MASK_BITS || K % m_SliceSize != 0)
    {
        throw std::runtime_error("Invalid slice size provided.");
    }
    unsigned noMasks = MASK_BITS / m_SliceSize;
    unsigned noWarpSlice = (K / m_SliceSize) * 8;

    std::vector<std::vector<unsigned>> rowIds(m_NoSliceSets);
    std::vector<std::vector<MASK>> masks(m_NoSliceSets);

    unsigned noEmptySliceSets = 0;
    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        unsigned sliceSetColStart = sliceSet * m_SliceSize;
        unsigned sliceSetColEnd = std::min(m_N, sliceSetColStart + m_SliceSize);
        
        std::vector<unsigned> ptrs(sliceSetColEnd - sliceSetColStart);
        for (unsigned j = sliceSetColStart; j < sliceSetColEnd; ++j)
        {
            ptrs[j - sliceSetColStart] = colPtrs[j]; // do note that for this approach to work out, adjacency list must be sorted
        }

        std::vector<unsigned> tempRowIds;
        std::vector<MASK> tempMasks;

        unsigned i = 0;
        while (i < m_N) 
        {
            MASK individual = 0;
            unsigned nextRow = m_N;

            for (unsigned j = sliceSetColStart; j < sliceSetColEnd; ++j) 
            {
                unsigned idx = j - sliceSetColStart;
                while (ptrs[idx] < colPtrs[j + 1] && rows[ptrs[idx]] < i)
                {
                    ++ptrs[idx];
                }

                if (ptrs[idx] < colPtrs[j + 1]) 
                {
                    unsigned r = rows[ptrs[idx]];
                    if (r == i)
                    {
                        individual |= MASK(1) << idx;
                        if ((ptrs[idx] + 1) < colPtrs[j + 1])
                        {
                            nextRow = std::min(nextRow, rows[ptrs[idx] + 1]);
                        }
                    }
                    else
                    {
                        nextRow = std::min(nextRow, r);
                    }
                }
            }

            if (individual != 0)
            {
                tempRowIds.emplace_back(i);
                tempMasks.emplace_back(individual);
            }

            if (tempRowIds.size() == noWarpSlice)
            {
                for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
                {
                    MASK cumulative = 0;
                    unsigned cumulativeCounter = 0;
                    for (unsigned k = thread; k < noWarpSlice; k += WARP_SIZE)
                    {
                        rowIds[sliceSet].emplace_back(tempRowIds[k]);
                        cumulative |= (static_cast<MASK>(tempMasks[k] << (m_SliceSize * cumulativeCounter++)));
                    }
                    masks[sliceSet].emplace_back(cumulative);
                }
                tempRowIds.clear();
                tempMasks.clear();
            }

            i = nextRow;
        }

        if (tempRowIds.size() != 0) // perhaps this could be written in a way first n with 4 * n <= tempRowIds.size() is found and then n thread distribution can be made before leftovers are distributed to (32 - n) threads in a non-coalesced manner
        {
            MASK cumulative = 0;
            unsigned cumulativeCounter = 0;
            for (unsigned k = 0; k < tempRowIds.size(); ++k)
            {
                rowIds[sliceSet].emplace_back(tempRowIds[k]);
                cumulative |= (static_cast<MASK>(tempMasks[k] << (m_SliceSize * cumulativeCounter++)));
                if (cumulativeCounter == noMasks)
                {
                    masks[sliceSet].emplace_back(cumulative);
                    cumulativeCounter = 0;
                    cumulative = 0;
                }
            }

            if (cumulativeCounter != 0)
            {
                for (unsigned k = cumulativeCounter; k < noMasks; ++k)
                {
                    rowIds[sliceSet].emplace_back(0);
                }
                masks[sliceSet].emplace_back(cumulative);
            }
        }

        assert(rowIds[sliceSet].size() == masks[sliceSet].size() * noMasks);

        if (rowIds[sliceSet].size() == 0)
        {
            #pragma omp critical
            {
                ++noEmptySliceSets;
            }
        }
    }

    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet) 
    {
        m_SliceSetPtrs[sliceSet + 1] = m_SliceSetPtrs[sliceSet] + rowIds[sliceSet].size();
    }

    m_RowIds = new unsigned[m_SliceSetPtrs[m_NoSliceSets]];
    unsigned idx = 0;
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        for (unsigned i = 0; i < rowIds[sliceSet].size(); ++i)
        {
            m_RowIds[idx++] = rowIds[sliceSet][i];
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

void BRS::coalescingTest()
{
    m_NoSliceSets = (m_N + m_SliceSize - 1) / m_SliceSize;

    unsigned noMasks = MASK_BITS / m_SliceSize;
    constexpr unsigned CACHELINE_SIZE = 256;

    double setAverage = 0;
    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        unsigned tileStart = m_SliceSetPtrs[sliceSet] / noMasks;
        unsigned tileEnd = m_SliceSetPtrs[sliceSet + 1] / noMasks;

        double average = 0;
        unsigned total = 0;
        for (unsigned tilePtr = tileStart; tilePtr < tileEnd; tilePtr += WARP_SIZE)
        {
            std::vector<std::unordered_map<unsigned, unsigned>> maps(noMasks);
            for (unsigned i = 0; i < WARP_SIZE; ++i)
            {
                if (tilePtr + i < tileEnd)
                {
                    for (unsigned slice = 0; slice < noMasks; ++slice)
                    {
                        unsigned rowIdx = m_RowIds[(tilePtr + i) * noMasks + slice];
                        unsigned bucket = rowIdx / CACHELINE_SIZE;
                        if (maps[slice].contains(bucket))
                        {
                            ++maps[slice][bucket];
                        }
                        else
                        {
                            maps[slice][bucket] = 1;
                        }
                    }
                }
            }
            double running = 0;
            for (unsigned slice = 0; slice < noMasks; ++slice)
            {
                running += maps[slice].size();
            }
            running /= noMasks;
            average += running;
            ++total;
        }
        average /= total; // works only on the assumption that there exists no empty slice set
        #pragma omp critical
        {
            setAverage += average;
        }
    }
    setAverage /= m_NoSliceSets;
    std::cout << "Number of cache lines needed for each mma on bit result on average: " << setAverage << std::endl;
}

void BRS::pattern8Test()
{
    std::vector<std::vector<unsigned>> patterns(m_N, std::vector<unsigned>(256, 0));

    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        unsigned tileStart = m_SliceSetPtrs[sliceSet] / 4;
        unsigned tileEnd = m_SliceSetPtrs[sliceSet + 1] / 4;

        for (unsigned tilePtr = tileStart; tilePtr < tileEnd; ++tilePtr)
        {
            for (unsigned slice = 0; slice < 4; ++slice)
            {
                unsigned row = m_RowIds[tilePtr * 4 + slice];
                unsigned char mask = static_cast<unsigned char>((m_Masks[tilePtr] >> (slice * 8)) & (0x000000FF));
                ++patterns[row][mask];
            }
        }
    }
    double average = 0;
    double maxAverage = 0;
    unsigned maxAverageRow = 0;
    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned i = 0; i < m_N; ++i)
    {
        double averageRow = 0;
        unsigned countRow = 0;
        for (unsigned pattern = 0; pattern < patterns[i].size(); ++pattern)
        {
            unsigned freq = patterns[i][pattern];
            if (freq != 0)
            {
                averageRow += freq;
                ++countRow;
            }
        }
        averageRow /= countRow; // works only on the assumption that there exists no empty row
        #pragma omp critical
        {
            if (averageRow > maxAverage)
            {
                maxAverage = averageRow;
                maxAverageRow = i;
            }
            average += averageRow;
        }
    }
    average /= m_N;
    std::cout << "Max average seen in the row id: " << maxAverageRow << " with value: " << maxAverage << std::endl;
    std::cout << "Average count of each pattern reduced row-wise: " << average << std::endl;
}

void BRS::printBRSData()
{
    unsigned noMasks = MASK_BITS / m_SliceSize;

    std::cout << "MASK size: " << MASK_BITS << std::endl;
    std::cout << "Slice size: " << m_SliceSize << std::endl;
    std::cout << "Number of slice sets: " << m_NoSliceSets << std::endl;
    std::cout << "Number of slices: " << m_SliceSetPtrs[m_NoSliceSets] << std::endl;
    std::cout << "Number of slices in each set row: " << K / m_SliceSize << std::endl;
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
    unsigned noChunks = 10;
    unsigned chunkSize = (m_NoSliceSets + noChunks - 1) / noChunks;
    for (unsigned q = 0; q < noChunks; ++q)
    {
        unsigned start = q * chunkSize;
        unsigned end = std::min(m_NoSliceSets, start + chunkSize);

        unsigned totalSliceSeen = 0;
        unsigned thisChunkSetBits = 0;
        for (unsigned set = start; set < end; ++set)
        {
            for (unsigned ptr = m_SliceSetPtrs[set] / noMasks; ptr < m_SliceSetPtrs[set + 1] / noMasks; ++ptr)
            {
                thisChunkSetBits += __builtin_popcount(m_Masks[ptr]);
            }
            totalSliceSeen += (m_SliceSetPtrs[set + 1] - m_SliceSetPtrs[set]);
        }

        noSetBits += thisChunkSetBits;
        double compressionEff = thisChunkSetBits;
        compressionEff /= (totalSliceSeen * m_SliceSize);
        std::cout << "Chunk: " << q << " - Slice Count: " << totalSliceSeen << " - Bits: " << thisChunkSetBits << " - Compression Efficiency: " << compressionEff << std::endl;
    }
    double compressionEff = noSetBits;
    compressionEff /= ((m_SliceSetPtrs[m_NoSliceSets] / noMasks) * MASK_BITS);
    std::cout << "Total - Set Bits: " << noSetBits << " - Compression Efficiency: " << compressionEff << std::endl;
}

#endif
