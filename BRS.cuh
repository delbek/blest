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
private:
    struct LocalityStatistics
    {
        double averageBucketPerMask = 0;
        double averageBucketPerWarp = 0;
        unsigned maxBucketPerMask = 0;
        unsigned maxBucketPerWarp = 0;
        double averageBucketRangeMask = 0;
        double averageBucketRangeWarp = 0;

        void operator+=(LocalityStatistics right)
        {
            averageBucketPerMask += right.averageBucketPerMask;
            averageBucketPerWarp += right.averageBucketPerWarp;
            if (right.maxBucketPerMask > maxBucketPerMask)
            {
                maxBucketPerMask = right.maxBucketPerMask;
            }
            if (right.maxBucketPerWarp > maxBucketPerWarp)
            {
                maxBucketPerWarp = right.maxBucketPerWarp;
            }
            averageBucketRangeMask += right.averageBucketRangeMask;
            averageBucketRangeWarp += right.averageBucketRangeWarp;
        }

        void maxMask(unsigned value)
        {
            if (value > maxBucketPerMask)
            {
                maxBucketPerMask = value;
            }
        }

        void maxWarp(unsigned value)
        {
            if (value > maxBucketPerWarp)
            {
                maxBucketPerWarp = value;
            }
        }

        void operator/=(unsigned right)
        {
            averageBucketPerMask /= right;
            averageBucketPerWarp /= right;
            averageBucketRangeMask /= right;
            averageBucketRangeWarp /= right;
        }
    };

public:
    BRS(unsigned sliceSize = 32);
    BRS(const BRS& other) = delete;
    BRS(BRS&& other) noexcept = delete;
    BRS& operator=(const BRS& other) = delete;
    BRS& operator=(BRS&& other) noexcept = delete;
    virtual ~BRS();

    void constructFromCSCMatrix(CSC* csc);
    void printBRSData();

    [[nodiscard]] inline unsigned getN() {return m_N;}
    [[nodiscard]] inline unsigned getSliceSize() {return m_SliceSize;}
    [[nodiscard]] inline unsigned getNoRealSliceSets() {return m_NoRealSliceSets;}
    [[nodiscard]] inline unsigned getNoVirtualSliceSets() {return m_NoVirtualSliceSets;}
    [[nodiscard]] inline unsigned* getSliceSetPtrs() {return m_SliceSetPtrs;}
    [[nodiscard]] inline unsigned* getSliceSetIds() {return m_SliceSetIds;}
    [[nodiscard]] inline unsigned* getSliceSetOffsets() {return m_SliceSetOffsets;}
    [[nodiscard]] inline unsigned* getRowIds() {return m_RowIds;}
    [[nodiscard]] inline MASK* getMasks() {return m_Masks;}

private:
    LocalityStatistics distributeSlices(unsigned sliceSet, std::vector<unsigned>& tempRowIds, std::vector<MASK>& tempMasks, std::vector<std::vector<unsigned>>& rowIds, std::vector<std::vector<MASK>>& masks);
    
    LocalityStatistics computeWarpStatistics(std::vector<unsigned>& warpDistributedRows);

private:
    unsigned m_N;
    unsigned m_SliceSize;
    unsigned m_NoRealSliceSets;
    unsigned m_NoVirtualSliceSets;

    unsigned* m_SliceSetPtrs;
    unsigned* m_SliceSetIds;
    unsigned* m_SliceSetOffsets;
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
    delete[] m_SliceSetIds;
    delete[] m_SliceSetOffsets;
    delete[] m_RowIds;
    delete[] m_Masks;
}

void BRS::constructFromCSCMatrix(CSC* csc)
{
    m_N = csc->getN();
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();

    m_NoRealSliceSets = (m_N + m_SliceSize - 1) / m_SliceSize;
    m_NoVirtualSliceSets = 0;

    if (m_SliceSize > MASK_BITS || K % m_SliceSize != 0)
    {
        throw std::runtime_error("Invalid slice size provided.");
    }
    unsigned noMasks = MASK_BITS / m_SliceSize;

    BRS::LocalityStatistics stats;
    std::vector<std::vector<std::vector<unsigned>>> rowIds(m_NoRealSliceSets);
    std::vector<std::vector<std::vector<MASK>>> masks(m_NoRealSliceSets);

    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        BRS::LocalityStatistics statsThread;

        #pragma omp for
        for (unsigned sliceSet = 0; sliceSet < m_NoRealSliceSets; ++sliceSet)
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
    
                i = nextRow;
            }
    
            statsThread += this->distributeSlices(sliceSet, tempRowIds, tempMasks, rowIds[sliceSet], masks[sliceSet]);
        }

        #pragma omp critical
        {
            stats += statsThread;
        }
    }

    m_SliceSetOffsets = new unsigned[m_NoRealSliceSets + 1]; // real to virtual range
    m_SliceSetOffsets[0] = 0;
    for (unsigned realSliceSet = 0; realSliceSet < m_NoRealSliceSets; ++realSliceSet)
    {
        m_SliceSetOffsets[realSliceSet + 1] = m_SliceSetOffsets[realSliceSet] + rowIds[realSliceSet].size();
    }
    m_NoVirtualSliceSets = m_SliceSetOffsets[m_NoRealSliceSets];

    stats /= m_NoVirtualSliceSets;
    std::cout << "Average buckets per mask: " << stats.averageBucketPerMask << std::endl;
    std::cout << "Average buckets per warp: " << stats.averageBucketPerWarp << std::endl;
    std::cout << "Max buckets per mask: " << stats.maxBucketPerMask << std::endl;
    std::cout << "Max buckets per warp: " << stats.maxBucketPerWarp << std::endl;
    std::cout << "Average bucket range per mask: " << stats.averageBucketRangeMask << std::endl;
    std::cout << "Average bucket range per warp: " << stats.averageBucketRangeWarp << std::endl;

    m_SliceSetPtrs = new unsigned[m_NoVirtualSliceSets + 1];
    m_SliceSetIds = new unsigned[m_NoVirtualSliceSets]; // virtual to real
    
    m_SliceSetPtrs[0] = 0;
    for (unsigned realSliceSet = 0; realSliceSet < m_NoRealSliceSets; ++realSliceSet)
    {
        unsigned start = m_SliceSetOffsets[realSliceSet];
        for (unsigned i = 0; i < rowIds[realSliceSet].size(); ++i)
        {
            unsigned virtualSliceSet = start + i;
            m_SliceSetPtrs[virtualSliceSet + 1] = m_SliceSetPtrs[virtualSliceSet] + rowIds[realSliceSet][i].size();
            m_SliceSetIds[virtualSliceSet] = realSliceSet;
        }
    }
    
    m_RowIds = new unsigned[m_SliceSetPtrs[m_NoVirtualSliceSets]];
    unsigned idx = 0;
    for (unsigned realSliceSet = 0; realSliceSet < m_NoRealSliceSets; ++realSliceSet)
    {
        for (unsigned virtualSliceSet = 0; virtualSliceSet < rowIds[realSliceSet].size(); ++virtualSliceSet)
        {
            for (unsigned i = 0; i < rowIds[realSliceSet][virtualSliceSet].size(); ++i)
            {
                m_RowIds[idx++] = rowIds[realSliceSet][virtualSliceSet][i];
            }
        }
    }

    m_Masks = new MASK[(m_SliceSetPtrs[m_NoVirtualSliceSets] / noMasks)];
    idx = 0;
    for (unsigned realSliceSet = 0; realSliceSet < m_NoRealSliceSets; ++realSliceSet)
    {
        for (unsigned virtualSliceSet = 0; virtualSliceSet < masks[realSliceSet].size(); ++virtualSliceSet)
        {
            for (unsigned i = 0; i < masks[realSliceSet][virtualSliceSet].size(); ++i)
            {
                m_Masks[idx++] = masks[realSliceSet][virtualSliceSet][i];
            }
        }
    }

    this->printBRSData();
}

BRS::LocalityStatistics BRS::distributeSlices(unsigned sliceSet, std::vector<unsigned>& tempRowIds, std::vector<MASK>& tempMasks, std::vector<std::vector<unsigned>>& rowIds, std::vector<std::vector<MASK>>& masks)
{
    unsigned noMasks = MASK_BITS / m_SliceSize;
    unsigned fullWork = WARP_SIZE * noMasks;

    LocalityStatistics stats;

    unsigned noComplete = tempRowIds.size() / fullWork;
    for (unsigned complete = 0; complete < noComplete; ++complete)
    {
        std::vector<unsigned> vsetRows;
        std::vector<MASK> vsetMasks;
        for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
        {
            MASK cumulative = 0;
            unsigned cumulativeCounter = 0;
            for (unsigned mask = 0; mask < noMasks; ++mask)
            {
                unsigned current = complete * fullWork + mask * WARP_SIZE + thread;
                vsetRows.emplace_back(tempRowIds[current]);
                cumulative |= (static_cast<MASK>(tempMasks[current] << (m_SliceSize * cumulativeCounter++)));
            }
            vsetMasks.emplace_back(cumulative);
        }
        stats += this->computeWarpStatistics(vsetRows);
        
        rowIds.emplace_back(vsetRows);
        masks.emplace_back(vsetMasks);
    }

    unsigned leftoverStart = noComplete * fullWork;
    if (leftoverStart < tempRowIds.size())
    {
        std::vector<unsigned> vsetRows;
        std::vector<MASK> vsetMasks;
    
        MASK cumulative = 0;
        unsigned cumulativeCounter = 0;
        for (unsigned slice = leftoverStart; slice < tempRowIds.size(); ++slice)
        {
            vsetRows.emplace_back(tempRowIds[slice]);
            cumulative |= (static_cast<MASK>(tempMasks[slice] << (m_SliceSize * cumulativeCounter++)));
            if (cumulativeCounter == noMasks)
            {
                vsetMasks.emplace_back(cumulative);
                cumulative = 0;
                cumulativeCounter = 0;
            }
        }
        if (cumulativeCounter != 0)
        {
            for (; cumulativeCounter < noMasks; ++cumulativeCounter)
            {
                vsetRows.emplace_back(0);
            }
            vsetMasks.emplace_back(cumulative);
        
        }
        stats += this->computeWarpStatistics(vsetRows);
        
        rowIds.emplace_back(vsetRows);
        masks.emplace_back(vsetMasks);
    }

    if (leftoverStart < tempRowIds.size())
    {
        ++noComplete;
    }
    if (noComplete != 0)
    {
        stats /= noComplete;
    }
    return stats;
}

BRS::LocalityStatistics BRS::computeWarpStatistics(std::vector<unsigned>& warp)
{
    unsigned noMasks = MASK_BITS / m_SliceSize;
    LocalityStatistics stats;

    std::unordered_map<unsigned, unsigned> warpBuckets;
    unsigned minWarp = UNSIGNED_MAX;
    unsigned maxWarp = 0;
    for (unsigned mask = 0; mask < noMasks; ++mask)
    {
        std::unordered_map<unsigned, unsigned> maskBuckets;
        unsigned minMask = UNSIGNED_MAX;
        unsigned maxMask = 0;
        for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
        {
            unsigned current = thread * noMasks + mask;
            if (current >= warp.size()) continue;
            unsigned bucket = warp[current] / MASK_BITS;
            if (maskBuckets.find(bucket) != maskBuckets.end())
            {
                ++maskBuckets[bucket];
            }
            else
            {
                maskBuckets[bucket] = 1;
            }
            if (warpBuckets.find(bucket) != warpBuckets.end())
            {
                ++warpBuckets[bucket];
            }
            else
            {
                warpBuckets[bucket] = 1;
            }
            if (bucket < minMask) minMask = bucket;
            if (bucket > maxMask) maxMask = bucket;
            if (bucket < minWarp) minWarp = bucket;
            if (bucket > maxWarp) maxWarp = bucket;
        }
        
        stats.averageBucketPerMask += maskBuckets.size();
        stats.maxMask(maskBuckets.size());
        stats.averageBucketRangeMask += (maxMask - minMask);
    }
    stats.averageBucketPerMask /= noMasks;
    stats.averageBucketRangeMask /= noMasks;
    
    stats.averageBucketPerWarp = warpBuckets.size();
    stats.maxWarp(warpBuckets.size());
    stats.averageBucketRangeWarp = (maxWarp - minWarp);

    return stats;
}

void BRS::printBRSData()
{
    unsigned noMasks = MASK_BITS / m_SliceSize;

    std::cout << "MASK size: " << MASK_BITS << std::endl;
    std::cout << "Slice size: " << m_SliceSize << std::endl;
    std::cout << "Number of real slice sets: " << m_NoRealSliceSets << std::endl;
    std::cout << "Number of virtual slice sets: " << m_NoVirtualSliceSets << std::endl;
    std::cout << "Number of slices: " << m_SliceSetPtrs[m_NoVirtualSliceSets] << std::endl;
    std::cout << "Number of slices in each mask: " << noMasks << std::endl;

    unsigned empty = 0;
    double average = 0;
    for (unsigned ss = 0; ss < m_NoVirtualSliceSets; ++ss)
    {
        if (m_SliceSetPtrs[ss + 1] == m_SliceSetPtrs[ss])
        {
            ++empty;
        }
        average += (m_SliceSetPtrs[ss + 1] - m_SliceSetPtrs[ss]);
    }
    average /= m_NoVirtualSliceSets;
    std::cout << "Number of empty slice sets: " << empty << std::endl;
    std::cout << "Average number of slices in each set " << average << std::endl;

    double variance = 0;
    for (unsigned ss = 0; ss < m_NoVirtualSliceSets; ++ss)
    {
        unsigned length = (m_SliceSetPtrs[ss + 1] - m_SliceSetPtrs[ss]);
        double diff = length - average;
        variance += diff * diff;
    }
    variance /= m_NoVirtualSliceSets;
    double standardDeviation = std::sqrt(variance);
    std::cout << "Standard deviation of the number of slices in each set: " << standardDeviation << std::endl;

    unsigned totalNumberOfEdgesCheck = 0;
    unsigned noChunks = 10;
    unsigned chunkSize = (m_NoVirtualSliceSets + noChunks - 1) / noChunks;
    for (unsigned q = 0; q < noChunks; ++q)
    {
        bool chunkEmpty = true;
        unsigned start = q * chunkSize;
        unsigned end = std::min(m_NoVirtualSliceSets, start + chunkSize);

        for (unsigned set = start; set < end; ++set)
        {
            for (unsigned ptr = m_SliceSetPtrs[set] / noMasks; ptr < m_SliceSetPtrs[set + 1] / noMasks; ++ptr)
            {
                totalNumberOfEdgesCheck += __builtin_popcount(m_Masks[ptr]);
            }
            chunkEmpty &= (m_SliceSetPtrs[set + 1] == m_SliceSetPtrs[set]);
        }

        if (!chunkEmpty)
        {
            std::cout << "Chunk: " << q 
            << " - Slice Count: " << (m_SliceSetPtrs[end] - m_SliceSetPtrs[start])
            << " - Bits: " << (m_SliceSetPtrs[end] - m_SliceSetPtrs[start]) * m_SliceSize << std::endl;
        }
    }
    std::cout << "Bits (total): " << m_SliceSetPtrs[m_NoVirtualSliceSets] * m_SliceSize << std::endl;

    std::cout << "Check: " << totalNumberOfEdgesCheck << std::endl;
}

#endif
