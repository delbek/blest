#pragma once

#include "BitMatrix.cuh"
#include "CSC.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <map>

class BRS: public BitMatrix
{
public:
    struct SliceSetInformation
    {   
        unsigned noActive = 0;
        unsigned noEntered = 0;
    };
    struct SliceInformation
    {
        unsigned noEntered = 0;
    };
    struct VSet
    {
        std::vector<unsigned> rows;
        std::vector<MASK> masks;
        unsigned rset;
    };
    struct VSetStatistics
    {
        double averageBucketPerMask = 0;
        double averageBucketPerWarp = 0;
        double averageBucketRangeMask = 0;
        double averageBucketRangeWarp = 0;

        void operator+=(const VSetStatistics& right)
        {
            averageBucketPerMask += right.averageBucketPerMask;
            averageBucketPerWarp += right.averageBucketPerWarp;
            averageBucketRangeMask += right.averageBucketRangeMask;
            averageBucketRangeWarp += right.averageBucketRangeWarp;
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
    BRS(unsigned sliceSize, unsigned isFullPadding, std::ofstream& file);
    BRS(const BRS& other) = delete;
    BRS(BRS&& other) noexcept = delete;
    BRS& operator=(const BRS& other) = delete;
    BRS& operator=(BRS&& other) noexcept = delete;
    virtual ~BRS();

    void constructFromBinary(std::string filename);
    void saveToBinary(std::string filename);

    void constructFromCSCMatrix(CSC* csc);
    void printBRSData();
    void brsAnalysis();
    void kernelAnalysis(unsigned source, unsigned totalLevels, unsigned totalVisited, double time, SliceSetInformation* rsetInformation, SliceSetInformation* vsetInformation, SliceInformation* sliceInformation);

    [[nodiscard]] inline unsigned getN() {return m_N;}
    [[nodiscard]] inline unsigned getSliceSize() {return m_SliceSize;}
    [[nodiscard]] inline unsigned getNoRealSliceSets() {return m_NoRealSliceSets;}
    [[nodiscard]] inline unsigned getNoVirtualSliceSets() {return m_NoVirtualSliceSets;}
    [[nodiscard]] inline unsigned getNoSlices() {return m_NoSlices;}
    [[nodiscard]] inline bool getIsFullPadding() {return m_IsFullPadding;}
    [[nodiscard]] inline unsigned getNoPaddedSlices() {return m_NoPaddedSlices;}
    [[nodiscard]] inline unsigned* getSliceSetPtrs() {return m_SliceSetPtrs;}
    [[nodiscard]] inline unsigned* getVirtualToReal() {return m_VirtualToReal;}
    [[nodiscard]] inline unsigned* getRealPtrs() {return m_RealPtrs;}
    [[nodiscard]] inline unsigned* getRowIds() {return m_RowIds;}
    [[nodiscard]] inline MASK* getMasks() {return m_Masks;}

private:
    void grayCodeOrder(VSet& rset);
    VSetStatistics distributeSlices(const VSet& rset, std::vector<VSet>& vsets);
    VSetStatistics computeVSetStatistics(const VSet& vset);

private:
    unsigned m_N;
    unsigned m_SliceSize;
    unsigned m_NoRealSliceSets;
    unsigned m_NoVirtualSliceSets;
    unsigned m_NoSlices;
    bool m_IsFullPadding;
    unsigned m_NoPaddedSlices;

    unsigned* m_SliceSetPtrs;
    unsigned* m_VirtualToReal;
    unsigned* m_RealPtrs;
    unsigned* m_RowIds;
    MASK* m_Masks;

    // profiling data
    std::ofstream& m_File;
};

BRS::BRS(unsigned sliceSize, unsigned isFullPadding, std::ofstream& file)
: BitMatrix(),
  m_SliceSize(sliceSize),
  m_IsFullPadding(isFullPadding),
  m_File(file)
{
    m_File << "N\tNNZ\tMASK_BITS\tSliceSize\t#RSet\t#VSet\t#Slices\tnoMasks\tAvg(Slice/VSet)\tMin(Slice/VSet)\tMax(Slice/VSet)\tStd(Slice/Vset)\t#PaddedSlices\tPadded/All\tBits" << std::endl;
}

BRS::~BRS()
{
    delete[] m_SliceSetPtrs;
    delete[] m_VirtualToReal;
    delete[] m_RealPtrs;
    delete[] m_RowIds;
    delete[] m_Masks;
}

void BRS::saveToBinary(std::string filename)
{
    std::ofstream out(filename, std::ios::binary);

    unsigned noMasks = MASK_BITS / m_SliceSize;

    out.write(reinterpret_cast<const char*>(&m_N), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_SliceSize), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_NoRealSliceSets), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_NoVirtualSliceSets), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_NoSlices), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_IsFullPadding), (sizeof(bool)));
    out.write(reinterpret_cast<const char*>(&m_NoPaddedSlices), (sizeof(unsigned)));

    out.write(reinterpret_cast<const char*>(m_SliceSetPtrs), (sizeof(unsigned) * (m_NoVirtualSliceSets + 1)));
    out.write(reinterpret_cast<const char*>(m_VirtualToReal), (sizeof(unsigned) * m_NoVirtualSliceSets));
    out.write(reinterpret_cast<const char*>(m_RealPtrs), (sizeof(unsigned) * (m_NoRealSliceSets + 1)));
    out.write(reinterpret_cast<const char*>(m_RowIds), (sizeof(unsigned) * m_NoSlices));
    out.write(reinterpret_cast<const char*>(m_Masks), (sizeof(unsigned) * (m_NoSlices / noMasks)));

    out.close();
}

void BRS::constructFromBinary(std::string filename)
{
    std::ifstream in(filename, std::ios::binary);

    in.read(reinterpret_cast<char*>(&m_N), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_SliceSize), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_NoRealSliceSets), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_NoVirtualSliceSets), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_NoSlices), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_IsFullPadding), sizeof(bool));
    in.read(reinterpret_cast<char*>(&m_NoPaddedSlices), sizeof(unsigned));

    unsigned noMasks = MASK_BITS / m_SliceSize;

    m_SliceSetPtrs = new unsigned[m_NoVirtualSliceSets + 1];
    in.read(reinterpret_cast<char*>(m_SliceSetPtrs), (sizeof(unsigned) * (m_NoVirtualSliceSets + 1)));

    m_VirtualToReal = new unsigned[m_NoVirtualSliceSets];
    in.read(reinterpret_cast<char*>(m_VirtualToReal), (sizeof(unsigned) * m_NoVirtualSliceSets));

    m_RealPtrs = new unsigned[m_NoRealSliceSets + 1];
    in.read(reinterpret_cast<char*>(m_RealPtrs), (sizeof(unsigned) * (m_NoRealSliceSets + 1)));

    m_RowIds = new unsigned[m_NoSlices];
    in.read(reinterpret_cast<char*>(m_RowIds), (sizeof(unsigned) * m_NoSlices));

    m_Masks = new unsigned[m_NoSlices / noMasks];
    in.read(reinterpret_cast<char*>(m_Masks), (sizeof(unsigned) * (m_NoSlices / noMasks)));

    in.close();

    std::cout << "BRS read from binary." << std::endl;

    this->printBRSData();
    this->brsAnalysis();
}

void BRS::constructFromCSCMatrix(CSC* csc)
{
    m_N = csc->getN();
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();
    fileFlush(m_File, m_N); fileFlush(m_File, colPtrs[m_N]);

    m_NoRealSliceSets = (m_N + m_SliceSize - 1) / m_SliceSize;
    m_NoVirtualSliceSets = 0;

    if (m_SliceSize > MASK_BITS || K % m_SliceSize != 0)
    {
        throw std::runtime_error("Invalid slice size provided.");
    }
    unsigned noMasks = MASK_BITS / m_SliceSize;
    unsigned fullWork = WARP_SIZE * noMasks;

    VSetStatistics stats;
    std::vector<std::vector<VSet>> rsets(m_NoRealSliceSets);
    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        VSetStatistics threadStats;
        #pragma omp for
        for (unsigned rset = 0; rset < m_NoRealSliceSets; ++rset)
        {
            std::map<unsigned, MASK> map;
            std::vector<std::pair<unsigned, MASK>> additionals;
         
            unsigned sliceSetColStart = rset * m_SliceSize;
            unsigned sliceSetColEnd = std::min(m_N, sliceSetColStart + m_SliceSize);
            unsigned noCols = sliceSetColEnd - sliceSetColStart;

            std::vector<unsigned> ptrs(noCols);
            for (unsigned idx = 0; idx < noCols; ++idx)
            {
                unsigned j = sliceSetColStart + idx;
                ptrs[idx] = colPtrs[j]; // do note that for this approach to work out, adjacency list must be sorted
            }
    
            unsigned i = 0;
            while (i < m_N)
            {
                MASK individual = 0;
                unsigned nextRow = m_N;
    
                for (unsigned idx = 0; idx < noCols; ++idx) 
                {
                    unsigned j = sliceSetColStart + idx;
                    while (ptrs[idx] < colPtrs[j + 1] && rows[ptrs[idx]] < i)
                    {
                        ++ptrs[idx];
                    }
    
                    unsigned p = ptrs[idx];
                    if (p < colPtrs[j + 1]) // have nonzero unprocessed
                    {
                        if (rows[p] == i) // nonzero in the target row
                        {
                            individual |= (1 << idx);
                            ++p;
                        }
                        ptrs[idx] = p;
                        if (p < colPtrs[j + 1]) // next closest nonzero across all columns having nonzero unprocessed
                        {
                            nextRow = std::min(nextRow, rows[p]);
                        }
                    }
                }
    
                if (individual != 0)
                {
                    map[i] = individual;
                    if (csc->isRoadNetwork())
                    {
                        if (__builtin_popcount(individual) >= ROAD_NETWORK_FUSION_CONSTANT)
                        {
                            additionals.emplace_back(i, individual);
                        }
                    }
                    else
                    {
                        if (__builtin_popcount(individual) >= SOCIAL_NETWORK_FUSION_CONSTANT)
                        {
                            additionals.emplace_back(i, individual);
                        }
                    }
                }
    
                i = nextRow;
            }

            unsigned noComplete = (map.size() / fullWork) * fullWork;
            unsigned left = (fullWork - (map.size() - noComplete));
            unsigned added = 0;

            if (left != 0)
            {
                while (true)
                {
                    std::vector<std::pair<unsigned, MASK>> next;
                    bool progressed = false;

                    for (const auto& add: additionals)
                    {
                        const unsigned j = add.first;
                        for (unsigned ptr = colPtrs[j]; ptr < colPtrs[j + 1]; ++ptr)
                        {
                            const unsigned i = rows[ptr];
                            if (map.contains(i)) continue;

                            map[i] = add.second;
                            next.emplace_back(i, add.second);
                            progressed = true;
                            ++added;

                            if (added == left)
                            {
                                goto end;
                            }
                        }
                    }
                    if (!progressed) break;

                    additionals.swap(next);
                }
            }

        end:
            VSet realSet;
            realSet.rset = rset;
            for (const auto& slice: map)
            {
                realSet.rows.emplace_back(slice.first);
                realSet.masks.emplace_back(slice.second);
            }

            this->grayCodeOrder(realSet);
    
            threadStats += this->distributeSlices(realSet, rsets[rset]);
        }
        #pragma omp critical
        {
            stats += threadStats;
        }
    }
    std::vector<VSet> vsets;
    for (unsigned rset = 0; rset < rsets.size(); ++rset)
    {
        for (auto& vset: rsets[rset])
        {
            vsets.emplace_back(std::move(vset));
        }
    }
    rsets.clear();
    m_NoVirtualSliceSets = vsets.size();

    stats /= m_NoVirtualSliceSets;
    std::cout << "Average buckets per mask: " << stats.averageBucketPerMask << std::endl;
    std::cout << "Average buckets per warp: " << stats.averageBucketPerWarp << std::endl;
    std::cout << "Average bucket range per mask: " << stats.averageBucketRangeMask << std::endl;
    std::cout << "Average bucket range per warp: " << stats.averageBucketRangeWarp << std::endl;

    m_RealPtrs = new unsigned[m_NoRealSliceSets + 1];
    std::fill(m_RealPtrs, m_RealPtrs + m_NoRealSliceSets + 1, 0);
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        unsigned rset = vsets[vset].rset;
        ++m_RealPtrs[rset + 1];
    }
    for (unsigned rset = 0; rset < m_NoRealSliceSets; ++rset)
    {
        m_RealPtrs[rset + 1] += m_RealPtrs[rset];
    }

    m_SliceSetPtrs = new unsigned[m_NoVirtualSliceSets + 1];
    m_VirtualToReal = new unsigned[m_NoVirtualSliceSets];
    
    m_SliceSetPtrs[0] = 0;
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        m_SliceSetPtrs[vset + 1] = m_SliceSetPtrs[vset] + vsets[vset].rows.size();
        m_VirtualToReal[vset] = vsets[vset].rset;
    }
    m_NoSlices = m_SliceSetPtrs[m_NoVirtualSliceSets];

    m_RowIds = new unsigned[m_NoSlices];
    unsigned idx = 0;
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        for (unsigned i = 0; i < vsets[vset].rows.size(); ++i)
        {
            m_RowIds[idx++] = vsets[vset].rows[i];
        }
    }

    m_Masks = new MASK[(m_NoSlices / noMasks)];
    idx = 0;
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        for (unsigned i = 0; i < vsets[vset].masks.size(); ++i)
        {
            m_Masks[idx++] = vsets[vset].masks[i];
        }
    }

    this->printBRSData();
    this->brsAnalysis();
}

void BRS::grayCodeOrder(VSet& rset)
{
    const unsigned bucketCount = (1 << m_SliceSize);

    std::vector<std::vector<unsigned>> buckets(bucketCount);

    const unsigned n = rset.rows.size();
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned mask = static_cast<unsigned>(rset.masks[i]);
        buckets[mask].emplace_back(rset.rows[i]);
    }

    VSet newSet;
    newSet.rset = rset.rset;
    newSet.rows.reserve(n);
    newSet.masks.reserve(n);

    for (unsigned i = 1; i < bucketCount; ++i)
    {
        unsigned bucket = i ^ (i >> 1);
        for (const auto& row: buckets[bucket])
        {
            newSet.rows.emplace_back(row);
            newSet.masks.emplace_back(static_cast<MASK>(bucket));
        }
    }

    rset = std::move(newSet);
}

BRS::VSetStatistics BRS::distributeSlices(const VSet& rset, std::vector<VSet>& vsets)
{
    unsigned noMasks = MASK_BITS / m_SliceSize;
    unsigned fullWork = WARP_SIZE * noMasks;

    VSetStatistics stats;

    unsigned noComplete = rset.rows.size() / fullWork;
    for (unsigned complete = 0; complete < noComplete; ++complete)
    {
        VSet vset;
        vset.rset = rset.rset;
        for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
        {
            MASK cumulative = 0;
            unsigned cumulativeCounter = 0;
            for (unsigned mask = 0; mask < noMasks; ++mask)
            {
                unsigned current = complete * fullWork + mask * WARP_SIZE + thread;
                vset.rows.emplace_back(rset.rows[current]);
                cumulative |= (rset.masks[current] << (m_SliceSize * cumulativeCounter++));
            }
            vset.masks.emplace_back(cumulative);
        }
        stats += this->computeVSetStatistics(vset);
        vsets.emplace_back(vset);
    }

    unsigned leftoverStart = noComplete * fullWork;
    if (leftoverStart < rset.rows.size())
    {
        VSet vset;
        vset.rset = rset.rset;

        if (m_IsFullPadding)
        {
            for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
            {
                MASK cumulative = 0;
                unsigned cumulativeCounter = 0;
                for (unsigned mask = 0; mask < noMasks; ++mask)
                {
                    unsigned current = leftoverStart + mask * WARP_SIZE + thread;
                    if (current < rset.rows.size())
                    {
                        vset.rows.emplace_back(rset.rows[current]);
                        cumulative |= (rset.masks[current] << (m_SliceSize * cumulativeCounter++));
                    }
                    else
                    {
                        vset.rows.emplace_back(0);
                        ++cumulativeCounter;
                    }
                }
                vset.masks.emplace_back(cumulative);
            }
        }
        else
        {
            MASK cumulative = 0;
            unsigned cumulativeCounter = 0;
            for (unsigned slice = leftoverStart; slice < rset.rows.size(); ++slice)
            {
                vset.rows.emplace_back(rset.rows[slice]);
                cumulative |= (rset.masks[slice] << (m_SliceSize * cumulativeCounter++));
                if (cumulativeCounter == noMasks)
                {
                    vset.masks.emplace_back(cumulative);
                    cumulative = 0;
                    cumulativeCounter = 0;
                }
            }
            if (cumulativeCounter != 0)
            {
                for (; cumulativeCounter < noMasks; ++cumulativeCounter)
                {
                    vset.rows.emplace_back(0);
                }
                vset.masks.emplace_back(cumulative);
            }
        }

        stats += this->computeVSetStatistics(vset);
        vsets.emplace_back(vset);
    }

    return stats;
}

BRS::VSetStatistics BRS::computeVSetStatistics(const VSet& vset)
{
    unsigned noMasks = MASK_BITS / m_SliceSize;
    VSetStatistics stats;

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
            if (current >= vset.rows.size()) continue;
            unsigned bucket = vset.rows[current] / MASK_BITS;
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
        stats.averageBucketRangeMask += (maxMask - minMask);
    }
    stats.averageBucketPerMask /= noMasks;
    stats.averageBucketRangeMask /= noMasks;
    
    stats.averageBucketPerWarp = warpBuckets.size();
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
    std::cout << "Number of slices: " << m_NoSlices << std::endl;
    std::cout << "Number of slices in each mask: " << noMasks << std::endl;

    double average = 0;
    unsigned min = UNSIGNED_MAX;
    unsigned max = 0;
    for (unsigned ss = 0; ss < m_NoVirtualSliceSets; ++ss)
    {
        unsigned slices = m_SliceSetPtrs[ss + 1] - m_SliceSetPtrs[ss];
        average += slices;
        min = std::min(min, slices);
        max = std::max(max, slices);
    }
    average /= m_NoVirtualSliceSets;
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

    std::cout << "Min number of slices in any set " << min << std::endl;
    std::cout << "Max number of slices in any set " << max << std::endl;

    unsigned padded = 0;
    unsigned totalNumberOfEdgesCheck = 0;
    unsigned noChunks = 10;
    unsigned chunkSize = (m_NoVirtualSliceSets + noChunks - 1) / noChunks;
    for (unsigned chunk = 0; chunk < noChunks; ++chunk)
    {
        unsigned start = chunk * chunkSize;
        unsigned end = std::min(m_NoVirtualSliceSets, start + chunkSize);

        for (unsigned vset = start; vset < end; ++vset)
        {
            for (unsigned ptr = m_SliceSetPtrs[vset] / noMasks; ptr < m_SliceSetPtrs[vset + 1] / noMasks; ++ptr)
            {
                MASK current = m_Masks[ptr];
                for (unsigned mask = 0; mask < noMasks; ++mask)
                {
                    unsigned shift = mask * m_SliceSize;
                    MASK pattern = ((current >> shift) & 0x000000FF);
                    unsigned edges = __builtin_popcount(pattern);
                    if (edges == 0)
                    {
                        ++padded;
                    }
                    totalNumberOfEdgesCheck += edges;
                }
            }
        }

        std::cout << "Chunk: " << chunk << " - Slice Count: " << (m_SliceSetPtrs[end] - m_SliceSetPtrs[start]) << " - Bits: " << (m_SliceSetPtrs[end] - m_SliceSetPtrs[start]) * m_SliceSize << std::endl;
    }
    m_NoPaddedSlices = padded;
    std::cout << "Is Fully Padded?: " << m_IsFullPadding << std::endl;
    std::cout << "Total padded slices: " << m_NoPaddedSlices << std::endl;
    std::cout << "Padded/All: " << (static_cast<double>(m_NoPaddedSlices) / m_NoSlices) << std::endl;
    std::cout << "Bits (total): " << (m_NoSlices * m_SliceSize) << std::endl;
    std::cout << "Check: " << totalNumberOfEdgesCheck << std::endl;

    fileFlush(m_File, MASK_BITS); fileFlush(m_File, m_SliceSize); fileFlush(m_File, m_NoRealSliceSets); fileFlush(m_File, m_NoVirtualSliceSets); fileFlush(m_File, m_NoSlices); fileFlush(m_File, noMasks);
    fileFlush(m_File, average); fileFlush(m_File, min); fileFlush(m_File, max); fileFlush(m_File, standardDeviation); fileFlush(m_File, m_NoPaddedSlices); fileFlush(m_File, (static_cast<double>(m_NoPaddedSlices) / m_NoSlices)); fileFlush(m_File, (m_NoSlices * m_SliceSize));
    m_File << std::endl;
}

void BRS::brsAnalysis()
{
    unsigned noMasks = MASK_BITS / m_SliceSize;

    std::vector<unsigned> patterns(1 << m_SliceSize, 0);
    std::vector<unsigned> counts(m_SliceSize + 1, 0);
    for (unsigned rset = 0; rset < m_NoRealSliceSets; ++rset)
    {
        for (unsigned vset = m_RealPtrs[rset]; vset < m_RealPtrs[rset + 1]; ++vset)
        {
            for (unsigned slice = m_SliceSetPtrs[vset]; slice < m_SliceSetPtrs[vset + 1]; ++slice)
            {
                MASK current = m_Masks[slice / noMasks];
                unsigned shift = (slice % noMasks) * 8;
                MASK pattern = ((current >> shift) & 0x000000FF);
                ++counts[__builtin_popcount(pattern)];
                ++patterns[pattern];
            }
        }
    }
    for (unsigned i = 0; i < (1 << m_SliceSize); ++i)
    {
        std::stringstream stream;
        for (int bit = 7; bit >= 0; --bit)
        {
            stream << ((i >> bit) & 1);
        }
        fileFlush(m_File, stream.str());
    }
    m_File << std::endl;
    for (unsigned i = 0; i < (1 << m_SliceSize); ++i)
    {
        fileFlush(m_File, patterns[i]);
        //std::cout << "Pattern: " << i << ' ' << patterns[i].totalCount << " - No Bits: " << __builtin_popcount(i) << std::endl;
    }
    m_File << std::endl;

    for (unsigned i = 0; i <= m_SliceSize; ++i)
    {
        fileFlush(m_File, std::to_string(i) + " Bits");
    }
    m_File << std::endl;
    for (unsigned i = 0; i <= m_SliceSize; ++i)
    {
        fileFlush(m_File, counts[i]);
        //std::cout << i << " Bit Count: " << counts[i] << std::endl;
    }
    m_File << std::endl;

    m_File << "Source\tTotalLevels\tTotalVisited\tTime(ms)\tAvgRSetEntrance\tAvgVSetEntrance\tAvgSliceEntrance" << std::endl;
}

void BRS::kernelAnalysis(unsigned source, unsigned totalLevels, unsigned totalVisited, double time, SliceSetInformation* rsetInformation, SliceSetInformation* vsetInformation, SliceInformation* sliceInformation)
{
    double averageRSet = 0;
    for (unsigned rset = 0; rset < m_NoRealSliceSets; ++rset)
    {
        SliceSetInformation set = rsetInformation[rset];
        averageRSet += set.noEntered;
    }
    averageRSet /= m_NoRealSliceSets;
    
    double averageVSetActive = 0;
    double averageVSet = 0;
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        SliceSetInformation set = vsetInformation[vset];
        averageVSetActive += set.noActive;
        averageVSet += set.noEntered;
    }
    averageVSetActive /= m_NoVirtualSliceSets;
    averageVSet /= m_NoVirtualSliceSets;

    double averageSlice = 0;
    for (unsigned slice = 0; slice < m_NoSlices; ++slice)
    {
        SliceInformation s = sliceInformation[slice];
        averageSlice += s.noEntered;
    }
    averageSlice /= (m_NoSlices - m_NoPaddedSlices);

    delete[] rsetInformation;
    delete[] vsetInformation;
    delete[] sliceInformation;

    fileFlush(m_File, source); fileFlush(m_File, totalLevels); fileFlush(m_File, totalVisited); fileFlush(m_File, time * 1000); fileFlush(m_File, averageRSet); fileFlush(m_File, averageVSet); fileFlush(m_File, averageSlice);
    m_File << std::endl;
}
