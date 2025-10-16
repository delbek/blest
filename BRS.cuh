#ifndef BRS_CUH
#define BRS_CUH

#include "BitMatrix.cuh"
#include "CSC.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <unordered_map>

class BRS: public BitMatrix
{
public:
    struct SliceSetInformation
    {
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
        std::vector<unsigned> rsets;
    };
    
public:
    BRS(unsigned sliceSize, unsigned isFullPadding, std::ofstream& file);
    BRS(const BRS& other) = delete;
    BRS(BRS&& other) noexcept = delete;
    BRS& operator=(const BRS& other) = delete;
    BRS& operator=(BRS&& other) noexcept = delete;
    virtual ~BRS();

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
    void distributeSlices(std::vector<unsigned>& tempRowIds, std::vector<MASK>& tempMasks, std::vector<VSet>& vsets);
    void formSuperVSet4(unsigned targetRegion, std::vector<unsigned>& tempRowIds, std::vector<MASK>& tempMasks, VSet& vset);

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

void BRS::constructFromCSCMatrix(CSC* csc)
{
    m_N = csc->getN();
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();
    fileFlush(m_File, m_N); fileFlush(m_File, colPtrs[m_N]);

    unsigned superSetSize = 4;

    m_NoRealSliceSets = (m_N + m_SliceSize - 1) / m_SliceSize;
    for (; m_NoRealSliceSets % superSetSize != 0; ++m_NoRealSliceSets);
    m_NoVirtualSliceSets = 0;

    if (m_SliceSize > MASK_BITS || K % m_SliceSize != 0)
    {
        throw std::runtime_error("Invalid slice size provided.");
    }
    unsigned noMasks = MASK_BITS / m_SliceSize;

    assert(m_NoRealSliceSets % superSetSize == 0);

    std::vector<VSet> vsets;
    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        std::vector<VSet> mySets;
        #pragma omp for
        for (unsigned rsetTile = 0; rsetTile < m_NoRealSliceSets; rsetTile += superSetSize)
        {
            VSet vset;
            vset.rows.resize(128);
            vset.rows.assign(128, 0);
            vset.masks.resize(32);
            vset.masks.assign(32, 0);
            for (unsigned rset = rsetTile; rset < rsetTile + superSetSize; ++rset)
            {
                vset.rsets.emplace_back(rset);
                unsigned sliceSetColStart = rset * m_SliceSize;
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
                        tempRowIds.emplace_back(i);
                        tempMasks.emplace_back(individual);
                    }
        
                    i = nextRow;
                }
        
                this->formSuperVSet4(rset % superSetSize, tempRowIds, tempMasks, vset);
            }
            mySets.emplace_back(vset);
        }
        #pragma omp critical
        {
            for (unsigned i = 0; i < mySets.size(); ++i)
            {
                vsets.emplace_back(mySets[i]);
            }
        }
    }

    m_RealPtrs = new unsigned[m_NoRealSliceSets + 1];
    std::fill(m_RealPtrs, m_RealPtrs + m_NoRealSliceSets + 1, 0);
    for (unsigned vset = 0; vset < vsets.size(); ++vset)
    {
        for (unsigned i = 0; i < vsets[vset].rsets.size(); ++i)
        {
            unsigned rset = vsets[vset].rsets[i];
            m_RealPtrs[rset] = vset;
        }
    }
    m_NoVirtualSliceSets = vsets.size();

    m_SliceSetPtrs = new unsigned[m_NoVirtualSliceSets + 1];
    m_VirtualToReal = new unsigned[m_NoVirtualSliceSets];
    
    m_SliceSetPtrs[0] = 0;
    for (unsigned vset = 0; vset < vsets.size(); ++vset)
    {
        m_SliceSetPtrs[vset + 1] = m_SliceSetPtrs[vset] + vsets[vset].rows.size();
        m_VirtualToReal[vset] = vsets[vset].rsets.front();
    }
    m_NoSlices = m_SliceSetPtrs[m_NoVirtualSliceSets];

    m_RowIds = new unsigned[m_NoSlices];
    unsigned idx = 0;
    for (unsigned vset = 0; vset < vsets.size(); ++vset)
    {
        for (unsigned i = 0; i < vsets[vset].rows.size(); ++i)
        {
            m_RowIds[idx++] = vsets[vset].rows[i];
        }
    }

    m_Masks = new MASK[(m_NoSlices / noMasks)];
    idx = 0;
    for (unsigned vset = 0; vset < vsets.size(); ++vset)
    {
        for (unsigned i = 0; i < vsets[vset].masks.size(); ++i)
        {
            m_Masks[idx++] = vsets[vset].masks[i];
        }
    }

    this->printBRSData();
    this->brsAnalysis();
}

void BRS::formSuperVSet4(unsigned targetRegion, std::vector<unsigned>& tempRowIds, std::vector<MASK>& tempMasks, VSet& vset)
{
    unsigned noMasks = MASK_BITS / m_SliceSize;
    unsigned regionThreadSize = WARP_SIZE / 4;

    for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
    {
        unsigned region = (thread % 4);
        if (region != targetRegion)
        {
            continue;
        }
        unsigned regionThreadId = thread / 4;
        MASK cumulative = 0;
        unsigned cumulativeCounter = 0;
        for (unsigned mask = 0; mask < noMasks; ++mask)
        {
            unsigned current = mask * regionThreadSize + regionThreadId;
            if (current < tempRowIds.size())
            {
                vset.rows[thread * noMasks + mask] = tempRowIds[current];
                cumulative |= (tempMasks[current] << (m_SliceSize * cumulativeCounter++));
            }
            else
            {
                vset.rows[thread * noMasks + mask] = 0;
                ++cumulativeCounter;
            }
        }
        vset.masks[thread] = cumulative;
    }
}

void BRS::distributeSlices(std::vector<unsigned>& tempRowIds, std::vector<MASK>& tempMasks, std::vector<VSet>& vsets)
{
    unsigned noMasks = MASK_BITS / m_SliceSize;
    unsigned fullWork = WARP_SIZE * noMasks;

    unsigned noComplete = tempRowIds.size() / fullWork;
    for (unsigned complete = 0; complete < noComplete; ++complete)
    {
        VSet set;
        for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
        {
            MASK cumulative = 0;
            unsigned cumulativeCounter = 0;
            for (unsigned mask = 0; mask < noMasks; ++mask)
            {
                unsigned current = complete * fullWork + mask * WARP_SIZE + thread;
                set.rows.emplace_back(tempRowIds[current]);
                cumulative |= (tempMasks[current] << (m_SliceSize * cumulativeCounter++));
            }
            set.masks.emplace_back(cumulative);
        }
        vsets.emplace_back(set);
    }

    unsigned leftoverStart = noComplete * fullWork;
    if (leftoverStart < tempRowIds.size())
    {
        VSet set;

        if (m_IsFullPadding)
        {
            for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
            {
                MASK cumulative = 0;
                unsigned cumulativeCounter = 0;
                for (unsigned mask = 0; mask < noMasks; ++mask)
                {
                    unsigned current = leftoverStart + mask * WARP_SIZE + thread;
                    if (current < tempRowIds.size())
                    {
                        set.rows.emplace_back(tempRowIds[current]);
                        cumulative |= (tempMasks[current] << (m_SliceSize * cumulativeCounter++));
                    }
                    else
                    {
                        set.rows.emplace_back(0);
                        ++cumulativeCounter;
                    }
                }
                set.masks.emplace_back(cumulative);
            }
        }
        else
        {
            MASK cumulative = 0;
            unsigned cumulativeCounter = 0;
            for (unsigned slice = leftoverStart; slice < tempRowIds.size(); ++slice)
            {
                set.rows.emplace_back(tempRowIds[slice]);
                cumulative |= (tempMasks[slice] << (m_SliceSize * cumulativeCounter++));
                if (cumulativeCounter == noMasks)
                {
                    set.masks.emplace_back(cumulative);
                    cumulative = 0;
                    cumulativeCounter = 0;
                }
            }
            if (cumulativeCounter != 0)
            {
                for (; cumulativeCounter < noMasks; ++cumulativeCounter)
                {
                    set.rows.emplace_back(0);
                }
                set.masks.emplace_back(cumulative);
            }
        }

        vsets.emplace_back(set);
    }
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
    std::cout << "Min number of slices in any set " << min << std::endl;
    std::cout << "Max number of slices in any set " << max << std::endl;

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
    
    double averageVSet = 0;
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        SliceSetInformation set = vsetInformation[vset];
        averageVSet += set.noEntered;
    }
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

#endif
