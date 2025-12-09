#pragma once

#include "BitMatrix.cuh"
#include "CSC.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <map>

class BVSS: public BitMatrix
{
public:
    struct VSet
    {
        unsigned rset;
        std::vector<unsigned> rows;
        std::vector<MASK> masks;
    };

public:
    BVSS(unsigned sliceSize, unsigned noMasks, std::ofstream& file);
    BVSS(const BVSS& other) = delete;
    BVSS(BVSS&& other) noexcept = delete;
    BVSS& operator=(const BVSS& other) = delete;
    BVSS& operator=(BVSS&& other) noexcept = delete;
    virtual ~BVSS();

    void constructFromBinary(std::string filename);
    void saveToBinary(std::string filename);

    void constructFromCSCMatrix(CSC* csc);
    void printBVSSData();
    void bvssAnalysis();
    void kernelAnalysis(unsigned source, unsigned totalLevels, unsigned totalVisited, double time);

    [[nodiscard]] inline unsigned& getN() {return m_N;}
    [[nodiscard]] inline unsigned& getSliceSize() {return m_SliceSize;}
    [[nodiscard]] inline unsigned& getNoMasks() {return m_NoMasks;}
    [[nodiscard]] inline unsigned& getNoRealSliceSets() {return m_NoRealSliceSets;}
    [[nodiscard]] inline unsigned& getNoVirtualSliceSets() {return m_NoVirtualSliceSets;}
    [[nodiscard]] inline SLICE_TYPE& getNoSlices() {return m_NoSlices;}
    [[nodiscard]] inline SLICE_TYPE& getNoUnpaddedSlices() {return m_NoUnpaddedSlices;}
    [[nodiscard]] inline SLICE_TYPE& getNoPaddedSlices() {return m_NoPaddedSlices;}
    [[nodiscard]] inline double& getUpdateDivergence() {return m_UpdateDivergence;}
    [[nodiscard]] inline SLICE_TYPE*& getSliceSetPtrs() {return m_SliceSetPtrs;}
    [[nodiscard]] inline unsigned*& getVirtualToReal() {return m_VirtualToReal;}
    [[nodiscard]] inline unsigned*& getRealPtrs() {return m_RealPtrs;}
    [[nodiscard]] inline unsigned*& getRowIds() {return m_RowIds;}
    [[nodiscard]] inline MASK*& getMasks() {return m_Masks;}

private:
    void distributeSlices(const VSet& rset, std::vector<VSet>& vsets);
    double computeUpdateDivergence(const std::vector<VSet>& vsets);

private:
    unsigned m_N;
    unsigned m_SliceSize;
    unsigned m_NoMasks;
    unsigned m_NoRealSliceSets;
    unsigned m_NoVirtualSliceSets;
    SLICE_TYPE m_NoSlices;
    SLICE_TYPE m_NoUnpaddedSlices;
    SLICE_TYPE m_NoPaddedSlices;
    double m_UpdateDivergence;

    SLICE_TYPE* m_SliceSetPtrs;
    unsigned* m_VirtualToReal;
    unsigned* m_RealPtrs;
    unsigned* m_RowIds;
    MASK* m_Masks;

    // profiling data
    std::ofstream& m_File;
};

BVSS::BVSS(unsigned sliceSize, unsigned noMasks, std::ofstream& file)
: BitMatrix(),
  m_SliceSize(sliceSize),
  m_NoMasks(noMasks),
  m_File(file)
{
    if (m_SliceSize > MASK_BITS || K % m_SliceSize != 0)
    {
        throw std::runtime_error("Invalid slice size provided.");
    }
    if (m_NoMasks != (32 / m_SliceSize))
    {
        throw std::runtime_error("Invalid noMasks provided.");
    }

    m_File << "N\tNNZ\tUpdateDivergence\tSliceSize\t#RSet\t#VSet\t#Slices\tAvg(Slice/VSet)\tMin(Slice/VSet)\tMax(Slice/VSet)\tStd(Slice/Vset)\t#PaddedSlices\tUnpaddedSlices\tBitsTotal\tBitsUnpadded\tCompressionRatio" << std::endl;
}

BVSS::~BVSS()
{
    delete[] m_SliceSetPtrs;
    delete[] m_VirtualToReal;
    delete[] m_RealPtrs;
    delete[] m_RowIds;
    delete[] m_Masks;
}

void BVSS::saveToBinary(std::string filename)
{
    std::ofstream out(filename, std::ios::binary);

    out.write(reinterpret_cast<const char*>(&m_N), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_SliceSize), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_NoMasks), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_NoRealSliceSets), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_NoVirtualSliceSets), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_NoSlices), (sizeof(SLICE_TYPE)));
    out.write(reinterpret_cast<const char*>(&m_NoUnpaddedSlices), (sizeof(SLICE_TYPE)));
    out.write(reinterpret_cast<const char*>(&m_NoPaddedSlices), (sizeof(SLICE_TYPE)));
    out.write(reinterpret_cast<const char*>(&m_UpdateDivergence), (sizeof(double)));

    out.write(reinterpret_cast<const char*>(m_SliceSetPtrs), (sizeof(SLICE_TYPE) * (m_NoVirtualSliceSets + 1)));
    out.write(reinterpret_cast<const char*>(m_VirtualToReal), (sizeof(unsigned) * m_NoVirtualSliceSets));
    out.write(reinterpret_cast<const char*>(m_RealPtrs), (sizeof(unsigned) * (m_NoRealSliceSets + 1)));
    out.write(reinterpret_cast<const char*>(m_RowIds), (sizeof(unsigned) * m_NoSlices));
    out.write(reinterpret_cast<const char*>(m_Masks), (sizeof(MASK) * (m_NoSlices / m_NoMasks)));

    out.close();
}

void BVSS::constructFromBinary(std::string filename)
{
    std::ifstream in(filename, std::ios::binary);

    in.read(reinterpret_cast<char*>(&m_N), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_SliceSize), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_NoMasks), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_NoRealSliceSets), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_NoVirtualSliceSets), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_NoSlices), sizeof(SLICE_TYPE));
    in.read(reinterpret_cast<char*>(&m_NoUnpaddedSlices), sizeof(SLICE_TYPE));
    in.read(reinterpret_cast<char*>(&m_NoPaddedSlices), sizeof(SLICE_TYPE));
    in.read(reinterpret_cast<char*>(&m_UpdateDivergence), sizeof(double));

    m_SliceSetPtrs = new SLICE_TYPE[m_NoVirtualSliceSets + 1];
    in.read(reinterpret_cast<char*>(m_SliceSetPtrs), (sizeof(SLICE_TYPE) * (m_NoVirtualSliceSets + 1)));

    m_VirtualToReal = new unsigned[m_NoVirtualSliceSets];
    in.read(reinterpret_cast<char*>(m_VirtualToReal), (sizeof(unsigned) * m_NoVirtualSliceSets));

    m_RealPtrs = new unsigned[m_NoRealSliceSets + 1];
    in.read(reinterpret_cast<char*>(m_RealPtrs), (sizeof(unsigned) * (m_NoRealSliceSets + 1)));

    m_RowIds = new unsigned[m_NoSlices];
    in.read(reinterpret_cast<char*>(m_RowIds), (sizeof(unsigned) * m_NoSlices));

    m_Masks = new MASK[m_NoSlices / m_NoMasks];
    in.read(reinterpret_cast<char*>(m_Masks), (sizeof(MASK) * (m_NoSlices / m_NoMasks)));

    in.close();

    std::cout << "BVSS read from binary." << std::endl;

    this->printBVSSData();
    //this->bvssAnalysis();
}

void BVSS::constructFromCSCMatrix(CSC* csc)
{
    m_N = csc->getN();
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();

    m_NoRealSliceSets = (m_N + m_SliceSize - 1) / m_SliceSize;
    m_NoVirtualSliceSets = 0;

    double start = omp_get_wtime();
    std::vector<std::vector<VSet>> rsets(m_NoRealSliceSets);
    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        #pragma omp for
        for (unsigned rset = 0; rset < m_NoRealSliceSets; ++rset)
        {
            std::map<unsigned, MASK> map;
         
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
                }
    
                i = nextRow;
            }

            VSet realSet;
            realSet.rset = rset;
            for (const auto& slice: map)
            {
                realSet.rows.emplace_back(slice.first);
                realSet.masks.emplace_back(slice.second);
            }
    
            this->distributeSlices(realSet, rsets[rset]);
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
    m_UpdateDivergence = this->computeUpdateDivergence(vsets);
    m_NoVirtualSliceSets = vsets.size();

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

    m_SliceSetPtrs = new SLICE_TYPE[m_NoVirtualSliceSets + 1];
    m_VirtualToReal = new unsigned[m_NoVirtualSliceSets];
    
    m_SliceSetPtrs[0] = 0;
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        m_SliceSetPtrs[vset + 1] = m_SliceSetPtrs[vset] + vsets[vset].rows.size();
        m_VirtualToReal[vset] = vsets[vset].rset;
    }
    m_NoSlices = m_SliceSetPtrs[m_NoVirtualSliceSets];

    m_RowIds = new unsigned[m_NoSlices];
    SLICE_TYPE idx = 0;
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        for (unsigned i = 0; i < vsets[vset].rows.size(); ++i)
        {
            m_RowIds[idx++] = vsets[vset].rows[i];
        }
    }

    m_Masks = new MASK[(m_NoSlices / m_NoMasks)];
    idx = 0;
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        for (unsigned i = 0; i < vsets[vset].masks.size(); ++i)
        {
            m_Masks[idx++] = vsets[vset].masks[i];
        }
    }
    double end = omp_get_wtime();
    std::cout << "BVSS construction took: " << end - start << " seconds." << std::endl;
    
    this->printBVSSData();
    //this->bvssAnalysis();
}

void BVSS::distributeSlices(const VSet& rset, std::vector<VSet>& vsets)
{
    unsigned fullWork = WARP_SIZE * m_NoMasks;

    unsigned noComplete = rset.rows.size() / fullWork;
    for (unsigned complete = 0; complete < noComplete; ++complete)
    {
        VSet vset;
        vset.rset = rset.rset;
        for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
        {
            MASK cumulative = 0;
            unsigned cumulativeCounter = 0;
            for (unsigned mask = 0; mask < m_NoMasks; ++mask)
            {
                unsigned current = complete * fullWork + mask * WARP_SIZE + thread;
                vset.rows.emplace_back(rset.rows[current]);
                cumulative |= (rset.masks[current] << (m_SliceSize * cumulativeCounter++));
            }
            vset.masks.emplace_back(cumulative);
        }
        vsets.emplace_back(vset);
    }

    unsigned leftoverStart = noComplete * fullWork;
    if (leftoverStart < rset.rows.size())
    {
        VSet vset;
        vset.rset = rset.rset;

        if (FULL_PADDING)
        {
            for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
            {
                MASK cumulative = 0;
                unsigned cumulativeCounter = 0;
                for (unsigned mask = 0; mask < m_NoMasks; ++mask)
                {
                    unsigned current = leftoverStart + mask * WARP_SIZE + thread;
                    if (current < rset.rows.size())
                    {
                        vset.rows.emplace_back(rset.rows[current]);
                        cumulative |= (rset.masks[current] << (m_SliceSize * cumulativeCounter++));
                    }
                    else
                    {
                        vset.rows.emplace_back(UNSIGNED_MAX);
                        ++cumulativeCounter;
                    }
                }
                vset.masks.emplace_back(cumulative);
            }
        }
        else
        {
            unsigned stride = ((rset.rows.size() - leftoverStart) / m_NoMasks);
            for (unsigned thread = 0; thread < stride; ++thread)
            {
                MASK cumulative = 0;
                unsigned cumulativeCounter = 0;
                for (unsigned mask = 0; mask < m_NoMasks; ++mask)
                {
                    unsigned current = leftoverStart + mask * stride + thread;
                    vset.rows.emplace_back(rset.rows[current]);
                    cumulative |= (rset.masks[current] << (m_SliceSize * cumulativeCounter++));
                }
                vset.masks.emplace_back(cumulative);
            }
            leftoverStart += stride * m_NoMasks;
            if (leftoverStart < rset.rows.size())
            {
                MASK cumulative = 0;
                unsigned cumulativeCounter = 0;
                for (unsigned mask = 0; mask < m_NoMasks; ++mask)
                {
                    unsigned current = leftoverStart + mask;
                    if (current < rset.rows.size())
                    {
                        vset.rows.emplace_back(rset.rows[current]);
                        cumulative |= (rset.masks[current] << (m_SliceSize * cumulativeCounter++));
                    }
                    else
                    {
                        vset.rows.emplace_back(UNSIGNED_MAX);
                        ++cumulativeCounter;
                    }
                }
                vset.masks.emplace_back(cumulative);
            }
        }
        vsets.emplace_back(vset);
    }
}

double BVSS::computeUpdateDivergence(const std::vector<VSet>& vsets)
{
    double updateDivergence = 0;
    for (const auto& vset: vsets)
    {
        double vsetMean = 0;
        for (unsigned mask = 0; mask < m_NoMasks; ++mask)
        {
            double mean = 0;
            unsigned count = 0;
            double stdev = 0;
            for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
            {
                unsigned slice = thread * m_NoMasks + mask;
                if (slice < vset.rows.size() && vset.rows[slice] != UNSIGNED_MAX)
                {
                    unsigned row = vset.rows[slice];
                    mean += row;
                    ++count;
                }
            }
            if (count != 0)
            {   
                mean /= count;
            }
            for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
            {
                unsigned slice = thread * m_NoMasks + mask;
                if (slice < vset.rows.size() && vset.rows[slice] != UNSIGNED_MAX)
                {
                    unsigned row = vset.rows[slice];
                    stdev += ((row - mean) * (row - mean));
                }
            }
            if (count != 0)
            {
                stdev /= count;
            }
            stdev = std::sqrt(stdev);
            vsetMean += stdev;
        }
        vsetMean /= m_NoMasks;
        updateDivergence += vsetMean;
    }
    updateDivergence /= vsets.size();

    return updateDivergence;
}

void BVSS::printBVSSData()
{
    std::cout << "MASK size: " << MASK_BITS << std::endl;
    std::cout << "Slice size: " << m_SliceSize << std::endl;
    std::cout << "Number of real slice sets: " << m_NoRealSliceSets << std::endl;
    std::cout << "Number of virtual slice sets: " << m_NoVirtualSliceSets << std::endl;
    std::cout << "Number of slices: " << m_NoSlices << std::endl;
    std::cout << "Number of slices in each mask: " << m_NoMasks << std::endl;

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

    MASK CONCEALER = (m_SliceSize == 32) ? static_cast<MASK>(0xFFFFFFFF) : ((static_cast<MASK>(1) << m_SliceSize) - 1);

    m_NoPaddedSlices = 0;
    unsigned totalNumberOfEdgesCheck = 0;
    for (unsigned vset = 0; vset < m_NoVirtualSliceSets; ++vset)
    {
        for (unsigned ptr = m_SliceSetPtrs[vset] / m_NoMasks; ptr < m_SliceSetPtrs[vset + 1] / m_NoMasks; ++ptr)
        {
            MASK current = m_Masks[ptr];
            for (unsigned mask = 0; mask < m_NoMasks; ++mask)
            {
                unsigned shift = mask * m_SliceSize;
                MASK pattern = ((current >> shift) & CONCEALER);
                unsigned edges = __builtin_popcount(pattern);
                if (edges == 0)
                {
                    ++m_NoPaddedSlices;
                }
                totalNumberOfEdgesCheck += edges;
            }
        }
    }
    std::cout << "Update divergence is: " << m_UpdateDivergence << std::endl;

    m_NoUnpaddedSlices = m_NoSlices - m_NoPaddedSlices;
    SLICE_TYPE bitsTotal = m_NoSlices * m_SliceSize;
    SLICE_TYPE bitsUnpadded = m_NoUnpaddedSlices * m_SliceSize;
    double compressionRatio = ((static_cast<double>(totalNumberOfEdgesCheck) / m_NoUnpaddedSlices) / m_SliceSize);
    std::cout << "Total unpadded slices: " << m_NoUnpaddedSlices << std::endl;
    std::cout << "Total padded slices: " << m_NoPaddedSlices << std::endl;
    std::cout << "Total connectivity bits used by the data structure: " << bitsTotal << std::endl;
    std::cout << "Total connectivity bits used by the unpadded part of the data structure: " << bitsUnpadded << std::endl;
    std::cout << "Compression ratio: " << compressionRatio << std::endl;
    std::cout << "Check: " << totalNumberOfEdgesCheck << std::endl;

    fileFlush(m_File, m_N); fileFlush(m_File, totalNumberOfEdgesCheck); fileFlush(m_File, m_UpdateDivergence); fileFlush(m_File, m_SliceSize); fileFlush(m_File, m_NoRealSliceSets); fileFlush(m_File, m_NoVirtualSliceSets); fileFlush(m_File, m_NoSlices);
    fileFlush(m_File, average); fileFlush(m_File, min); fileFlush(m_File, max); fileFlush(m_File, standardDeviation);
    fileFlush(m_File, m_NoPaddedSlices); fileFlush(m_File, m_NoUnpaddedSlices); fileFlush(m_File, bitsTotal); fileFlush(m_File, bitsUnpadded); fileFlush(m_File, compressionRatio);
    m_File << std::endl;
    m_File << "Source\tTotalLevels\tTotalVisited\tTime(ms)" << std::endl;
}

void BVSS::bvssAnalysis()
{
    MASK CONCEALER = (m_SliceSize == 32) ? static_cast<MASK>(0xFFFFFFFF) : ((static_cast<MASK>(1) << m_SliceSize) - 1);

    unsigned noPatterns = static_cast<unsigned>(CONCEALER);
    std::vector<unsigned> patterns(noPatterns, 0);
    std::vector<unsigned> counts(m_SliceSize + 1, 0);
    for (unsigned rset = 0; rset < m_NoRealSliceSets; ++rset)
    {
        for (unsigned vset = m_RealPtrs[rset]; vset < m_RealPtrs[rset + 1]; ++vset)
        {
            for (SLICE_TYPE slice = m_SliceSetPtrs[vset]; slice < m_SliceSetPtrs[vset + 1]; ++slice)
            {
                MASK current = m_Masks[slice / m_NoMasks];
                unsigned shift = (slice % m_NoMasks) * m_SliceSize;
                MASK pattern = ((current >> shift) & CONCEALER);
                if (pattern != 0)
                {
                    ++counts[__builtin_popcount(pattern)];
                    ++patterns[pattern];
                }
            }
        }
    }

    for (unsigned i = 1; i <= m_SliceSize; ++i)
    {
        fileFlush(m_File, std::to_string(i) + " Bits");
    }
    m_File << std::endl;
    for (unsigned i = 1; i <= m_SliceSize; ++i)
    {
        fileFlush(m_File, counts[i]);
    }
    m_File << std::endl;
}

void BVSS::kernelAnalysis(unsigned source, unsigned totalLevels, unsigned totalVisited, double time)
{
    fileFlush(m_File, source); fileFlush(m_File, totalLevels); fileFlush(m_File, totalVisited); fileFlush(m_File, time * 1000);
    m_File << std::endl;
}
