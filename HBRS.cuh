#ifndef HBRS_CUH
#define HBRS_CUH

#include "BitMatrix.cuh"
#include "CSC.cuh"
#include <algorithm>
#include <cassert>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <stdexcept>
#include <vector>
#include <unordered_map>

class HBRS: public BitMatrix
{
private:
    struct Encoding
    {
        MASK mask;
        std::vector<unsigned> rowIds;
    };

    struct LocalityStatistics
    {
        double wordPerMemory_avg = 0;
        double encodingSize_avg = 0;
        double encodingSize_dev = 0;
        unsigned complete = 0;
        unsigned incomplete = 0;
    };

public:
    HBRS(unsigned sliceSize = 32, unsigned sliceSizeHub = 32);
    HBRS(const HBRS& other) = delete;
    HBRS(HBRS&& other) noexcept = delete;
    HBRS& operator=(const HBRS& other) = delete;
    HBRS& operator=(HBRS&& other) noexcept = delete;
    virtual ~HBRS();

    void constructFromCSCMatrix(CSC* csc, unsigned k);

    // Normal
    [[nodiscard]] inline unsigned getN() {return m_N;}
    [[nodiscard]] inline unsigned getSliceSize() {return m_SliceSize;}
    [[nodiscard]] inline unsigned getNoSliceSets() {return m_NoSliceSets;}

    [[nodiscard]] inline unsigned* getSliceSetPtrs() {return m_SliceSetPtrs;}
    [[nodiscard]] inline unsigned* getRowIds() {return m_RowIds;}
    [[nodiscard]] inline MASK* getMasks() {return m_Masks;}

    // Hub
    [[nodiscard]] inline unsigned getN_Hub() {return m_N_Hub;}
    [[nodiscard]] inline unsigned getSliceSize_Hub() {return m_SliceSize_Hub;}
    [[nodiscard]] inline unsigned getNoSliceSets_Hub() {return m_NoSliceSets_Hub;}

    [[nodiscard]] inline unsigned* getSliceSetPtrs_Hub() {return m_SliceSetPtrs_Hub;}
    [[nodiscard]] inline unsigned* getEncodingPtrs_Hub() {return m_EncodingPtrs_Hub;}
    [[nodiscard]] inline unsigned* getRowIds_Hub() {return m_RowIds_Hub;}
    [[nodiscard]] inline MASK* getMasks_Hub() {return m_Masks_Hub;}

private:
    void constructNormalFromCSCMatrix(CSC* csc);
    void printHBRSData();

    void constructHubFromCSCMatrix(CSC* csc);
    void printHBRSHubData();
    
    void bucketDistribution(std::vector<Encoding>& encodings, unsigned max);
    LocalityStatistics distributeEncodingsToWarp(std::vector<Encoding>& encodings, std::vector<MASK>& masks, std::vector<unsigned>& encodingSizes, std::vector<unsigned>& rowIds, unsigned noMasks, unsigned sliceSize);
    LocalityStatistics computeWarpLocalityStatistics(std::vector<Encoding>& warp, unsigned noMasks);

private:
    // Normal
    unsigned m_N;
    unsigned m_SliceSize;
    unsigned m_NoSliceSets;
    
    unsigned* m_SliceSetPtrs;
    unsigned* m_RowIds;
    MASK* m_Masks;

    // Hub
    unsigned m_N_Hub;
    unsigned m_SliceSize_Hub;
    unsigned m_NoSliceSets_Hub;

    unsigned* m_SliceSetPtrs_Hub;
    unsigned* m_EncodingPtrs_Hub;
    unsigned* m_RowIds_Hub;
    MASK* m_Masks_Hub;
};

HBRS::HBRS(unsigned sliceSize, unsigned sliceSizeHub)
: BitMatrix(),
  m_SliceSize(sliceSize),
  m_SliceSize_Hub(sliceSizeHub)
{
}

HBRS::~HBRS()
{
    delete[] m_SliceSetPtrs;
    delete[] m_RowIds;
    delete[] m_Masks;

    delete[] m_SliceSetPtrs_Hub;
    delete[] m_EncodingPtrs_Hub;
    delete[] m_RowIds_Hub;
    delete[] m_Masks_Hub;
}

void HBRS::constructFromCSCMatrix(CSC* csc, unsigned k)
{
    unsigned kAligned = k + ((m_SliceSize_Hub - (k % m_SliceSize_Hub)) % m_SliceSize_Hub);
    kAligned = std::min(kAligned, csc->getN());
    k = csc->getN();
    m_N_Hub = kAligned;
    m_N = csc->getN() - kAligned;

    std::cout << "Hub range: 0 - " << m_N_Hub << std::endl;
    std::cout << "Normal range: " << m_N_Hub << " - " << csc->getN() << std::endl;

    this->constructHubFromCSCMatrix(csc);
    this->constructNormalFromCSCMatrix(csc);
}

void HBRS::constructNormalFromCSCMatrix(CSC* csc)
{
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();

    m_NoSliceSets = (m_N + m_SliceSize - 1) / m_SliceSize;
    m_SliceSetPtrs = new unsigned[m_NoSliceSets + 1];
    std::fill(m_SliceSetPtrs, m_SliceSetPtrs + m_NoSliceSets + 1, 0);

    if (m_SliceSize > MASK_BITS || K % m_SliceSize != 0 || MASK_BITS % m_SliceSize != 0)
    {
        throw std::runtime_error("Invalid slice size provided for normal slice sets.");
    }
    unsigned noMasks = MASK_BITS / m_SliceSize;

    std::vector<std::vector<unsigned>> rowIds(m_NoSliceSets);
    std::vector<std::vector<MASK>> masks(m_NoSliceSets);

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet)
    {
        unsigned sliceSetColStart = m_N_Hub + sliceSet * m_SliceSize;
        unsigned sliceSetColEnd = std::min(m_N_Hub + m_N, sliceSetColStart + m_SliceSize);
        
        std::vector<unsigned> ptrs(sliceSetColEnd - sliceSetColStart);
        for (unsigned j = sliceSetColStart; j < sliceSetColEnd; ++j)
        {
            ptrs[j - sliceSetColStart] = colPtrs[j]; // do note that for this approach to work out, adjacency list must be sorted
        }

        std::vector<Encoding> encodings;

        unsigned i = 0;
        while (i < csc->getN()) 
        {
            MASK individual = 0;
            unsigned nextRow = csc->getN();

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
                Encoding encoding;
                encoding.mask = individual;
                encoding.rowIds.emplace_back(i);
                encodings.emplace_back(encoding);
            }

            i = nextRow;
        }

        std::vector<unsigned> encodingSizes;
        LocalityStatistics stats = this->distributeEncodingsToWarp(encodings, masks[sliceSet], encodingSizes, rowIds[sliceSet], noMasks, m_SliceSize);
        m_SliceSetPtrs[sliceSet + 1] = rowIds[sliceSet].size();

        assert(rowIds[sliceSet].size() == masks[sliceSet].size() * noMasks);
    }

    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets; ++sliceSet) 
    {
        m_SliceSetPtrs[sliceSet + 1] += m_SliceSetPtrs[sliceSet];
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

    this->printHBRSData();
}

void HBRS::constructHubFromCSCMatrix(CSC* csc)
{
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();

    if (m_SliceSize_Hub > MASK_BITS || K % m_SliceSize_Hub != 0 || MASK_BITS % m_SliceSize_Hub != 0)
    {
        throw std::runtime_error("Invalid slice size provided for hub slice sets.");
    }
    unsigned noMasks = MASK_BITS / m_SliceSize_Hub;

    m_NoSliceSets_Hub = (m_N_Hub + m_SliceSize_Hub - 1) / m_SliceSize_Hub;
    m_SliceSetPtrs_Hub = new unsigned[m_NoSliceSets_Hub + 1];
    std::fill(m_SliceSetPtrs_Hub, m_SliceSetPtrs_Hub + m_NoSliceSets_Hub + 1, 0);

    std::vector<std::vector<unsigned>> rowIds(m_NoSliceSets_Hub);
    std::vector<std::vector<unsigned>> encodingSizes(m_NoSliceSets_Hub);
    std::vector<std::vector<MASK>> masks(m_NoSliceSets_Hub);

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets_Hub; ++sliceSet)
    {
        unsigned sliceSetColStart = sliceSet * m_SliceSize_Hub;
        unsigned sliceSetColEnd = std::min(m_N_Hub, sliceSetColStart + m_SliceSize_Hub);
        
        std::vector<unsigned> ptrs(sliceSetColEnd - sliceSetColStart);
        for (unsigned j = sliceSetColStart; j < sliceSetColEnd; ++j)
        {
            ptrs[j - sliceSetColStart] = colPtrs[j]; // do note that for this approach to work out, adjacency list must be sorted
        }

        std::vector<std::vector<unsigned>> patterns(1 << m_SliceSize_Hub);

        unsigned i = 0;
        while (i < csc->getN()) 
        {
            MASK individual = 0;
            unsigned nextRow = csc->getN();

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
                patterns[individual].emplace_back(i);
            }

            i = nextRow;
        }

        std::vector<Encoding> encodings;
        unsigned max = 0;

        for (unsigned pattern = 1; pattern < patterns.size(); ++pattern)
        {
            if (patterns[pattern].empty()) continue;

            Encoding encoding;
            encoding.mask = static_cast<MASK>(pattern);
            for (unsigned i = 0; i < patterns[pattern].size(); ++i)
            {
                max = std::max(max, patterns[pattern][i]);
                encoding.rowIds.emplace_back(patterns[pattern][i]);
            }
            encodings.emplace_back(encoding);
        }

        this->bucketDistribution(encodings, max + 1);
        LocalityStatistics stats = this->distributeEncodingsToWarp(encodings, masks[sliceSet], encodingSizes[sliceSet], rowIds[sliceSet], noMasks, m_SliceSize_Hub);
        if (!encodings.empty())
        {
            //printf("Slice Set: %u, No Complete Work: %u, No Incomplete Work: %u, Average word per memory request: %f, Average encoding size: %f, Std encoding size: %f\n", sliceSet, stats.complete, stats.incomplete, stats.wordPerMemory_avg, stats.encodingSize_avg, stats.encodingSize_dev);
        }
        m_SliceSetPtrs_Hub[sliceSet + 1] = encodingSizes[sliceSet].size();

        assert(encodingSizes[sliceSet].size() == masks[sliceSet].size() * noMasks);
    }

    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets_Hub; ++sliceSet) 
    {
        m_SliceSetPtrs_Hub[sliceSet + 1] += m_SliceSetPtrs_Hub[sliceSet]; // you get encoding from slice set
    }

    unsigned noUniqueSlices = m_SliceSetPtrs_Hub[m_NoSliceSets_Hub];

    m_Masks_Hub = new MASK[noUniqueSlices / noMasks];
    unsigned idx = 0;
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets_Hub; ++sliceSet)
    {
        for (unsigned i = 0; i < masks[sliceSet].size(); ++i)
        {
            m_Masks_Hub[idx++] = masks[sliceSet][i]; // you get mask from (slice set / noMasks) -- mask of the corresponding encoding
        }
    }

    m_EncodingPtrs_Hub = new unsigned[noUniqueSlices + 1];
    std::fill(m_EncodingPtrs_Hub, m_EncodingPtrs_Hub + noUniqueSlices + 1, 0);
    idx = 0;
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets_Hub; ++sliceSet)
    {
        for (unsigned e = 0; e < encodingSizes[sliceSet].size(); ++e)
        {
            m_EncodingPtrs_Hub[idx + 1] = (m_EncodingPtrs_Hub[idx] + encodingSizes[sliceSet][e]); // you get slice from encoding
            ++idx;
        }
    }

    unsigned noSlices = m_EncodingPtrs_Hub[m_SliceSetPtrs_Hub[m_NoSliceSets_Hub]];

    m_RowIds_Hub = new unsigned[noSlices];
    idx = 0;
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets_Hub; ++sliceSet)
    {
        for (unsigned i = 0; i < rowIds[sliceSet].size(); ++i)
        {
            m_RowIds_Hub[idx++] = rowIds[sliceSet][i]; // you get row from slice
        }
    }

    this->printHBRSHubData();
}

void HBRS::bucketDistribution(std::vector<Encoding>& encodings, unsigned max)
{
    const unsigned BUCKET_RANGE = max;
    const unsigned MAX_ENCODING_SIZE = 32;

    const unsigned numBuckets = (max + BUCKET_RANGE - 1) / BUCKET_RANGE;
    std::vector<std::vector<Encoding>> buckets(numBuckets);

    for (const auto& enc: encodings)
    {
        const auto& rows = enc.rowIds;
        unsigned i = 0;
        while (i < rows.size())
        {
            const unsigned bucketId = rows[i] / BUCKET_RANGE;

            Encoding part;
            part.mask = enc.mask;

            while (i < rows.size() && (rows[i] / BUCKET_RANGE) == bucketId)
            {
                part.rowIds.emplace_back(rows[i]);
                ++i;
            }
            buckets[bucketId].emplace_back(std::move(part));
        }
    }

    std::vector<Encoding> out;
    out.reserve(encodings.size());
    for (const auto& bucket: buckets)
    {
        for (auto& e: bucket)
        {
            if (e.rowIds.size() > MAX_ENCODING_SIZE)
            {
                unsigned noPartition = (e.rowIds.size() + MAX_ENCODING_SIZE - 1) / MAX_ENCODING_SIZE;
                for (unsigned part = 0; part < noPartition; ++part)
                {
                    unsigned start = part * MAX_ENCODING_SIZE;
                    unsigned end = std::min(start + MAX_ENCODING_SIZE, static_cast<unsigned>(e.rowIds.size()));
                    Encoding newEncoding;
                    newEncoding.mask = e.mask;
                    for (unsigned i = start; i < end; ++i)
                    {
                        newEncoding.rowIds.emplace_back(e.rowIds[i]);
                    }
                    out.emplace_back(newEncoding);
                }
            }
            else
            {
                out.emplace_back(e);
            }
        }
    }

    encodings.swap(out);
}

HBRS::LocalityStatistics HBRS::computeWarpLocalityStatistics(std::vector<Encoding>& warp, unsigned noMasks)
{
    LocalityStatistics stats;

    unsigned encCount = static_cast<unsigned>(warp.size());

    const auto idxOk = [&](unsigned mask, unsigned lane)
    {
        unsigned idx = mask * WARP_SIZE + lane;
        return idx < encCount;
    };

    double sum = 0;
    double sumsq = 0;
    for (unsigned mask = 0; mask < noMasks; ++mask)
    {
        for (unsigned lane = 0; lane < WARP_SIZE; ++lane)
        {
            if (!idxOk(mask, lane)) continue;
            unsigned idx = mask * WARP_SIZE + lane;
            double sz = static_cast<double>(warp[idx].rowIds.size());
            sum += sz;
            sumsq += sz * sz;
        }
    }
    double count = static_cast<double>(encCount);
    if (count > 0)
    {
        stats.encodingSize_avg = sum / count;
        double var = sumsq / count - stats.encodingSize_avg * stats.encodingSize_avg;
        stats.encodingSize_dev = std::sqrt(var);
    }

    unsigned wordCount = 0;
    unsigned maxLen = 0;
    for (unsigned i = 0; i < encCount; ++i)
    {
        maxLen = std::max(maxLen, static_cast<unsigned>(warp[i].rowIds.size()));
    }

    for (unsigned mask = 0; mask < noMasks; ++mask)
    {
        for (unsigned i = 0; i < maxLen; ++i)
        {
            std::unordered_map<unsigned, unsigned> freq;
            bool any = false;
            for (unsigned lane = 0; lane < WARP_SIZE; ++lane)
            {
                if (!idxOk(mask, lane)) continue;
                unsigned idx = mask * WARP_SIZE + lane;
                const auto& rows = warp[idx].rowIds;
                if (i < rows.size())
                {
                    any = true;
                    unsigned word = rows[i] / MASK_BITS;
                    if (freq.contains(word))
                    {
                        ++freq[word];
                    }
                    else
                    {
                        freq[word] = 1;
                    }
                }
            }
            if (any) 
            {
                stats.wordPerMemory_avg += freq.size();
                ++wordCount;
            }
        }
    }
    if (wordCount > 0)
    {
        stats.wordPerMemory_avg /= wordCount;
    }
    stats.complete = (warp.size() == WARP_SIZE * noMasks);
    stats.incomplete = !stats.complete;

    return stats;
}

HBRS::LocalityStatistics HBRS::distributeEncodingsToWarp(std::vector<Encoding>& encodings, std::vector<MASK>& masks, std::vector<unsigned>& encodingSizes, std::vector<unsigned>& rowIds, unsigned noMasks, unsigned sliceSize)
{
    std::vector<LocalityStatistics> statistics;

    std::vector<Encoding> distributed;

    unsigned gridSize = WARP_SIZE * noMasks;
    std::vector<Encoding> warp;
    for (unsigned encoding = 0; encoding < encodings.size(); ++encoding)
    {
        warp.emplace_back(encodings[encoding]);
        if (warp.size() == gridSize)
        {
            statistics.emplace_back(this->computeWarpLocalityStatistics(warp, noMasks));
            for (unsigned lane = 0; lane < WARP_SIZE; ++lane)
            {
                for (unsigned mask = 0; mask < noMasks; ++mask)
                {
                    unsigned current = mask * WARP_SIZE + lane;
                    distributed.emplace_back(warp[current]);
                }
            }
            warp.clear();
        }
    }
    if (!warp.empty())
    {
        statistics.emplace_back(this->computeWarpLocalityStatistics(warp, noMasks));
        for (unsigned lane = 0; lane < WARP_SIZE; ++lane)
        {
            /*
            if (lane >= warp.size())
            {
                break;
            }
            */
            for (unsigned mask = 0; mask < noMasks; ++mask)
            {
                unsigned current = mask * WARP_SIZE + lane;
                if (current < warp.size())
                {
                    distributed.emplace_back(warp[current]);
                }
                else
                {
                    Encoding emptyEncoding;
                    emptyEncoding.mask = static_cast<MASK>(0);
                    emptyEncoding.rowIds.emplace_back(0);
                    distributed.emplace_back(emptyEncoding);
                }
            }
        }
    }
    assert(distributed.size() % (WARP_SIZE * noMasks) == 0);

    MASK cumulative = 0;
    unsigned cumulativeCounter = 0;
    for (unsigned encoding = 0; encoding < distributed.size(); ++encoding)
    {
        cumulative |= (static_cast<MASK>(distributed[encoding].mask << (sliceSize * cumulativeCounter++)));
        encodingSizes.emplace_back(distributed[encoding].rowIds.size());
        const std::vector<unsigned>& rows = distributed[encoding].rowIds;
        for (unsigned i = 0; i < encodingSizes.back(); ++i)
        {
            rowIds.emplace_back(rows[i]);
        }
        if (cumulativeCounter == noMasks)
        {
            masks.emplace_back(cumulative);
            cumulative = 0;
            cumulativeCounter = 0;
        }
    }

    assert(cumulativeCounter == 0);

    LocalityStatistics total;
    for (const auto& s: statistics) 
    {
        total.wordPerMemory_avg += s.wordPerMemory_avg;
        total.encodingSize_avg += s.encodingSize_avg;
        total.encodingSize_dev += s.encodingSize_dev;
        total.complete += s.complete;
        total.incomplete += s.incomplete;
    }
    if (!statistics.empty()) 
    {
        total.wordPerMemory_avg /= statistics.size();
        total.encodingSize_avg /= statistics.size();
        total.encodingSize_dev /= statistics.size();
    }

    return total;
}

void HBRS::printHBRSData()
{
    std::cout << "************ PRINTING NORMAL DATA ************" << std::endl;

    unsigned noMasks = MASK_BITS / m_SliceSize;

    std::cout << "MASK size: " << MASK_BITS << std::endl;
    std::cout << "Slice size: " << m_SliceSize << std::endl;
    std::cout << "Number of slice sets: " << m_NoSliceSets << std::endl;
    std::cout << "Number of slices: " << m_SliceSetPtrs[m_NoSliceSets] << std::endl;
    std::cout << "Number of slices in each mask: " << noMasks << std::endl;

    unsigned empty = 0;
    double average = 0;
    for (unsigned ss = 0; ss < m_NoSliceSets; ++ss)
    {
        if (m_SliceSetPtrs[ss + 1] == m_SliceSetPtrs[ss])
        {
            ++empty;
        }
        average += (m_SliceSetPtrs[ss + 1] - m_SliceSetPtrs[ss]);
    }
    average /= m_NoSliceSets;
    std::cout << "Number of empty slice sets in the normal partition: " << empty << std::endl;
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

    unsigned totalNumberOfEdgesCheck = 0;
    unsigned noChunks = 10;
    unsigned chunkSize = (m_NoSliceSets + noChunks - 1) / noChunks;
    for (unsigned q = 0; q < noChunks; ++q)
    {
        bool chunkEmpty = true;
        unsigned start = q * chunkSize;
        unsigned end = std::min(m_NoSliceSets, start + chunkSize);

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
    std::cout << "Bits (total): " << m_SliceSetPtrs[m_NoSliceSets] * m_SliceSize << std::endl;

    std::cout << "Check: " << totalNumberOfEdgesCheck << std::endl;
}

void HBRS::printHBRSHubData()
{
    std::cout << "************ PRINTING HUB DATA ************" << std::endl;

    unsigned noMasks = MASK_BITS / m_SliceSize_Hub;
    unsigned noEncodings = m_SliceSetPtrs_Hub[m_NoSliceSets_Hub];
    unsigned noSlices = m_EncodingPtrs_Hub[m_SliceSetPtrs_Hub[m_NoSliceSets_Hub]];

    std::cout << "MASK size: " << MASK_BITS << std::endl;
    std::cout << "Slice size: " << m_SliceSize_Hub << std::endl;
    std::cout << "Number of slice sets: " << m_NoSliceSets_Hub << std::endl;
    std::cout << "Number of encodings: " << noEncodings << std::endl;
    std::cout << "Number of slices: " << noSlices << std::endl;
    std::cout << "Number of slices in each mask: " << noMasks << std::endl;

    unsigned empty = 0;
    double averageSetCount = 0;
    double averageEncodingSize = 0;
    for (unsigned ss = 0; ss < m_NoSliceSets_Hub; ++ss)
    {
        if (m_SliceSetPtrs_Hub[ss + 1] == m_SliceSetPtrs_Hub[ss])
        {
            ++empty;
        }
        for (unsigned encoding = m_SliceSetPtrs_Hub[ss]; encoding < m_SliceSetPtrs_Hub[ss + 1]; ++encoding)
        {
            averageEncodingSize += (m_EncodingPtrs_Hub[encoding + 1] - m_EncodingPtrs_Hub[encoding]);
        }
        averageSetCount += (m_SliceSetPtrs_Hub[ss + 1] - m_SliceSetPtrs_Hub[ss]);
    }
    averageSetCount /= m_NoSliceSets_Hub;
    averageEncodingSize /= noEncodings;
    std::cout << "Number of empty slice sets in the hub partition: " << empty << std::endl;

    double setCountVariance = 0;
    double encodingSizeVariance = 0;
    for (unsigned ss = 0; ss < m_NoSliceSets_Hub; ++ss)
    {
        for (unsigned encoding = m_SliceSetPtrs_Hub[ss]; encoding < m_SliceSetPtrs_Hub[ss + 1]; ++encoding)
        {
            unsigned len = m_EncodingPtrs_Hub[encoding + 1] - m_EncodingPtrs_Hub[encoding];
            double diff = len - averageEncodingSize;
            encodingSizeVariance += diff * diff;
        }
        unsigned setLen = m_SliceSetPtrs_Hub[ss + 1] - m_SliceSetPtrs_Hub[ss];
        double diffSet = setLen - averageSetCount;
        setCountVariance += diffSet * diffSet;
    }
    setCountVariance /= m_NoSliceSets_Hub;
    encodingSizeVariance /= noEncodings;

    double setCountStd = std::sqrt(setCountVariance);
    double encodingSizeStd = std::sqrt(encodingSizeVariance);

    std::cout << "Average number of encodings in each set: " << averageSetCount << std::endl;
    std::cout << "Standard deviation of the number of encodings in each set: " << setCountStd << std::endl;
    std::cout << "Average encoding size: " << averageEncodingSize << std::endl;
    std::cout << "Standard deviation of the encoding size: " << encodingSizeStd << std::endl;

    unsigned totalNumberOfEdgesCheck = 0;
    unsigned noChunks = 10;
    unsigned chunkSize = (m_NoSliceSets_Hub + noChunks - 1) / noChunks;
    MASK andMask = (0xFFFFFFFF >> (MASK_BITS - m_SliceSize_Hub));
    for (unsigned q = 0; q < noChunks; ++q)
    {
        bool chunkEmpty = true;
        unsigned start = q * chunkSize;
        unsigned end = std::min(m_NoSliceSets_Hub, start + chunkSize);

        unsigned long long baselineBitsChunk = 0;
        unsigned long long maskBitsChunk = 0;

        for (unsigned set = start; set < end; ++set)
        {
            unsigned encodingsInSet = m_SliceSetPtrs_Hub[set + 1] - m_SliceSetPtrs_Hub[set];
            unsigned slicesInSet = m_EncodingPtrs_Hub[m_SliceSetPtrs_Hub[set + 1]] - m_EncodingPtrs_Hub[m_SliceSetPtrs_Hub[set]];

            for (unsigned encoding = m_SliceSetPtrs_Hub[set]; encoding < m_SliceSetPtrs_Hub[set + 1]; ++encoding)
            {
                unsigned word = encoding / noMasks;
                unsigned bit = encoding % noMasks;
                unsigned edge = __builtin_popcount((m_Masks_Hub[word] >> (bit * m_SliceSize_Hub)) & andMask);

                totalNumberOfEdgesCheck += (edge * (m_EncodingPtrs_Hub[encoding + 1] - m_EncodingPtrs_Hub[encoding]));
            }

            baselineBitsChunk += (slicesInSet * m_SliceSize_Hub);
            maskBitsChunk += (encodingsInSet * m_SliceSize_Hub);
            chunkEmpty &= (m_SliceSetPtrs_Hub[set + 1] == m_SliceSetPtrs_Hub[set]);
        }

        double chunkEff = double(baselineBitsChunk) / maskBitsChunk;
        if (!chunkEmpty)
        {
            std::cout << "Chunk: " << q
            << " - Encoding count: " << m_SliceSetPtrs_Hub[end] - m_SliceSetPtrs_Hub[start]
            << " - Slice count: " << m_EncodingPtrs_Hub[m_SliceSetPtrs_Hub[end]] - m_EncodingPtrs_Hub[m_SliceSetPtrs_Hub[start]]
            << " - Baseline bits: " << baselineBitsChunk
            << " - Stored mask bits: " << maskBitsChunk
            << " - Compression efficiency: x" << chunkEff << std::endl;
        }
    }

    unsigned long long baselineBits = noSlices * m_SliceSize_Hub;
    unsigned long long storedMaskBits = noEncodings * m_SliceSize_Hub;
    double compressionEff = double(baselineBits) / storedMaskBits;

    std::cout << "Baseline bits (total): " << baselineBits << std::endl;
    std::cout << "Stored mask bits (total): " << storedMaskBits << std::endl;
    std::cout << "Compression efficiency (baseline / stored): x" << compressionEff << std::endl;

    std::cout << "Check: " << totalNumberOfEdgesCheck << std::endl;
}

#endif
