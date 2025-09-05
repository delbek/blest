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
#include <tuple>

class HBRS: public BitMatrix
{
private:
    typedef std::tuple<MASK, unsigned, std::vector<unsigned>, std::unordered_map<unsigned, unsigned>> Encoding;

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
    
    void warpAlign(std::vector<Encoding>& warp, unsigned noMasks);
    void distributeEncodingsToWarp(std::vector<Encoding>& encodings, unsigned noMasks);
    double euclideanDistance(std::unordered_map<unsigned, unsigned>& hash1, std::unordered_map<unsigned, unsigned>& hash2);
    void hubEncodingOrderedDistribution(std::vector<MASK>& tempMasks, std::vector<unsigned>& tempEncodingSizes, std::vector<unsigned>& tempRowIds, std::vector<MASK>& masks, std::vector<unsigned>& encodingSizes, std::vector<unsigned>& rowIds, unsigned noMasks, unsigned& total);

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
    for (; k % m_SliceSize_Hub != 0; ++k);

    m_N = (csc->getN() - k);
    m_N_Hub = k;

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
    unsigned noWarpSlice = (K / m_SliceSize) * 8;

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

        std::vector<unsigned> tempRowIds;
        std::vector<MASK> tempMasks;

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

        std::vector<std::vector<unsigned>> encodings(1 << m_SliceSize_Hub);

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
                encodings[individual].emplace_back(i);
            }

            i = nextRow;
        }

        unsigned total = 0;
        std::vector<unsigned> tempRowIds;
        std::vector<unsigned> tempEncodingSizes;
        std::vector<MASK> tempMasks;
        for (unsigned encoding = 1; encoding < encodings.size(); ++encoding)
        {
            if (encodings[encoding].empty()) continue;

            unsigned chunkSizeDefault = 32;
            unsigned chunkSizeReverse = (encodings[encoding].size() + WARP_SIZE - 1) / WARP_SIZE;
            unsigned chunkSize = std::max(chunkSizeDefault, chunkSizeReverse);
            unsigned noPartition = (encodings[encoding].size() + chunkSize - 1) / chunkSize;

            std::vector<std::vector<unsigned>> outerRowIds(noPartition);
            std::vector<bool> first(noPartition, false);
            unsigned added = 0;
            for (unsigned inner = 0; inner < chunkSize; ++inner)
            {
                for (unsigned part = 0; part < noPartition; ++part)
                {
                    if (first[part] == false)
                    {
                        first[part] = true;
                    }
                    if (added < encodings[encoding].size())
                    {
                        outerRowIds[part].emplace_back(encodings[encoding][added++]);
                    }
                }
            }
            for (unsigned part = 0; part < noPartition; ++part)
            {
                MASK individual = static_cast<MASK>(encoding);
                tempMasks.emplace_back(individual);
                for (unsigned i = 0; i < outerRowIds[part].size(); ++i)
                {
                    tempRowIds.emplace_back(outerRowIds[part][i]);
                }
                tempEncodingSizes.emplace_back(outerRowIds[part].size());
            }
        }
        this->hubEncodingOrderedDistribution(tempMasks, tempEncodingSizes, tempRowIds, masks[sliceSet], encodingSizes[sliceSet], rowIds[sliceSet], noMasks, total);
        m_SliceSetPtrs_Hub[sliceSet + 1] = total;
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

    m_RowIds_Hub = new unsigned[noSlices]; // you get row from slice
    idx = 0;
    for (unsigned sliceSet = 0; sliceSet < m_NoSliceSets_Hub; ++sliceSet)
    {
        for (unsigned i = 0; i < rowIds[sliceSet].size(); ++i)
        {
            m_RowIds_Hub[idx++] = rowIds[sliceSet][i];
        }
    }

    this->printHBRSHubData();
}

void HBRS::warpAlign(std::vector<Encoding>& warp, unsigned noMasks)
{
    
}

void HBRS::distributeEncodingsToWarp(std::vector<Encoding>& encodings, unsigned noMasks)
{
    std::vector<Encoding> distributed;

    unsigned current = 0;
    unsigned gridSize = WARP_SIZE * noMasks;
    std::vector<Encoding> warp;
    for (unsigned encoding = 0; encoding < encodings.size(); ++encoding)
    {
        warp.emplace_back(encodings[encoding]);
        ++current;
        if (current == gridSize)
        {
            this->warpAlign(warp, noMasks);
            current = 0;
            for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
            {
                for (unsigned iter = thread; iter < warp.size(); iter += WARP_SIZE)
                {
                    distributed.emplace_back(warp[iter]);
                }
            }
            warp.clear();
        }
    }
    if (!warp.empty())
    {
        /*
        current = 0;
        for (unsigned i = 0; i < warp.size(); ++i)
        {
            distributed.emplace_back(warp[i]);
            ++current;
        }
        while (current % noMasks != 0)
        {
            Encoding emptyEncoding;
            std::get<0>(emptyEncoding) = static_cast<MASK>(0);
            std::get<1>(emptyEncoding) = static_cast<MASK>(0);
            distributed.emplace_back(emptyEncoding);
            ++current;
        }
        */
        for (unsigned thread = 0; thread < WARP_SIZE; ++thread)
        {
            if (thread >= warp.size())
            {
                break;
            }
            for (unsigned mask = 0; mask < noMasks; ++mask)
            {
                unsigned encoding = mask * WARP_SIZE + thread;
                if (encoding < warp.size())
                {
                    distributed.emplace_back(warp[encoding]);
                }
                else
                {
                    Encoding emptyEncoding;
                    std::get<0>(emptyEncoding) = static_cast<MASK>(0);
                    std::get<1>(emptyEncoding) = static_cast<MASK>(0);
                    distributed.emplace_back(emptyEncoding);
                }
            }
        }
    }

    encodings = std::move(distributed);
}

double HBRS::euclideanDistance(std::unordered_map<unsigned, unsigned>& hash1, std::unordered_map<unsigned, unsigned>& hash2)
{
    double distance = 0;
    for (const auto& iter: hash1)
    {
        unsigned word1 = iter.first;
        unsigned frequency1 = iter.second;
        unsigned frequency2 = 0;
        if (hash2.contains(word1))
        {
            frequency2 = hash2[word1];
        }
        double temp = frequency1 - frequency2;
        distance += std::sqrt(temp * temp);
    }
    for (const auto& iter: hash2)
    {
        unsigned word2 = iter.first;
        unsigned frequency2 = iter.second;
        if (!hash1.contains(word2))
        {
            distance += std::sqrt(frequency2 * frequency2);
        }
    }
    return distance;
}

void HBRS::hubEncodingOrderedDistribution(std::vector<MASK>& tempMasks, std::vector<unsigned>& tempEncodingSizes, std::vector<unsigned>& tempRowIds, std::vector<MASK>& masks, std::vector<unsigned>& encodingSizes, std::vector<unsigned>& rowIds, unsigned noMasks, unsigned& total)
{
    std::vector<Encoding> encodings;
    unsigned acc = 0;
    for (unsigned k = 0; k < tempMasks.size(); ++k)
    {
        Encoding encoding;
        std::get<0>(encoding) = tempMasks[k];
        std::get<1>(encoding) = tempEncodingSizes[k];
        std::vector<unsigned>& rows = std::get<2>(encoding);
        std::unordered_map<unsigned, unsigned>& hash = std::get<3>(encoding);
        for (unsigned i = 0; i < tempEncodingSizes[k]; ++i)
        {
            unsigned row = tempRowIds[acc++];
            unsigned word = row / MASK_BITS;
            if (hash.contains(word))
            {
                ++hash[word];
            }
            else
            {
                hash[word] = 1;
            }
            rows.emplace_back(row);
        }
        encodings.emplace_back(encoding);
    }

    std::vector<Encoding> ordered;
    if (!encodings.empty())
    {
        std::vector<bool> used(encodings.size(), false);
        ordered.emplace_back(encodings[0]);
        used[0] = true;
        for (unsigned i = 1; i < encodings.size(); ++i)
        {
            double minDist = std::numeric_limits<double>::max();
            unsigned minIdx = 0;
            for (unsigned j = 0; j < encodings.size(); ++j)
            {
                if (used[j])
                {
                    continue;
                }
                double dist = this->euclideanDistance(std::get<3>(ordered.back()), std::get<3>(encodings[j]));
                if (dist < minDist)
                {
                    minDist = dist;
                    minIdx = j;
                }
            }
            ordered.emplace_back(encodings[minIdx]);
            used[minIdx] = true;
        }
    }
    encodings.clear();
    this->distributeEncodingsToWarp(ordered, noMasks);
    encodings = std::move(ordered);

    MASK cumulative = 0;
    unsigned cumulativeCounter = 0;
    for (unsigned encoding = 0; encoding < encodings.size(); ++encoding)
    {
        cumulative |= (static_cast<MASK>(std::get<0>(encodings[encoding]) << (m_SliceSize_Hub * cumulativeCounter++)));
        encodingSizes.emplace_back(std::get<1>(encodings[encoding]));
        const std::vector<unsigned>& rows = std::get<2>(encodings[encoding]);
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
        ++total;
    }
}

void HBRS::printHBRSData()
{
    std::cout << "************ PRINTING NORMAL DATA ************" << std::endl;

    unsigned noMasks = MASK_BITS / m_SliceSize;

    std::cout << "MASK size: " << MASK_BITS << std::endl;
    std::cout << "Slice size: " << m_SliceSize << std::endl;
    std::cout << "Number of slice sets: " << m_NoSliceSets << std::endl;
    std::cout << "Number of slices: " << m_SliceSetPtrs[m_NoSliceSets] << std::endl;
    std::cout << "Number of slices in each set row: " << K / m_SliceSize << std::endl;
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
    unsigned long long totalMaskBits = 0;
    unsigned long long totalBaselineBits = 0;
    unsigned noChunks = 10;
    unsigned chunkSize = (m_NoSliceSets_Hub + noChunks - 1) / noChunks;
    MASK andMask = (0xFFFFFFFF >> (MASK_BITS - m_SliceSize_Hub));
    for (unsigned q = 0; q < noChunks; ++q)
    {
        unsigned start = q * chunkSize;
        unsigned end = std::min(m_NoSliceSets_Hub, start + chunkSize);

        unsigned long long maskBitsChunk = 0;
        unsigned long long baselineBitsChunk = 0;

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

            maskBitsChunk += (encodingsInSet * m_SliceSize_Hub);
            baselineBitsChunk += (slicesInSet * m_SliceSize_Hub);
        }

        totalMaskBits += maskBitsChunk;
        totalBaselineBits += baselineBitsChunk;

        double chunkEff = double(baselineBitsChunk) / maskBitsChunk;
        std::cout << "Chunk: " << q
                  << " - Baseline bits: " << baselineBitsChunk
                  << " - Stored mask bits: " << maskBitsChunk
                  << " - Compression efficiency: x" << chunkEff << std::endl;
    }

    unsigned long long storedMaskBits = noEncodings* m_SliceSize_Hub;
    unsigned long long baselineBits = noSlices * m_SliceSize_Hub;
    double compressionEff = double(baselineBits) / storedMaskBits;

    std::cout << "Stored mask bits (total): " << storedMaskBits << std::endl;
    std::cout << "Baseline bits (total): " << baselineBits << std::endl;
    std::cout << "Compression efficiency (baseline / stored): x" << compressionEff << std::endl;

    std::cout << "Check: " << totalNumberOfEdgesCheck << std::endl;
}

#endif
