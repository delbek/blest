#pragma once

#include "Common.cuh"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <stdexcept>

// Gorder
#include "Graph.h"
// Gorder

class CSC
{
public:
	CSC() = default;
	CSC(std::string filename, bool undirected, bool binary);
	CSC(const CSC& other) = delete;
	CSC(CSC&& other) noexcept = delete;
	CSC& operator=(const CSC& other) = delete;
	CSC& operator=(CSC&& other) noexcept = delete;
	~CSC();

	void constructFromBinary(std::string filename);
    void saveToBinary(std::string filename);

	[[nodiscard]] inline unsigned& getN() {return m_N;}
	[[nodiscard]] inline unsigned& getNNZ() {return m_NNZ;}
	[[nodiscard]] inline unsigned*& getColPtrs() {return m_ColPtrs;}
	[[nodiscard]] inline unsigned*& getRows() {return m_Rows;}
	[[nodiscard]] inline bool& isSocialNetwork() {return m_IsSocial;}

	// metrics
	unsigned maxBandwidth();
	double averageBandwidth();
	unsigned maxProfile();
	double averageProfile();
	unsigned maxDegree();
	double averageDegree();
	//

	unsigned* orderFromBinary(std::string filename);
	void saveOrderingToBinary(std::string filename, unsigned* inversePermutation);
	unsigned* reorder(unsigned sliceSize);
	bool checkSymmetry();
	CSC* symmetrize();
	CSC* transpose();
	std::vector<bool> bfsSwitchPolicy();

private:
	bool socialNetworkHelper();

	unsigned* natural();
	unsigned* random();
	unsigned* jackard(unsigned sliceSize);
	unsigned* jackardWithWindow(unsigned sliceSize, unsigned windowSize);
	unsigned* rcmWithJackard(unsigned sliceSize, unsigned windowSize);
	unsigned* gorderWithJackard(unsigned sliceSize, unsigned windowSize);
	unsigned* gorder(unsigned sliceSize);
	unsigned* degreeSort(bool ascending = true);
	unsigned findPseudoPeripheralNode(CSC* csc, unsigned startNode);
	unsigned* rcm();
	unsigned* soloVertexPermutation();
	
	void applyPermutation(unsigned* inversePermutation);
	bool permutationCheck(unsigned* inversePermutation);

private:
	unsigned m_N;
	unsigned m_NNZ;
	bool m_IsSocial;

	unsigned* m_ColPtrs;
	unsigned* m_Rows;
};

CSC::CSC(std::string filename, bool undirected, bool binary)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file from which to construct CSC.");
	}

	while (file.peek() == '%')
	{
		file.ignore(2048, '\n');
	}

	double trash;
	unsigned noLines;

	file >> m_N >> m_N >> noLines;

	std::vector<std::vector<unsigned>> adjList(m_N);

	// there exists an edge j -> i
	unsigned i, j;
	m_NNZ = 0;
	for (unsigned iter = 0; iter < noLines; ++iter)
	{
		file >> j >> i; // read transpose
		if (!binary) file >> trash;
		if (i == j) continue;

		--i;
		--j;

		adjList[j].emplace_back(i);
		++m_NNZ;

		if (undirected)
		{
			++m_NNZ;
			adjList[i].emplace_back(j);
		}
	}
	file.close();

	std::cout << "Number of vertices: " << m_N << std::endl;
	std::cout << "Number of edges: " << m_NNZ << std::endl;
	std::cout << "Sparsity: " << std::fixed << static_cast<double>(m_NNZ) / (static_cast<double>(m_N) * static_cast<double>(m_N)) << std::endl;

	m_ColPtrs = new unsigned[m_N + 1];
	m_Rows = new unsigned[m_NNZ];

	std::fill(m_ColPtrs, m_ColPtrs + m_N + 1, 0);
	for (unsigned j = 0; j < m_N; ++j)
	{
		m_ColPtrs[j + 1] = adjList[j].size();
	}

	for (unsigned j = 0; j < m_N; ++j)
	{
		m_ColPtrs[j + 1] += m_ColPtrs[j];
		for (unsigned nnz = 0; nnz < adjList[j].size(); ++nnz)
		{
			m_Rows[m_ColPtrs[j] + nnz] = adjList[j][nnz];
		}
	}

	double average = this->averageDegree();
	std::cout << "Average degree: " << average << std::endl;
	std::cout << "Max degree: " << this->maxDegree() << std::endl;
	//

	#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic, 2048)
	for (unsigned j = 0; j < m_N; ++j)
	{
		unsigned start = m_ColPtrs[j];
		unsigned end = m_ColPtrs[j + 1];
		std::sort(m_Rows + start, m_Rows + end);
	}

	m_IsSocial = (this->socialNetworkHelper() | (average > SOCIAL_THRESHOLD));
	if (m_IsSocial)
	{
		std::cout << "The graph is a social network." << std::endl;
	}
	else
	{
		std::cout << "The graph is not a social network." << std::endl;
	}
}

bool CSC::socialNetworkHelper()
{
    std::vector<unsigned> outDegree(m_N);
    for (unsigned j = 0; j < m_N; ++j)
    {
        outDegree[j] = m_ColPtrs[j + 1] - m_ColPtrs[j];
    }

    std::vector<unsigned> inDegree(m_N, 0);
    for (unsigned col = 0; col < m_N; ++col)
    {
        for (unsigned p = m_ColPtrs[col]; p < m_ColPtrs[col + 1]; ++p)
        {
            unsigned row = m_Rows[p];
            if (row < m_N)
            {
                ++inDegree[row];
            }
        }
    }

    bool socialByIn = isSocialDegreeDistribution(inDegree);
    bool socialByOut = isSocialDegreeDistribution(outDegree);

    if (socialByIn || socialByOut)
    {
        return true;
    }

    return false;
}

CSC::~CSC()
{
	delete[] m_ColPtrs;
	delete[] m_Rows;
}

void CSC::saveToBinary(std::string filename)
{
    std::ofstream out(filename, std::ios::binary);

    out.write(reinterpret_cast<const char*>(&m_N), (sizeof(unsigned)));
    out.write(reinterpret_cast<const char*>(&m_NNZ), (sizeof(unsigned)));
	out.write(reinterpret_cast<const char*>(&m_IsSocial), (sizeof(bool)));

    out.write(reinterpret_cast<const char*>(m_ColPtrs), (sizeof(unsigned) * (m_N + 1)));
    out.write(reinterpret_cast<const char*>(m_Rows), (sizeof(unsigned) * m_NNZ));

    out.close();
}

void CSC::constructFromBinary(std::string filename)
{
    std::ifstream in(filename, std::ios::binary);

    in.read(reinterpret_cast<char*>(&m_N), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_NNZ), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&m_IsSocial), sizeof(bool));

    m_ColPtrs = new unsigned[m_N + 1];
    in.read(reinterpret_cast<char*>(m_ColPtrs), (sizeof(unsigned) * (m_N + 1)));

    m_Rows = new unsigned[m_NNZ];
    in.read(reinterpret_cast<char*>(m_Rows), (sizeof(unsigned) * m_NNZ));

    in.close();

	std::cout << "Average degree: " << this->averageDegree() << std::endl;
	std::cout << "Max degree: " << this->maxDegree() << std::endl;
	if (m_IsSocial)
	{
		std::cout << "The graph is a social network." << std::endl;
	}
	else
	{
		std::cout << "The graph is not a social network." << std::endl;
	}
    std::cout << "CSC read from binary." << std::endl;
}

unsigned CSC::maxBandwidth()
{
	unsigned band = 0;
	for (unsigned j = 0; j < m_N; ++j)
	{
		if (m_ColPtrs[j] == m_ColPtrs[j + 1]) continue;
		band = std::max(band, static_cast<unsigned>(std::abs(static_cast<int>(j) - static_cast<int>(m_Rows[m_ColPtrs[j + 1] - 1])))); // assuming sorted adj
	}
	return band;
}

double CSC::averageBandwidth()
{
	double average = 0;
	for (unsigned j = 0; j < m_N; ++j)
	{
		if (m_ColPtrs[j] == m_ColPtrs[j + 1]) continue;
		average += static_cast<unsigned>(std::abs(static_cast<int>(j) - static_cast<int>(m_Rows[m_ColPtrs[j + 1] - 1]))); // assuming sorted adj
	}
	average /= m_N;
	return average;
}

unsigned CSC::maxProfile()
{
	unsigned prof = 0;
	for (unsigned j = 0; j < m_N; ++j)
	{
		if (m_ColPtrs[j] + 2 > m_ColPtrs[j + 1]) continue;
		prof = std::max(prof, static_cast<unsigned>(m_Rows[m_ColPtrs[j + 1] - 1] - m_Rows[m_ColPtrs[j]])); // assuming sorted adj
	}
	return prof;
}

double CSC::averageProfile()
{
	double average = 0;
	for (unsigned j = 0; j < m_N; ++j)
	{
		if (m_ColPtrs[j] + 2 > m_ColPtrs[j + 1]) continue;
		average += static_cast<unsigned>(m_Rows[m_ColPtrs[j + 1] - 1] - m_Rows[m_ColPtrs[j]]); // assuming sorted adj
	}
	average /= m_N;
	return average;
}

unsigned CSC::maxDegree()
{
	unsigned max = 0;
	for (unsigned j = 0; j < m_N; ++j)
	{
		max = std::max(max, m_ColPtrs[j + 1] - m_ColPtrs[j]);
	}
	return max;
}

double CSC::averageDegree()
{
	double average = 0;
	for (unsigned j = 0; j < m_N; ++j)
	{
		average += (m_ColPtrs[j + 1] - m_ColPtrs[j]);
	}
	average /= m_N;
	return average;
}

unsigned* CSC::orderFromBinary(std::string filename)
{
	std::ifstream in(filename, std::ios::binary);

	unsigned* inversePermutation = new unsigned[m_N];
	in.read(reinterpret_cast<char*>(inversePermutation), (sizeof(unsigned) * m_N));

    in.close();
    std::cout << "Ordering read from binary." << std::endl;
	applyPermutation(inversePermutation);

	return inversePermutation;
}

void CSC::saveOrderingToBinary(std::string filename, unsigned* inversePermutation)
{
    std::ofstream out(filename, std::ios::binary);

    out.write(reinterpret_cast<const char*>(inversePermutation), (sizeof(unsigned) * m_N));

    out.close();
}

unsigned* CSC::reorder(unsigned sliceSize)
{
	double start = omp_get_wtime();
	unsigned* inversePermutation = nullptr;
	if (this->isSocialNetwork())
	{
		//unsigned* soloPerm = this->soloVertexPermutation();
		//unsigned* innerPerm;
		if (JACKARD_ON)
		{
			std::cout << "Reordering with JaccardWithWindows" << std::endl;
			inversePermutation = this->jackardWithWindow(sliceSize, WINDOW_SIZE);
		}
		else
		{
			std::cout << "Natural Ordering" << std::endl;
			inversePermutation = this->natural();
		}
		//inversePermutation = chainPermutations(m_N, soloPerm, innerPerm);
	}
	else
	{
		std::cout << "Reordering with RCM" << std::endl;
		inversePermutation = this->rcm();
	}
	double end = omp_get_wtime();
	std::cout << "Time took to reorder: " << end - start << std::endl;
	return inversePermutation;
}

unsigned* CSC::soloVertexPermutation()
{
	unsigned* m_InversePermutation = new unsigned[m_N];

	unsigned forwardCounter = 0;
	unsigned backwardCounter = m_N - 1;
	for (unsigned j = 0; j < m_N; ++j)
	{
		if ((m_ColPtrs[j + 1] - m_ColPtrs[j]) == 0)
		{
			m_InversePermutation[j] = backwardCounter--;
		}
		else
		{
			m_InversePermutation[j] = forwardCounter++;
		}
	}

	applyPermutation(m_InversePermutation);
	return m_InversePermutation;
}

bool CSC::checkSymmetry()
{
    for (unsigned j = 0; j < m_N; ++j)
    {
        for (unsigned nnz = m_ColPtrs[j]; nnz < m_ColPtrs[j + 1]; ++nnz)
        {
            unsigned i = m_Rows[nnz]; // (i, j)
            if (i == j) continue;
			bool found = false;
			for (unsigned nnz2 = m_ColPtrs[i]; nnz2 < m_ColPtrs[i + 1]; ++nnz2)
			{
				unsigned j2 = m_Rows[nnz2]; // (j2, i)
				if (j2 == j)
				{
					found = true;
					break;
				}
			}
			if (!found)
			{
				return false;
			}
        }
    }
    return true;
}

CSC* CSC::transpose()
{
	CSC* transpose = new CSC;
	transpose->m_ColPtrs = new unsigned[m_N + 1];
	transpose->m_Rows = new unsigned[m_ColPtrs[m_N]];

	unsigned* ptrs = transpose->m_ColPtrs;
	unsigned* ids = transpose->m_Rows;

	std::fill(ptrs, ptrs + m_N + 1, 0);
	for (unsigned j = 0; j <  m_ColPtrs[m_N]; ++j)
	{
		++ptrs[m_Rows[j] + 1];
	}

	for (unsigned i = 1; i <= m_N; ++i)
	{
		ptrs[i] += ptrs[i - 1];
	}

	for (unsigned j = 0; j < m_N; ++j)
	{
		for (unsigned p = m_ColPtrs[j]; p < m_ColPtrs[j + 1]; ++p)
		{
			unsigned i = m_Rows[p];
			ids[ptrs[i]++] = j;
		}
	}

	for (unsigned i = m_N; i > 0; --i)
	{
		ptrs[i] = ptrs[i - 1];
	}
	ptrs[0] = 0;

	return transpose;
}

CSC* CSC::symmetrize()
{
    std::vector<std::vector<unsigned>> cols(m_N);
    cols.reserve(m_N);
    for (unsigned j = 0; j < m_N; ++j)
    {
        for (unsigned nnz = m_ColPtrs[j]; nnz < m_ColPtrs[j + 1]; ++nnz)
        {
            unsigned i = m_Rows[nnz];
            cols[j].push_back(i);
            if (i != j)
			{
				cols[i].push_back(j);
			}
        }
    }

	unsigned nnz = 0;
    for (unsigned j = 0; j < m_N; ++j)
    {
        auto& v = cols[j];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
		nnz += v.size();
    }

    CSC* csc = new CSC;
    csc->getN() = m_N;
	csc->getColPtrs() = new unsigned[m_N + 1];
	csc->getRows() = new unsigned[nnz];
    unsigned* colPtrs = csc->getColPtrs();
    unsigned* rows = csc->getRows();

    colPtrs[0] = 0;
    for (unsigned j = 0; j < m_N; ++j)
	{
		colPtrs[j + 1] = colPtrs[j] + cols[j].size();
	}

    for (unsigned j = 0; j < m_N; ++j)
    {
        std::copy(cols[j].begin(), cols[j].end(), &rows[colPtrs[j]]);
    }

    return csc;
}

unsigned* CSC::natural()
{
	unsigned* inversePermutation = new unsigned[m_N];
	for (unsigned i = 0; i < m_N; ++i)
	{
		inversePermutation[i] = i;
	}
	return inversePermutation;
}

unsigned* CSC::random()
{
	auto rng = [&]()
	{
		static std::mt19937 gen(std::random_device{}());
		return gen;
	};
	unsigned* inversePermutation = new unsigned[m_N];
	std::vector<unsigned> perm(m_N);
	std::iota(perm.begin(), perm.end(), 0);
	std::shuffle(perm.begin(), perm.end(), rng());

	for (unsigned i = 0; i < m_N; ++i)
	{
		inversePermutation[perm[i]] = i;
	}
	applyPermutation(inversePermutation);

	return inversePermutation;
}

std::vector<bool> CSC::bfsSwitchPolicy()
{
	constexpr unsigned NO_TRIAL = 20;
	const unsigned lazyLimit = m_N / 32 / 6;
	std::unordered_set<unsigned> selecteds;
	
	std::vector<unsigned> votes;
	for (unsigned i = 0; i < NO_TRIAL; ++i)
	{
		unsigned source;
		do
		{
			source = rand(0, m_N - 1);
		} while (selecteds.contains(source));
		selecteds.insert(source);

		std::vector<unsigned> frontierDistribution;
		frontierDistribution.reserve(m_N);
		unsigned currentFrontierSize = 0;

		bool* visited = new bool[m_N];
		unsigned* queue = new unsigned[m_N];
		unsigned qs = 0;
		unsigned qe = 0;
		queue[qe++] = source;
		visited[source] = true;
		unsigned levelEnd = qe;

		while (qs < qe)
		{
			unsigned u = queue[qs++];
			++currentFrontierSize;
			for (unsigned ptr = m_ColPtrs[u]; ptr < m_ColPtrs[u + 1]; ++ptr)
			{
				unsigned v = m_Rows[ptr];
				if (visited[v] == false)
				{
					visited[v] = true;
					queue[qe++] = v;
				}
			}
			if (qs == levelEnd)
			{
				frontierDistribution.emplace_back(currentFrontierSize);
				currentFrontierSize = 0;
				levelEnd = qe;
			}
		}
		delete[] visited;
		delete[] queue;

		if (votes.size() < frontierDistribution.size())
		{
			for (unsigned l = votes.size(); l < frontierDistribution.size(); ++l)
			{
				votes.emplace_back(0);
			}
		}
		for (unsigned l = 0; l < frontierDistribution.size(); ++l)
		{
			if (frontierDistribution[l] >= lazyLimit)
			{
				++votes[l];
			}
		}
	}

	std::vector<bool> results;
	for (const auto& v: votes)
	{
		if (v >= (NO_TRIAL / 4))
		{
			results.emplace_back(true);
		}
		else
		{
			results.emplace_back(false);
		}
	}
	return results;
}

unsigned* CSC::jackard(unsigned sliceSize)
{
    unsigned* inversePermutation = new unsigned[m_N];

    std::vector<std::pair<unsigned, unsigned>> queue;
    queue.reserve(m_N);
    for (unsigned j = 0; j < m_N; ++j)
    {
        unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
        queue.push_back({deg, j});
    }

    std::vector<unsigned> rowMark(m_N, 0);
    unsigned epoch = 1;
    unsigned unionSize = 0;

    unsigned sharedBestListIdx;
    double sharedBestVal;

    unsigned noSliceSets = (m_N + sliceSize - 1) / sliceSize;
    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        for (unsigned sliceSet = 0; sliceSet < noSliceSets; ++sliceSet)
        {
            #pragma omp single
            {
                if (!queue.empty())
                {
                    ++epoch;
                    unionSize = 0;

                    unsigned bestListIdx = 0;
                    unsigned maxDeg = queue[0].first;
                    for (unsigned i = 1; i < queue.size(); ++i)
                    {
                        if (queue[i].first > maxDeg)
                        {
                            maxDeg = queue[i].first;
                            bestListIdx = i;
                        }
                    }
                    unsigned col = queue[bestListIdx].second;
                    queue[bestListIdx] = queue.back();
                    queue.pop_back();

                    for (unsigned nnz = m_ColPtrs[col]; nnz < m_ColPtrs[col + 1]; ++nnz)
                    {
                        unsigned r = m_Rows[nnz];
                        if (rowMark[r] != epoch)
                        {
                            rowMark[r] = epoch;
                            ++unionSize;
                        }
                    }
                    inversePermutation[col] = sliceSet * sliceSize;
                    sharedBestListIdx = 0; 
                }
                else
                {
                    sharedBestListIdx = UNSIGNED_MAX;
                }
            }

            if (sharedBestListIdx == UNSIGNED_MAX)
            {
                break;
            }

            unsigned sliceStart = sliceSet * sliceSize;
            unsigned sliceEnd = std::min(sliceStart + sliceSize, m_N);
            for (unsigned newCol = sliceStart + 1; newCol < sliceEnd; ++newCol)
            {
                #pragma omp single
                {
                    if (queue.empty())
                    {
                        sharedBestListIdx = UNSIGNED_MAX;
                    }
                    else
                    {
                        sharedBestListIdx = UNSIGNED_MAX;
                        sharedBestVal = -1;
                    }
                }

                if (queue.empty())
                {
                    break;
                }

                double myBestVal = -1;
                unsigned myBestListIdx = UNSIGNED_MAX;
                size_t qSize = queue.size();

                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < qSize; ++i)
                {
                    unsigned j = queue[i].second;
                    unsigned inter = 0;
                    for (unsigned nnz = m_ColPtrs[j]; nnz < m_ColPtrs[j + 1]; ++nnz)
                    {
                        if (rowMark[m_Rows[nnz]] == epoch)
                        {
                            inter++;
                        }
                    }

                    unsigned deg = queue[i].first;
                    unsigned uni = unionSize + deg - inter;
                    double val = (uni == 0) ? 1.0 : static_cast<double>(inter) / uni;

                    if (val > myBestVal)
                    {
                        myBestVal = val;
                        myBestListIdx = i;
                    }
                }

                #pragma omp critical
                {
                    if (myBestVal > sharedBestVal)
                    {
                        sharedBestVal = myBestVal;
                        sharedBestListIdx = myBestListIdx;
                    }
                }
                #pragma omp barrier

                #pragma omp single
                {
                    if (sharedBestListIdx != UNSIGNED_MAX)
                    {
                        unsigned bestCol = queue[sharedBestListIdx].second;
                        queue[sharedBestListIdx] = queue.back();
                        queue.pop_back();

                        for (unsigned nnz = m_ColPtrs[bestCol]; nnz < m_ColPtrs[bestCol + 1]; ++nnz)
                        {
                            unsigned r = m_Rows[nnz];
                            if (rowMark[r] != epoch)
                            {
                                rowMark[r] = epoch;
                                ++unionSize;
                            }
                        }
                        inversePermutation[bestCol] = newCol;
                    }
                }

                if (sharedBestListIdx == UNSIGNED_MAX)
                {
                    break;
                }
            }
        }
    }

    applyPermutation(inversePermutation);
    return inversePermutation;
}

unsigned* CSC::jackardWithWindow(unsigned sliceSize, unsigned windowSize)
{
	assert(windowSize % sliceSize == 0);

	CSC* transpose = this->transpose();
	unsigned* ptrs = transpose->getColPtrs();
	unsigned* ids = transpose->getRows();
	
    unsigned* inversePermutation = new unsigned[m_N];
    
    unsigned noWindows = (m_N + windowSize - 1) / windowSize;
    #pragma omp parallel num_threads(omp_get_max_threads())
    {
		// per window reset
		bool* cmarkers = new bool[windowSize];
		unsigned* rowPtrs = new unsigned[m_N];
		//std::unordered_map<unsigned, unsigned> rowPtrs;

		// per slice set reset
		unsigned* intersectionCounts = new unsigned[windowSize];
		unsigned jackardQueueSize = 0;
		unsigned* jackardQueue = new unsigned[windowSize];

		// per slice set increment
		unsigned* rmarkers = new unsigned[m_N];
		std::fill(rmarkers, rmarkers + m_N, 0);
        unsigned epoch = 1;

        #pragma omp for schedule(dynamic)
        for (unsigned window = 0; window < noWindows; ++window)
        {
			std::fill(cmarkers, cmarkers + windowSize, false);
			//rowPtrs.clear();
			std::fill(rowPtrs, rowPtrs + m_N, UNSIGNED_MAX);

            unsigned windowStart = window * windowSize;
            unsigned windowEnd = std::min(windowStart + windowSize, m_N);
            unsigned windowLength = windowEnd - windowStart;

            unsigned noSliceSets = (windowLength + sliceSize - 1) / sliceSize;
            for (unsigned sliceSet = 0; sliceSet < noSliceSets; ++sliceSet)
            {
				std::fill(intersectionCounts, intersectionCounts + windowSize, 0); // can it be avoided?
				jackardQueueSize = 0;
	      
	      		unsigned sliceStart = windowStart + sliceSet * sliceSize;
                unsigned sliceEnd = std::min(sliceStart + sliceSize, windowEnd);

                ++epoch;
                unsigned unionSize = 0;

				auto selectSingleton = [&](unsigned& singleton) -> bool
				{
					singleton = UNSIGNED_MAX;
					unsigned max = 0;
					for (unsigned j = windowStart; j < windowEnd; ++j) // expensive loop
					{
						if (cmarkers[j - windowStart] == false)
						{
							unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
							if (deg >= max)
							{
								singleton = j;
								max = deg;
							}
						}
					}
					return singleton != UNSIGNED_MAX;
				};
				unsigned singleton;
				selectSingleton(singleton);
				cmarkers[singleton - windowStart] = true;
		
                for (unsigned nnz = m_ColPtrs[singleton]; nnz < m_ColPtrs[singleton + 1]; ++nnz)
                {
					unsigned r = m_Rows[nnz];
					rmarkers[r] = epoch;
					++unionSize;
					if (rowPtrs[r] != UNSIGNED_MAX)
					{
						nowContainsSingleton:
						for (unsigned p = rowPtrs[r]; p < ptrs[r + 1]; ++p)
						{
							unsigned c = ids[p];
							if (c < windowEnd && cmarkers[c - windowStart] == false)
							{
								if (intersectionCounts[c - windowStart] == 0)
								{
									jackardQueue[jackardQueueSize++] = c;
								}
								intersectionCounts[c - windowStart]++;
							}
							if (c >= windowEnd)
							{
								break;
							}
						}
					}
					else
					{
						for (unsigned p = ptrs[r]; p < ptrs[r + 1]; ++p)
						{
							unsigned c = ids[p];
							if (c >= windowStart)
							{
								rowPtrs[r] = p;
								goto nowContainsSingleton;
							}
						}
					}
                }
                inversePermutation[singleton] = sliceStart;

                for (unsigned newCol = sliceStart + 1; newCol < sliceEnd; ++newCol)
		  		{
					unsigned bestCol;
					if (jackardQueueSize == 0)
					{
						if (selectSingleton(bestCol) == false)
						{
							break;
						}
					}
					else
					{
						unsigned colIdx = UNSIGNED_MAX;
						double bestJackard = -1;
						for (unsigned jidx = 0; jidx < jackardQueueSize; ++jidx)
						{
							unsigned j = jackardQueue[jidx];

							unsigned inter = intersectionCounts[j - windowStart];
							unsigned deg = (m_ColPtrs[j + 1] - m_ColPtrs[j]);
							unsigned uni = unionSize + deg - inter;
							double val = (uni == 0) ? 1.0 : static_cast<double>(inter) / uni;
				
							if (val > bestJackard)
							{
								colIdx = jidx;
								bestJackard = val;
							}
						}
						bestCol = jackardQueue[colIdx];
						jackardQueue[colIdx] = jackardQueue[jackardQueueSize - 1];
						--jackardQueueSize;
					}
					cmarkers[bestCol - windowStart] = true;

                    for (unsigned nnz = m_ColPtrs[bestCol]; nnz < m_ColPtrs[bestCol + 1]; ++nnz)
		      		{
                        unsigned r = m_Rows[nnz];
                        if (rmarkers[r] != epoch)
			  			{
                            rmarkers[r] = epoch;
                            ++unionSize;
							if (rowPtrs[r] != UNSIGNED_MAX)
							{
								nowContains:
								for (unsigned p = rowPtrs[r]; p < ptrs[r + 1]; ++p)
								{
									unsigned c = ids[p];
									if (c < windowEnd && cmarkers[c - windowStart] == false)
									{
										if (intersectionCounts[c - windowStart] == 0)
										{
											jackardQueue[jackardQueueSize++] = c;
										}
										intersectionCounts[c - windowStart]++;
									}
									if (c >= windowEnd)
									{
										break;
									}
								}
							}
							else
							{
								for (unsigned p = ptrs[r]; p < ptrs[r + 1]; ++p)
								{
									unsigned c = ids[p];
									if (c >= windowStart)
									{
										rowPtrs[r] = p;
										goto nowContains;
									}
								}
							}
			  			}
		      		}
                    inversePermutation[bestCol] = newCol;
				}
            }
        }
		delete[] rmarkers;
		delete[] intersectionCounts;
		delete[] jackardQueue;
		delete[] cmarkers;
		delete[] rowPtrs;
    }
	delete transpose;

	applyPermutation(inversePermutation);
	return inversePermutation;
}

unsigned* CSC::rcmWithJackard(unsigned sliceSize, unsigned windowSize)
{
	unsigned* rcmPermutation = this->rcm();
	unsigned* jackardPermutation = this->jackardWithWindow(sliceSize, windowSize);

	unsigned* chained = chainPermutations(m_N, rcmPermutation, jackardPermutation);

	return chained;
}

unsigned* CSC::gorderWithJackard(unsigned sliceSize, unsigned windowSize)
{
	Gorder::Graph g;
	g.setFilename("gorder");
	g.readGraph(m_N, m_NNZ, m_ColPtrs, m_Rows);
	g.Transform();
	std::vector<long long int> order;
	g.GorderGreedy(order, windowSize);

	unsigned* inversePermutation = new unsigned[m_N];
	std::copy(order.data(), order.data() + m_N, inversePermutation);
	applyPermutation(inversePermutation);
	unsigned* inversePermutation2 = this->jackardWithWindow(sliceSize, windowSize);

	unsigned* chained = chainPermutations(m_N, inversePermutation, inversePermutation2);
	return chained;
}

unsigned* CSC::gorder(unsigned sliceSize)
{
	unsigned windowSize = sliceSize;

	Gorder::Graph g;
	g.setFilename("gorder");
	g.readGraph(m_N, m_NNZ, m_ColPtrs, m_Rows);
	g.Transform();
	std::vector<long long int> order;
	g.GorderGreedy(order, windowSize);

	unsigned* inversePermutation = new unsigned[m_N];
	std::copy(order.data(), order.data() + m_N, inversePermutation);
	applyPermutation(inversePermutation);

	return inversePermutation;
}

unsigned* CSC::degreeSort(bool ascending)
{
	std::vector<std::pair<unsigned, unsigned>> degrees(m_N);

	for (unsigned j = 0; j < m_N; ++j)
	{
		degrees[j].first = j;
		degrees[j].second = m_ColPtrs[j + 1] - m_ColPtrs[j];
	}
	std::sort(degrees.begin(), degrees.end(), [&](const auto& a, const auto& b)
	{
		if (a.second != b.second) return (ascending ? a.second < b.second : a.second > b.second);
		return a.first < b.first;
	});

	unsigned* inversePermutation = new unsigned[m_N];

	for (unsigned j = 0; j < m_N; ++j)
	{
		inversePermutation[degrees[j].first] = j;
	}
	applyPermutation(inversePermutation);

	return inversePermutation;
}

unsigned CSC::findPseudoPeripheralNode(CSC* csc, unsigned startNode)
{
	unsigned* ptrs = csc->getColPtrs();
	unsigned* inds = csc->getRows();

	unsigned* visited = new unsigned[m_N];
	std::fill(visited, visited + m_N, 0);
	
	unsigned* q = new unsigned[m_N];

	unsigned maxLevel = 0;
	unsigned trial = 0;
	unsigned marker = 1;
	while (trial < 10)
	{
		++marker;

		q[0] = startNode;
		visited[startNode] = marker;
		unsigned qe = 1;
		unsigned qs = 0;

		unsigned levelStart = 0;
		unsigned levelEnd = qe;
		unsigned levelid = 1;
		
		while (qs < qe)
		{
			unsigned u = q[qs++];
			for (unsigned ptr = ptrs[u]; ptr < ptrs[u + 1]; ++ptr)
			{
				unsigned v = inds[ptr];
				if (visited[v] != marker)
				{
					visited[v] = marker;
					q[qe++] = v;
				}
			}

			if (qs == levelEnd && qs < qe)
			{
			  levelStart = qs;
			  levelEnd = qe;
			  levelid++;
			}
		}
		if (levelid == maxLevel)
		{
			trial++; 
		}
		else 
		{
			trial = 0;
		}
		maxLevel = levelid;
		unsigned nextStart = q[levelStart + (rand() % (levelEnd - levelStart))];
		startNode = nextStart;
	}

	delete[] visited;
	delete[] q;

	return startNode;
}

unsigned* CSC::rcm()
{
	CSC* csc = this;
	if (!this->checkSymmetry())
	{
		csc = this->symmetrize();
	}

	unsigned* ptrs = csc->getColPtrs();
	unsigned* inds = csc->getRows();
	
	unsigned* inversePermutation = new unsigned[m_N];

	unsigned* degrees = new unsigned[m_N];
	for (unsigned i = 0; i < m_N; ++i)
	{
		degrees[i] = ptrs[i + 1] - ptrs[i];
	}

	unsigned* visited = new unsigned[m_N];
	std::fill(visited, visited + m_N, 0);

	unsigned* q = new unsigned[m_N];
	unsigned* nbrs = new unsigned[m_N];

	auto compAsc = [&](unsigned v1, unsigned v2)
	{
		if (degrees[v1] < degrees[v2])
		{
			return true;
		}
		return false;
	};

	unsigned permuted = 0;
	for (unsigned i = 0; i < m_N; ++i)
	{
		if (visited[i] == 1) continue;
		
		unsigned start = this->findPseudoPeripheralNode(csc, i);
		unsigned qs = 0;
		unsigned qe = 0;
		q[qe++] = start;
		visited[start] = 1;
		while (qs < qe)
		{
			unsigned u = q[qs++];
			inversePermutation[u] = permuted++;

			unsigned noNeigh = 0;
			for (unsigned ptr = ptrs[u]; ptr < ptrs[u + 1]; ++ptr)
			{
				unsigned v = inds[ptr];
				if (visited[v] == 0)
				{
					nbrs[noNeigh++] = v;
				  	visited[v] = 1;
				}
			}

			std::sort(nbrs, nbrs + noNeigh, compAsc);
			
			for (unsigned nbr = 0; nbr < noNeigh; ++nbr)
			{
			  	unsigned v = nbrs[nbr];
			  	q[qe++] = v;
			}
		}
	}

	delete[] degrees;
	delete[] visited;
	delete[] q;
	delete[] nbrs;

	if (csc != this)
	{
		delete csc;
	}

	applyPermutation(inversePermutation);

	return inversePermutation;
}

void CSC::applyPermutation(unsigned* inversePermutation)
{
	if (!permutationCheck(inversePermutation))
	{
		throw std::runtime_error("Permutation check has failed.");
	}

	unsigned* newColPtrs = new unsigned[m_N + 1];
	std::fill(newColPtrs, newColPtrs + m_N + 1, 0);

	for (unsigned oldCol = 0; oldCol < m_N; ++oldCol)
	{
		unsigned newCol = inversePermutation[oldCol];
		newColPtrs[newCol + 1] = m_ColPtrs[oldCol + 1] - m_ColPtrs[oldCol];
	}

	for (unsigned j = 0; j < m_N; ++j)
	{
		newColPtrs[j + 1] += newColPtrs[j];
	}

	unsigned* newRows = new unsigned[m_NNZ];

	for (unsigned oldCol = 0; oldCol < m_N; ++oldCol)
	{
		unsigned newCol = inversePermutation[oldCol];
		unsigned start = m_ColPtrs[oldCol];
		unsigned end = m_ColPtrs[oldCol + 1];

		for (unsigned nnz = start; nnz < end; ++nnz)
		{
			unsigned oldRow = m_Rows[nnz];
			unsigned newRow = inversePermutation[oldRow];
			newRows[newColPtrs[newCol] + (nnz - start)] = newRow;
		}
	}

	for (unsigned col = 0; col < m_N; ++col)
	{
		std::sort(newRows + newColPtrs[col], newRows + newColPtrs[col + 1]);
	}

	delete[] m_ColPtrs;
	delete[] m_Rows;

	m_ColPtrs = newColPtrs;
	m_Rows = newRows;
}

bool CSC::permutationCheck(unsigned* inversePermutation)
{
	bool* check = new bool[m_N];
	std::fill(check, check + m_N, false);

	for (unsigned j = 0; j < m_N; ++j)
	{
		if (check[inversePermutation[j]] == true) return false;
		check[inversePermutation[j]] = true;
	}
	for (unsigned j = 0; j < m_N; ++j)
	{
		if (check[j] == false) return false;
	}
	delete[] check;
	
	return true;
}
