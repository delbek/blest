#pragma once

#include "Common.cuh"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_map>

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

	[[nodiscard]] inline unsigned& getN() {return m_N;}
	[[nodiscard]] inline unsigned& getNNZ() {return m_NNZ;}
	[[nodiscard]] inline unsigned*& getColPtrs() {return m_ColPtrs;}
	[[nodiscard]] inline unsigned*& getRows() {return m_Rows;}

	// metrics
	unsigned maxBandwidth();
	double averageBandwidth();
	unsigned maxProfile();
	double averageProfile();

	// orderings
	unsigned* random();
	unsigned* jackard(unsigned sliceSize);
	unsigned* gorder(unsigned windowSize);
	unsigned* gorderWithJackard(unsigned sliceSize);
	unsigned* jackardWithWindow(unsigned sliceSize, unsigned windowSize);
	unsigned* degreeSort();
	bool checkSymmetry();
	CSC* symmetrize();
	// symmetric ones
	unsigned findPseudoPeripheralNode(CSC* csc, unsigned startNode);
	unsigned* rcm();
	//
	void applyPermutation(unsigned* inversePermutation);

private:
	unsigned m_N;
	unsigned m_NNZ;
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

	std::vector<std::pair<unsigned, unsigned>> nnzs;

	unsigned i, j;
	for (unsigned iter = 0; iter < noLines; ++iter)
	{
		file >> j >> i; // read transpose
		if (!binary) file >> trash;

		--i;
		--j;

		nnzs.emplace_back(i, j);

		if (undirected && i != j)
		{
			nnzs.emplace_back(j, i);
		}
	}

	file.close();

	std::cout << "Number of vertices: " << m_N << std::endl;
	std::cout << "Number of edges: " << nnzs.size() << std::endl;
	std::cout << "Sparsity: " << std::fixed << static_cast<double>(nnzs.size()) / (static_cast<double>(m_N) * static_cast<double>(m_N)) << std::endl;

	std::sort(nnzs.begin(), nnzs.end(), [](const auto& a, const auto& b) 
	{
		if (a.second == b.second)
		{
			return a.first < b.first;
		}
		else
		{
			return a.second < b.second;
		}
	});

	m_NNZ = nnzs.size();
	m_ColPtrs = new unsigned[m_N + 1];
	m_Rows = new unsigned[m_NNZ];

	std::fill(m_ColPtrs, m_ColPtrs + m_N + 1, 0);

	for (unsigned iter = 0; iter < m_NNZ; ++iter)
	{
		++m_ColPtrs[nnzs[iter].second + 1];
		m_Rows[iter] = nnzs[iter].first;
	}

	for (unsigned j = 0; j < m_N; ++j)
	{
		m_ColPtrs[j + 1] += m_ColPtrs[j];
	}
}

CSC::~CSC()
{
	delete[] m_ColPtrs;
	delete[] m_Rows;
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

unsigned* CSC::jackard(unsigned sliceSize)
{
	unsigned* inversePermutation = new unsigned[m_N];

	std::vector<bool> permuted(m_N, false);

	std::vector<unsigned> rowMark(m_N, 0);
	unsigned epoch = 1;
	unsigned unionSize = 0;

	auto nextEpoch = [&]()
	{
		++epoch;
		unionSize = 0;
		if (epoch == std::numeric_limits<unsigned>::max())
		{
			std::fill(rowMark.begin(), rowMark.end(), 0);
			epoch = 1;
		}
	};
	auto isMarked = [&](unsigned r)
	{
		return rowMark[r] == epoch;
	};
	auto markRow = [&](unsigned r)
	{
		if (rowMark[r] != epoch)
		{
			rowMark[r] = epoch;
			++unionSize;
		}
	};

	unsigned noSliceSets = (m_N + sliceSize - 1) / sliceSize;
	for (unsigned sliceSet = 0; sliceSet < noSliceSets; ++sliceSet)
	{
		nextEpoch();

		unsigned sliceStart = sliceSet * sliceSize;
		unsigned sliceEnd = std::min(sliceStart + sliceSize, m_N);

		unsigned highestDegree = 0;
		unsigned col = UNSIGNED_MAX;
		for (unsigned j = 0; j < m_N; ++j)
		{
			if (!permuted[j])
			{
				unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
				if (deg >= highestDegree)
				{
					highestDegree = deg;
					col = j;
				}
			}
		}
		if (col == UNSIGNED_MAX) break;

		for (unsigned nnz = m_ColPtrs[col]; nnz < m_ColPtrs[col + 1]; ++nnz)
		{
			unsigned r = m_Rows[nnz];
			markRow(r);
		}
		inversePermutation[col] = sliceStart;
		permuted[col] = true;

		for (unsigned newCol = sliceStart + 1; newCol < sliceEnd; ++newCol)
		{
			double bestJackard = -1;
			unsigned bestCol;
			#pragma omp parallel num_threads(omp_get_max_threads())
			{
				double myBest = -1;
				unsigned myCol;
				#pragma omp for schedule(static)
				for (unsigned j = 0; j < m_N; ++j)
				{
					if (permuted[j]) continue;
	
					unsigned inter = 0;
					for (unsigned nnz = m_ColPtrs[j]; nnz < m_ColPtrs[j + 1]; ++nnz)
					{
						inter += isMarked(m_Rows[nnz]);
					}
					unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
					unsigned uni = unionSize + deg - inter;
					double currentJackard = (uni == 0) ? 1 : static_cast<double>(inter) / static_cast<double>(uni);
	
					if (currentJackard > myBest)
					{
						myBest = currentJackard;
						myCol = j;
					}
				}
				#pragma omp critical
				{
					if (myBest > bestJackard)
					{
						bestJackard = myBest;
						bestCol = myCol;
					}
				}
			}
			if (bestJackard == -1) break;

			for (unsigned nnz = m_ColPtrs[bestCol]; nnz < m_ColPtrs[bestCol + 1]; ++nnz)
			{
				unsigned r = m_Rows[nnz];
				markRow(r);
			}
			inversePermutation[bestCol] = newCol;
			permuted[bestCol] = true;
		}
	}
	applyPermutation(inversePermutation);

	return inversePermutation;
}

unsigned* CSC::gorder(unsigned windowSize)
{
	Gorder::Graph g;
	g.setFilename("gorder");
	g.readGraph(m_N, m_NNZ, m_ColPtrs, m_Rows);
	g.Transform();
	std::vector<int> order;
	g.GorderGreedy(order, windowSize);

	unsigned* inversePermutation = new unsigned[m_N];
	std::copy(order.data(), order.data() + m_N, inversePermutation);

	applyPermutation(inversePermutation);

	return inversePermutation;
}

unsigned* CSC::gorderWithJackard(unsigned sliceSize)
{
	unsigned windowSize = sliceSize * 33168;

	Gorder::Graph g;
	g.setFilename("gorder");
	g.readGraph(m_N, m_NNZ, m_ColPtrs, m_Rows);
	g.Transform();
	std::vector<int> order;
	g.GorderGreedy(order, windowSize);

	unsigned* inversePermutation = new unsigned[m_N];
	std::copy(order.data(), order.data() + m_N, inversePermutation);
	applyPermutation(inversePermutation);
	unsigned* inversePermutation2 = this->jackardWithWindow(sliceSize, windowSize);

	unsigned* chained = chainPermutations(m_N, inversePermutation, inversePermutation2);
	return chained;
}

unsigned* CSC::jackardWithWindow(unsigned sliceSize, unsigned windowSize)
{
	assert(windowSize % sliceSize == 0);

	unsigned* inversePermutation = new unsigned[m_N];

	unsigned noWindows = (m_N + windowSize - 1) / windowSize;
	#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
	for (unsigned window = 0; window < noWindows; ++window)
	{
		unsigned windowStart = window * windowSize;
		unsigned windowEnd = std::min(windowStart + windowSize, m_N);
		unsigned windowLength = windowEnd - windowStart;

		std::vector<bool> permuted(m_N, false);

		std::vector<unsigned> rowMark(m_N, 0);
		unsigned epoch = 1;
		unsigned unionSize = 0;
	
		auto nextEpoch = [&]()
		{
			++epoch;
			unionSize = 0;
			if (epoch == std::numeric_limits<unsigned>::max())
			{
				std::fill(rowMark.begin(), rowMark.end(), 0);
				epoch = 1;
			}
		};
		auto isMarked = [&](unsigned r)
		{
			return rowMark[r] == epoch;
		};
		auto markRow = [&](unsigned r)
		{
			if (rowMark[r] != epoch)
			{
				rowMark[r] = epoch;
				++unionSize;
			}
		};

		unsigned noSliceSets = (windowLength + sliceSize - 1) / sliceSize;
		for (unsigned sliceSet = 0; sliceSet < noSliceSets; ++sliceSet)
		{
			nextEpoch();
	
			unsigned sliceStart = windowStart + sliceSet * sliceSize;
			unsigned sliceEnd = std::min(sliceStart + sliceSize, windowEnd);
	
			unsigned highestDegree = 0;
			unsigned col = UNSIGNED_MAX;
			for (unsigned j = windowStart; j < windowEnd; ++j)
			{
				if (!permuted[j])
				{
					unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
					if (deg >= highestDegree)
					{
						highestDegree = deg;
						col = j;
					}
				}
			}
			if (col == UNSIGNED_MAX) break;
	
			for (unsigned nnz = m_ColPtrs[col]; nnz < m_ColPtrs[col + 1]; ++nnz)
			{
				unsigned r = m_Rows[nnz];
				markRow(r);
			}
			inversePermutation[col] = sliceStart;
			permuted[col] = true;
	
			for (unsigned newCol = sliceStart + 1; newCol < sliceEnd; ++newCol)
			{
				double bestJackard = -1;
				unsigned bestCol;
				for (unsigned j = windowStart; j < windowEnd; ++j)
				{
					if (permuted[j]) continue;

					unsigned inter = 0;
					for (unsigned nnz = m_ColPtrs[j]; nnz < m_ColPtrs[j + 1]; ++nnz)
					{
						inter += isMarked(m_Rows[nnz]);
					}
					unsigned deg = m_ColPtrs[j + 1] - m_ColPtrs[j];
					unsigned uni = unionSize + deg - inter;
					double currentJackard = (uni == 0) ? 1 : static_cast<double>(inter) / static_cast<double>(uni);

					if (currentJackard > bestJackard)
					{
						bestJackard = currentJackard;
						bestCol = j;
					}
				}
				if (bestJackard == -1) break;
	
				for (unsigned nnz = m_ColPtrs[bestCol]; nnz < m_ColPtrs[bestCol + 1]; ++nnz)
				{
					unsigned r = m_Rows[nnz];
					markRow(r);
				}
				inversePermutation[bestCol] = newCol;
				permuted[bestCol] = true;
			}
		}
	}
	applyPermutation(inversePermutation);

	return inversePermutation;
}

unsigned* CSC::degreeSort()
{
	std::vector<std::pair<unsigned, unsigned>> degrees(m_N);

	for (unsigned j = 0; j < m_N; ++j)
	{
		degrees[j].first = j;
		degrees[j].second = m_ColPtrs[j + 1] - m_ColPtrs[j];
	}
	std::sort(degrees.begin(), degrees.end(), [](const auto& a, const auto& b)
	{
		if (a.second != b.second) return a.second > b.second;
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
			int u = q[qs++];
			for (unsigned ptr = ptrs[u]; ptr < ptrs[u + 1]; ++ptr)
			{
				int v = inds[ptr];
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

void CSC::applyPermutation(unsigned* inversePermutation)
{
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
