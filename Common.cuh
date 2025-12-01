#pragma once

#include <iostream>
#include "omp.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <limits>
#include <fstream>
#include <type_traits>
#include <cuda/pipeline>
#include <vector>
#include <algorithm>
using namespace cooperative_groups;

#define DIRECTION_SWITCHING_CONSTANT 1

#define M 8
#define K 128
#define MASK unsigned
constexpr unsigned MASK_BITS = sizeof(MASK) * 8;
#define WARP_SIZE 32
#define UNSIGNED_MAX 4294967295

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ __forceinline__ void m8n8k128(unsigned* const __restrict__ fragC, const unsigned& fragA, const unsigned& fragB)
{
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"
                 " { %0, %1 }, "
                 " { %2 }, "
                 " { %3 }, "
                 " { %4, %5 };"
                 : "+r"(fragC[0]), "+r"(fragC[1])
                 : "r"(fragA), "r"(fragB), "r"(fragC[0]), "r"(fragC[1]));
}

template <class T>
static void fileFlush(std::ofstream& file, T el)
{
    file << "'" << el << '\t';
}

static unsigned* chainPermutations(unsigned n, unsigned* perm1, unsigned* perm2)
{
    unsigned* chained = new unsigned[n];
    for (unsigned i = 0; i < n; ++i)
    {
        chained[i] = perm2[perm1[i]];
    }
    delete[] perm1;
    delete[] perm2;
    return chained;
}

static bool isSocialDegreeDistribution(const std::vector<unsigned> &deg)
{
    unsigned n = static_cast<unsigned>(deg.size());
    if (n == 0)
    {
        return false;
    }

    unsigned maxDeg = 0;
    unsigned long long sumDeg = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned d = deg[i];
        sumDeg += static_cast<unsigned long long>(d);
        if (d > maxDeg)
        {
            maxDeg = d;
        }
    }

    if (maxDeg == 0 || sumDeg == 0)
    {
        return false;
    }

    std::vector<unsigned> sorted = deg;
    std::sort(sorted.begin(), sorted.end());

    auto shareTop = [&](double frac) -> double
    {
        double dn = static_cast<double>(n);
        double raw = frac * dn;
        double ceilVal = std::ceil(raw);
        unsigned k = static_cast<unsigned>(ceilVal);
        if (k == 0)
        {
            k = 1;
        }
        if (k > n)
        {
            k = n;
        }

        unsigned long long sumTop = 0;
        for (unsigned i = 0; i < k; ++i)
        {
            unsigned idx = n - 1 - i;
            sumTop += static_cast<unsigned long long>(sorted[idx]);
        }

        double sTop = static_cast<double>(sumTop);
        double sDeg = static_cast<double>(sumDeg);
        return sTop / sDeg;
    };

    double shareTop1 = shareTop(0.01);
    double shareTop10 = shareTop(0.10);

    bool heavyTailByShares = (shareTop1 >= 0.05 && shareTop10 >= 0.40);

    std::vector<unsigned> freq(maxDeg + 1, 0);
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned d = deg[i];
        ++freq[d];
    }

    unsigned k_min = 5;
    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(maxDeg);
    ys.reserve(maxDeg);

    for (unsigned k = k_min; k <= maxDeg; ++k)
    {
        unsigned c = freq[k];
        if (c != 0)
        {
            double x = std::log(static_cast<double>(k));
            double y = std::log(static_cast<double>(c));
            xs.push_back(x);
            ys.push_back(y);
        }
    }

    unsigned mPoints = static_cast<unsigned>(xs.size());
    bool powerLawTail = false;

    if (mPoints >= 3u)
    {
        double sumx = 0.0;
        double sumy = 0.0;
        double sumxy = 0.0;
        double sumxx = 0.0;

        for (unsigned i = 0; i < mPoints; ++i)
        {
            double x = xs[i];
            double y = ys[i];
            sumx += x;
            sumy += y;
            sumxy += x * y;
            sumxx += x * x;
        }

        double nd = static_cast<double>(mPoints);
        double denom = nd * sumxx - sumx * sumx;

        if (denom > 0.0)
        {
            double slope = (nd * sumxy - sumx * sumy) / denom;
            double intercept = (sumy - slope * sumx) / nd;

            double yMean = sumy / nd;
            double ssTot = 0.0;
            double ssRes = 0.0;

            for (unsigned i = 0; i < mPoints; ++i)
            {
                double y = ys[i];
                double y_hat = intercept + slope * xs[i];
                double diffTot = y - yMean;
                double diffRes = y - y_hat;
                ssTot += diffTot * diffTot;
                ssRes += diffRes * diffRes;
            }

            if (ssTot > 0.0)
            {
                double r2 = 1.0 - ssRes / ssTot;
                double gamma = -slope;

                double minR2 = 0.70;
                double minGamma = 1.0;
                double maxGamma = 5.0;

                if (r2 >= minR2 && gamma >= minGamma && gamma <= maxGamma)
                {
                    powerLawTail = true;
                }
            }
        }
    }

    if (heavyTailByShares || powerLawTail)
    {
        return true;
    }

    return false;
}
