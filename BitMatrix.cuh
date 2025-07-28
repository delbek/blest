#ifndef BITMATRIX_CUH
#define BITMATRIX_CUH

#include "Common.cuh"

class BitMatrix
{
public:
    BitMatrix() = default;
    BitMatrix(const BitMatrix& other) = default;
    BitMatrix(BitMatrix&& other) noexcept = default;
    BitMatrix& operator=(const BitMatrix& other) = default;
    BitMatrix& operator=(BitMatrix&& other) noexcept = default;
    virtual ~BitMatrix() = default;

    virtual void save(std::string filename) = 0;
};

#endif
